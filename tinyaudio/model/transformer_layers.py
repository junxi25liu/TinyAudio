from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from tinyaudio.model.rotary_embeddings import apply_rope
from tinyaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP


_DISABLE_FLASH_ATTENTION = False


@contextmanager
def disable_flash_attention():
    global _DISABLE_FLASH_ATTENTION
    prev = _DISABLE_FLASH_ATTENTION
    _DISABLE_FLASH_ATTENTION = True
    try:
        yield
    finally:
        _DISABLE_FLASH_ATTENTION = prev


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift  # scale is actually the add term for x (res connect for modulation)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # flash attention is not compatible with JVP calculation
    if q.is_cuda and _DISABLE_FLASH_ATTENTION:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(q, k, v)
    else:
        out = F.scaled_dot_product_attention(q, k, v)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out


class SelfAttention(nn.Module):

    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

        self.split_into_heads = Rearrange('b n (h d j) -> b h n d j',
                                          h=nheads,
                                          d=dim // nheads,
                                          j=3)

    def pre_attention(  # get qkv for input x, apply rotary pos embedding if needed
            self, x: torch.Tensor,
            rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)  # chunk: split the input into 3 components
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x)
        out = attention(q, k, v)
        return out


class SingleStreamBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 nhead: int,
                 mlp_ratio: float = 4.0,
                 kernel_size: int = 7,
                 padding: int = 3,
                 ffn_kernel_size: Optional[int] = None,
                 ffn_padding: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)

        if kernel_size == 1:
            self.linear1 = nn.Linear(dim, dim)
        else:
            self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        if ffn_kernel_size is None:
            ffn_kernel_size = kernel_size
        if ffn_padding is None:
            ffn_padding = padding

        if ffn_kernel_size == 1:
            self.ffn = MLP(dim, int(dim * mlp_ratio))
        else:
            self.ffn = ConvMLP(dim,
                               int(dim * mlp_ratio),
                               kernel_size=ffn_kernel_size,
                               padding=ffn_padding)

    def pre_attention(self,
                      x: torch.Tensor,
                      rot: Optional[torch.Tensor],
                      shared_modulation: torch.Tensor,
                      shared_gate_modulation: torch.Tensor):
        shift_msa, scale_msa, shift_mlp, scale_mlp = shared_modulation.chunk(4, dim=-1)
        gate_msa, gate_mlp = shared_gate_modulation.chunk(2, dim=-1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor]):
        gate_msa, shift_mlp, scale_mlp, gate_mlp = c
        x = x + self.linear1(attn_out) * gate_msa
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp
        return x

    def forward(self,
                x: torch.Tensor,
                rot: Optional[torch.Tensor],
                shared_modulation: torch.Tensor,
                shared_gate_modulation: torch.Tensor) -> torch.Tensor:
        x_qkv, x_conditions = self.pre_attention(x, rot, shared_modulation, shared_gate_modulation)
        attn_out = attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions)
        return x


class FinalBlock(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent
