"""TinyAudio backbone.

A single-stream diffusion transformer: audio latents and text tokens are
concatenated into one sequence and processed by a stack of attention blocks.
The AdaLN shift/scale and gate projections are shared across all layers, with a
learned per-layer embedding added to the shared condition. The global text
condition is obtained with a learned pooling over the text tokens.

Two variants are provided:

* ``TinyAudio``   - flow-matching model (multi-step Euler sampling).
* ``TinyAudioMF`` - MeanFlow model (one-step or few-step sampling).
"""
import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from tinyaudio.model.embeddings import TimestepEmbedder
from tinyaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from tinyaudio.model.rotary_embeddings import compute_rope_rotations
from tinyaudio.model.transformer_layers import FinalBlock, SingleStreamBlock

log = logging.getLogger()

_FLEX_CFG_PREFIXES = (
    'omega_embed.',
    'cfg_t_start_embed.',
    'cfg_t_end_embed.',
    'omega_tokens',
    't_min_tokens',
    't_max_tokens',
)


@dataclass
class PreprocessedConditions:
    text_f: torch.Tensor
    text_f_c: Optional[torch.Tensor]


def _basic_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class TinyAudio(nn.Module):
    """Flow-matching TinyAudio network."""

    def __init__(self,
                 *,
                 latent_dim: int,
                 text_dim: int,
                 text_c_dim: int,
                 hidden_dim: int,
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 empty_string_feat_c: Optional[torch.Tensor] = None,
                 empty_string_padding_mask: Optional[torch.Tensor] = None,
                 use_rope: bool = True,
                 block_kernel_size: int = 3,
                 block_padding: int = 1,
                 ffn_kernel_size: Optional[int] = None,
                 ffn_padding: Optional[int] = None,
                 audio_input_conv_kernel_size: int = 3,
                 audio_input_conv_padding: int = 1,
                 use_learned_text_pad_token: bool = True) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.depth = depth
        self.use_rope = use_rope
        self.use_learned_text_pad_token = use_learned_text_pad_token

        # text projections
        self.text_input_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )
        self.text_cond_proj = nn.Sequential(
            nn.Linear(text_c_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )

        self.final_layer = FinalBlock(hidden_dim, latent_dim)
        self.t_embed = TimestepEmbedder(hidden_dim, frequency_embedding_size=256, max_period=10000)

        # latent statistics / empty-prompt features (filled from checkpoint or feature extractor)
        if latent_mean is None:
            assert latent_std is None
            latent_mean = torch.full((latent_dim, ), float('nan'))
            latent_std = torch.full((latent_dim, ), float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        if empty_string_feat_c is None:
            empty_string_feat_c = torch.zeros((text_c_dim, ))
        assert empty_string_feat.shape[-1] == text_dim
        assert empty_string_feat_c.shape[-1] == text_c_dim

        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)
        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_string_feat_c = nn.Parameter(empty_string_feat_c, requires_grad=False)
        if empty_string_padding_mask is None:
            empty_string_padding_mask = torch.zeros(text_seq_len, dtype=torch.bool)
        else:
            empty_string_padding_mask = empty_string_padding_mask.to(dtype=torch.bool).view(-1)
        self.register_buffer('_empty_string_padding_mask', empty_string_padding_mask, persistent=False)
        if self.use_learned_text_pad_token:
            self.learned_text_pad_token = nn.Parameter(torch.zeros(text_dim))

        self.apply(_basic_init)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        if self.use_rope:
            head_dim = hidden_dim // num_heads
            max_latent_len = 2048  # ~47.6 s of audio at 43 Hz
            latent_rot = compute_rope_rotations(max_latent_len, head_dim, 10000, freq_scaling=1.0)
            text_rot = compute_rope_rotations(text_seq_len, head_dim, 10000, freq_scaling=1.0)
            self.register_buffer('_latent_rot_buffer', latent_rot, persistent=False)
            self.register_buffer('_text_rot_buffer', text_rot, persistent=False)

        # audio input projection
        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(latent_dim,
                              hidden_dim,
                              kernel_size=audio_input_conv_kernel_size,
                              padding=audio_input_conv_padding),
            nn.SELU(),
            ConvMLP(hidden_dim,
                    hidden_dim * 4,
                    kernel_size=audio_input_conv_kernel_size,
                    padding=audio_input_conv_padding),
        )
        self.audio_input_proj.apply(_basic_init)

        # AdaLN modulation shared across layers, with per-layer embeddings
        self.shared_shift_scale_modulation = nn.Sequential(nn.SiLU(),
                                                           nn.Linear(hidden_dim, 4 * hidden_dim, bias=True))
        self.shared_gate_modulation = nn.Sequential(nn.SiLU(),
                                                    nn.Linear(hidden_dim, 2 * hidden_dim, bias=True))
        self.shared_adaln_layer_emb = nn.Parameter(torch.zeros(depth, hidden_dim))
        self.shared_gate_layer_emb = nn.Parameter(torch.zeros(depth, hidden_dim))
        self.single_stream_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_dim,
                              num_heads,
                              mlp_ratio=mlp_ratio,
                              kernel_size=block_kernel_size,
                              padding=block_padding,
                              ffn_kernel_size=ffn_kernel_size,
                              ffn_padding=ffn_padding) for _ in range(depth)
        ])
        nn.init.constant_(self.shared_shift_scale_modulation[-1].weight, 0)
        nn.init.constant_(self.shared_shift_scale_modulation[-1].bias, 0)
        nn.init.constant_(self.shared_gate_modulation[-1].weight, 0)
        nn.init.constant_(self.shared_gate_modulation[-1].bias, 0)

        # token-to-global pooling for the AdaLN text condition
        self.text_token_pool = nn.Linear(text_seq_len, 1, bias=True)
        nn.init.constant_(self.text_token_pool.weight, 1.0 / text_seq_len)
        nn.init.constant_(self.text_token_pool.bias, 0.0)

    # ------------------------------------------------------------------ utils
    def update_seq_lengths(self, latent_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.latent_std).add_(self.latent_mean)

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    # ------------------------------------------------------------- conditions
    def apply_text_padding_token(self,
                                 text_f: torch.Tensor,
                                 text_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_learned_text_pad_token or text_padding_mask is None:
            return text_f
        pad_mask = text_padding_mask.to(device=text_f.device, dtype=torch.bool).unsqueeze(-1)
        learned_pad = self.learned_text_pad_token.to(device=text_f.device, dtype=text_f.dtype).view(1, 1, -1)
        return torch.where(pad_mask, learned_pad, text_f)

    def _pool_text_features(self, text_f: torch.Tensor) -> torch.Tensor:
        if text_f.shape[1] != self._text_seq_len:
            raise ValueError(f'text_token_pool expects seq_len={self._text_seq_len}, got {text_f.shape[1]}')
        return self.text_token_pool(text_f.transpose(1, 2)).squeeze(-1)

    def preprocess_conditions(self,
                              text_f: torch.Tensor,
                              text_f_c: torch.Tensor,
                              text_padding_mask: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        """Cache computations that do not depend on the latent / time step."""
        text_f = self.apply_text_padding_token(text_f, text_padding_mask)
        text_f_c = self._pool_text_features(text_f)
        text_f_c = self.text_cond_proj(text_f_c)
        text_f = self.text_input_proj(text_f)
        return PreprocessedConditions(text_f=text_f, text_f_c=text_f_c)

    def get_empty_string_sequence(self, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1), \
            self.empty_string_feat_c.unsqueeze(0).expand(bs, -1)

    def get_empty_conditions(self,
                             bs: int,
                             *,
                             negative_text_features: Optional[tuple] = None) -> PreprocessedConditions:
        if negative_text_features is not None:
            if len(negative_text_features) == 3:
                empty_feat, empty_feat_c, negative_padding_mask = negative_text_features
            else:
                empty_feat, empty_feat_c = negative_text_features
                negative_padding_mask = None
        else:
            empty_feat, empty_feat_c = self.get_empty_string_sequence(1)
            negative_padding_mask = self._empty_string_padding_mask.unsqueeze(0)

        conditions = self.preprocess_conditions(empty_feat, empty_feat_c, text_padding_mask=negative_padding_mask)
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)
        return conditions

    # --------------------------------------------------------------- backbone
    def _joint_rot(self, prefix_len: int, latent_len: int, text_len: int, device: torch.device):
        if not self.use_rope:
            return None
        latent_rot = self._latent_rot_buffer[:, :latent_len]
        text_rot = self._text_rot_buffer[:, :text_len]
        if prefix_len == 0:
            return torch.cat([latent_rot, text_rot], dim=1)
        head_dim = self.hidden_dim // self.num_heads
        prefix_rot = compute_rope_rotations(prefix_len, head_dim, 10000, freq_scaling=1.0, device=device)
        return torch.cat([prefix_rot, latent_rot, text_rot], dim=1)

    def _run_blocks(self, joint_features: torch.Tensor, joint_rot: Optional[torch.Tensor],
                    global_c: torch.Tensor) -> torch.Tensor:
        for layer_idx, block in enumerate(self.single_stream_blocks):
            shift_scale_c = global_c + self.shared_adaln_layer_emb[layer_idx].view(1, 1, -1)
            gate_c = global_c + self.shared_gate_layer_emb[layer_idx].view(1, 1, -1)
            joint_features = block(joint_features, joint_rot, self.shared_shift_scale_modulation(shift_scale_c),
                                   self.shared_gate_modulation(gate_c))
        return joint_features

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions) -> torch.Tensor:
        audio_features = self.audio_input_proj(latent)
        text_features = conditions.text_f
        joint_features = torch.cat([audio_features, text_features], dim=1)
        joint_rot = self._joint_rot(0, audio_features.shape[1], text_features.shape[1], latent.device)

        global_c = self.t_embed(t).unsqueeze(1) + conditions.text_f_c.unsqueeze(1)
        joint_features = self._run_blocks(joint_features, joint_rot, global_c)
        return self.final_layer(joint_features[:, :audio_features.shape[1]], global_c)

    def forward(self,
                latent: torch.Tensor,
                text_f: torch.Tensor,
                text_f_c: torch.Tensor,
                t: torch.Tensor,
                text_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        latent: (B, N, C), text_f: (B, T, D), t: (B,)
        """
        conditions = self.preprocess_conditions(text_f, text_f_c, text_padding_mask=text_padding_mask)
        return self.predict_flow(latent, t, conditions)

    def ode_wrapper(self, t: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength: float) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)
        return (cfg_strength * self.predict_flow(latent, t, conditions) +
                (1 - cfg_strength) * self.predict_flow(latent, t, empty_conditions))

    def load_weights(self, src_dict) -> None:
        for key in ('t_embed.freqs', 'latent_rot', 'text_rot', 'joint_rot', '_joint_rot_buffer'):
            src_dict.pop(key, None)
        self.load_state_dict(src_dict, strict=True)


class TinyAudioMF(TinyAudio):
    """MeanFlow TinyAudio network.

    Adds an embedding of the interval end point ``r``. Optionally, the guidance
    scale and guidance interval can be fed to the network as prefix tokens
    (``use_flexible_cfg_tokens``), so that guided one-step sampling needs a
    single forward pass.
    """

    def __init__(self,
                 *args,
                 use_flexible_cfg_tokens: bool = False,
                 num_cfg_tokens: int = 4,
                 num_interval_tokens: int = 2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_flexible_cfg_tokens = use_flexible_cfg_tokens
        self.num_cfg_tokens = num_cfg_tokens if use_flexible_cfg_tokens else 0
        self.num_interval_tokens = num_interval_tokens if use_flexible_cfg_tokens else 0

        self.r_embed = TimestepEmbedder(self.hidden_dim, frequency_embedding_size=256, max_period=10000)
        self._init_zero_embedder(self.r_embed)

        if self.use_flexible_cfg_tokens:
            self.omega_embed = TimestepEmbedder(self.hidden_dim, frequency_embedding_size=256, max_period=10000)
            self.cfg_t_start_embed = TimestepEmbedder(self.hidden_dim,
                                                      frequency_embedding_size=256,
                                                      max_period=10000)
            self.cfg_t_end_embed = TimestepEmbedder(self.hidden_dim,
                                                    frequency_embedding_size=256,
                                                    max_period=10000)
            for embed in (self.omega_embed, self.cfg_t_start_embed, self.cfg_t_end_embed):
                self._init_zero_embedder(embed)
            self.omega_tokens = nn.Parameter(torch.zeros(self.num_cfg_tokens, self.hidden_dim))
            self.t_min_tokens = nn.Parameter(torch.zeros(self.num_interval_tokens, self.hidden_dim))
            self.t_max_tokens = nn.Parameter(torch.zeros(self.num_interval_tokens, self.hidden_dim))

    @staticmethod
    def _init_zero_embedder(embed: TimestepEmbedder) -> None:
        nn.init.normal_(embed.mlp[0].weight, std=0.02)
        nn.init.constant_(embed.mlp[0].bias, 0)
        nn.init.constant_(embed.mlp[2].weight, 0)
        nn.init.constant_(embed.mlp[2].bias, 0)

    @staticmethod
    def _expand(value: Optional[Union[torch.Tensor, float]], *, batch_size: int, device: torch.device,
                dtype: torch.dtype, default: float) -> torch.Tensor:
        if value is None:
            return torch.full((batch_size, ), default, device=device, dtype=dtype)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch_size)
        return value

    def _build_prefix_tokens(self, t: torch.Tensor, omega, t_min, t_max) -> torch.Tensor:
        batch_size = t.shape[0]
        if not self.use_flexible_cfg_tokens:
            return t.new_empty(batch_size, 0, self.hidden_dim)
        kw = dict(batch_size=batch_size, device=t.device, dtype=t.dtype)
        omega = self._expand(omega, default=1.0, **kw).clamp_min(1.0)
        t_min = self._expand(t_min, default=0.0, **kw)
        t_max = self._expand(t_max, default=1.0, **kw)

        omega_input = 1.0 - omega.reciprocal().clamp(min=1e-6)
        omega_tokens = self.omega_tokens.unsqueeze(0) + self.omega_embed(omega_input).unsqueeze(1)
        t_min_tokens = self.t_min_tokens.unsqueeze(0) + self.cfg_t_start_embed(t_min).unsqueeze(1)
        t_max_tokens = self.t_max_tokens.unsqueeze(0) + self.cfg_t_end_embed(t_max).unsqueeze(1)
        return torch.cat([omega_tokens, t_min_tokens, t_max_tokens], dim=1)

    def predict_flow(self,
                     latent: torch.Tensor,
                     t: torch.Tensor,
                     r: torch.Tensor,
                     conditions: PreprocessedConditions,
                     omega=None,
                     t_min=None,
                     t_max=None) -> torch.Tensor:
        audio_features = self.audio_input_proj(latent)
        text_features = conditions.text_f
        prefix_tokens = self._build_prefix_tokens(t, omega, t_min, t_max)
        prefix_len = prefix_tokens.shape[1]
        joint_features = torch.cat([prefix_tokens, audio_features, text_features], dim=1)
        joint_rot = self._joint_rot(prefix_len, audio_features.shape[1], text_features.shape[1],
                                    joint_features.device)

        global_c = (self.t_embed(t).unsqueeze(1) + self.r_embed(r).unsqueeze(1) +
                    conditions.text_f_c.unsqueeze(1))
        joint_features = self._run_blocks(joint_features, joint_rot, global_c)
        audio_features = joint_features[:, prefix_len:prefix_len + audio_features.shape[1]]
        return self.final_layer(audio_features, global_c)

    def forward(self,
                latent: torch.Tensor,
                text_f: torch.Tensor,
                text_f_c: torch.Tensor,
                r: torch.Tensor,
                t: torch.Tensor,
                omega=None,
                t_min=None,
                t_max=None,
                text_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        conditions = self.preprocess_conditions(text_f, text_f_c, text_padding_mask=text_padding_mask)
        return self.predict_flow(latent, t, r, conditions, omega=omega, t_min=t_min, t_max=t_max)

    def ode_wrapper(self,
                    t: torch.Tensor,
                    r: torch.Tensor,
                    latent: torch.Tensor,
                    conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions,
                    cfg_strength: float,
                    omega=None,
                    t_min=None,
                    t_max=None) -> torch.Tensor:
        bs = len(latent)
        kw = dict(batch_size=bs, device=latent.device, dtype=latent.dtype)
        t = t * torch.ones(bs, device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(bs, device=latent.device, dtype=latent.dtype)

        if self.use_flexible_cfg_tokens:
            # guidance is handled inside the network: a single forward pass
            omega = self._expand(omega if omega is not None else cfg_strength, default=cfg_strength,
                                 **kw).clamp_min(1.0)
            t_min = self._expand(t_min, default=0.0, **kw)
            t_max = self._expand(t_max, default=1.0, **kw)
            return self.predict_flow(latent, t, r, conditions, omega=omega, t_min=t_min, t_max=t_max)

        if omega is None and t_min is None and t_max is None and cfg_strength < 1.0:
            return self.predict_flow(latent, t, r, conditions)

        # interval classifier-free guidance with two forward passes
        omega = self._expand(omega if omega is not None else cfg_strength, default=cfg_strength,
                             **kw).clamp_min(1.0)
        t_min = self._expand(t_min, default=0.0, **kw)
        t_max = self._expand(t_max, default=1.0, **kw)
        w = torch.where((t >= t_min) & (t <= t_max), omega, torch.ones_like(omega))
        flow_cond = self.predict_flow(latent, t, r, conditions)
        flow_uncond = self.predict_flow(latent, t, r, empty_conditions)
        scale = (1.0 - 1.0 / w).view(bs, *([1] * (latent.ndim - 1)))
        return flow_uncond + scale * (flow_cond - flow_uncond)

    def load_weights(self, src_dict) -> None:
        src_dict.pop('r_embed.freqs', None)
        if not self.use_flexible_cfg_tokens:
            for key in list(src_dict.keys()):
                if key.startswith(_FLEX_CFG_PREFIXES):
                    del src_dict[key]
        current_state = self.state_dict()
        for key in current_state:
            if key not in src_dict and key.startswith(('r_embed.', ) + _FLEX_CFG_PREFIXES):
                src_dict[key] = current_state[key]
        super().load_weights(src_dict)


_TINYAUDIO_ARCH = dict(
    latent_dim=64,
    hidden_dim=384,  # 8 heads x 48
    depth=18,
    num_heads=8,
    mlp_ratio=2,
    latent_seq_len=431,  # 10 s at 44.1 kHz with hop 1024
    block_kernel_size=3,
    block_padding=1,
    ffn_kernel_size=1,
    ffn_padding=0,
    audio_input_conv_kernel_size=3,
    audio_input_conv_padding=1,
)


def tinyaudio(**kwargs) -> TinyAudio:
    return TinyAudio(**_TINYAUDIO_ARCH, **kwargs)


def tinyaudio_mf(**kwargs) -> TinyAudioMF:
    return TinyAudioMF(**_TINYAUDIO_ARCH, **kwargs)


_MODELS = {
    'tinyaudio': tinyaudio,
    'tinyaudio_mf': tinyaudio_mf,
}


def get_model(name: str, **kwargs) -> TinyAudio:
    if name not in _MODELS:
        raise ValueError(f'Unknown model name: {name}. Available: {list(_MODELS)}')
    log.info(f'Building model: {name}')
    return _MODELS[name](**kwargs)


def is_meanflow_model(name: str) -> bool:
    return name.endswith('_mf')


def build_model_from_cfg(cfg, **kwargs) -> TinyAudio:
    """Build the network described by a TinyAudio config.

    Extra keyword arguments (e.g. ``latent_mean``, ``empty_string_feat``) are forwarded to the constructor.
    """
    model_kwargs = dict(text_dim=cfg.text_dim,
                        text_c_dim=cfg.text_c_dim,
                        use_rope=cfg.get('use_rope', True),
                        use_learned_text_pad_token=cfg.get('use_learned_text_pad_token', True))
    if is_meanflow_model(cfg.model):
        mf_cfg = cfg.get('meanflow', None) or {}
        model_kwargs.update(use_flexible_cfg_tokens=bool(mf_cfg.get('flexible_cfg_tokens', False)),
                            num_cfg_tokens=int(mf_cfg.get('num_cfg_tokens', 4)),
                            num_interval_tokens=int(mf_cfg.get('num_interval_tokens', 2)))
    model_kwargs.update(kwargs)
    return get_model(cfg.model, **model_kwargs)
