import json
import math
from typing import List
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .attn_proj import AttnProjection
from .base import CodecMixin
from .bigvgan import BigVGAN
from .l3ac_backbone import L3ACDecoder
from .l3ac_backbone import L3ACEncoder
from .regulator import InterpolateRegulator
from tinyaudio.ext.dac.model.utils import AttrDict
from tinyaudio.ext.dac.model.utils import make_pad_mask
from tinyaudio.ext.dac.model.utils import masked_mean
from tinyaudio.ext.dac.nn.layers import Snake1d
from tinyaudio.ext.dac.nn.layers import WNConv1d
from tinyaudio.ext.dac.nn.layers import WNConvTranspose1d


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DAC(nn.Module, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        vae_dim: Union[int, list] = 8,
        sample_rate: int = 44100,
        distill: bool = False,
        distill_hidden_dim: int = 768,
        decoder_type: str = "dac",  # bigvgan | dac
        backbone_type: str = "dac",  # dac | l3ac
        attn_proj: bool = False,
        post_vae_block: bool = False,
        bigvgan_conf: str = "conf/bigvgan_conf/bigvgan_v2_24khz_100band_256x.json",
        sampling_ratios: list = [0, 1],
        l3ac_feature_dim: int = 32,
        l3ac_compress_rates: List[int] = [4, 4, 5, 5],
        l3ac_encoder_dims: List[int] = [32, 64, 128, 256, 512],
        l3ac_encoder_depths: List[int] = [1, 1, 2, 2, 2],
        l3ac_decode_rates: List[int] = [5, 5, 4, 4],
        l3ac_decoder_dims: List[int] = [512, 256, 128, 64, 32],
        l3ac_decoder_depths: List[int] = [2, 2, 2, 1, 1],
        l3ac_use_norm: bool = True,
        l3ac_use_snake_act: bool = True,
        l3ac_decoder_last_layer: str = "legacy",
        **kwargs
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.decoder_type = decoder_type
        self.backbone_type = backbone_type

        self.vae_dim = vae_dim
        self.attn_proj = attn_proj
        self.post_vae_block = post_vae_block

        if self.backbone_type == "l3ac" or self.decoder_type == "l3ac":
            self.backbone_type = "l3ac"
            if latent_dim is None:
                latent_dim = l3ac_feature_dim
            self.latent_dim = latent_dim
            self.hop_length = np.prod(l3ac_compress_rates)
            self.encoder = L3ACEncoder(
                feature_dim=latent_dim,
                strides=tuple(l3ac_compress_rates),
                depths=tuple(l3ac_encoder_depths),
                dims=tuple(l3ac_encoder_dims),
                use_norm=l3ac_use_norm,
                use_snake_act=l3ac_use_snake_act,
            )
            self.decoder = L3ACDecoder(
                feature_dim=vae_dim,
                strides=tuple(l3ac_decode_rates),
                depths=tuple(l3ac_decoder_depths),
                dims=tuple(l3ac_decoder_dims),
                use_norm=l3ac_use_norm,
                use_snake_act=l3ac_use_snake_act,
                decoder_last_layer=l3ac_decoder_last_layer,
            )
        else:
            if latent_dim is None:
                latent_dim = encoder_dim * (2 ** len(encoder_rates))
            self.latent_dim = latent_dim
            self.hop_length = np.prod(encoder_rates)
            self.sample_rate = sample_rate
            self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

            self.bigvgan_conf = bigvgan_conf
            with open(self.bigvgan_conf) as f:
                data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)
            self.decoder = BigVGAN(h)

        if self.attn_proj:
            self.pre_block = AttnProjection(self.latent_dim, self.vae_dim, num_heads=8)
        elif self.latent_dim == self.vae_dim:
            self.pre_block = nn.Identity()
        else:
            self.pre_block = nn.Linear(self.latent_dim, self.vae_dim)

        self.fc_mu = nn.Linear(self.vae_dim, self.vae_dim)
        self.fc_var = nn.Linear(self.vae_dim, self.vae_dim)

        self.distill = distill
        #hard code here
        self.distill_hidden_dim = 768#distill_hidden_dim

        proj_dim = self.vae_dim * 2
        self.projectors = InterpolateRegulator(
            sampling_ratios, self.distill_hidden_dim, proj_dim, self.vae_dim
        )

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_kl_loss(self, mu, log_var):
        kl_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - (log_var.exp() + 1e-6), dim=-1
        )
        return kl_loss.mean()

    def encode(self, audio_data: torch.Tensor):
        z = self.encoder(audio_data).transpose(
            1, 2
        )  # torch.Size([72, 1024, 29]),[B x D x T] -> torch.Size([72, 29, 1024]),[B x T x D] ->vq torch.Size([72, 29, 8]),[B x D x T]
        z = self.pre_block(z)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        log_var = torch.clamp(log_var, min=-12, max=12)

        z_hat = self.reparameterize(mu, log_var)
        kl_loss = self.compute_kl_loss(mu, log_var)

        return z_hat, mu, log_var, kl_loss

    def decode(self, z: torch.Tensor):
        recon = self.decoder(z.transpose(1, 2))
        return recon

    def forward(
        self,
        audio_data: torch.Tensor,  # B, 1, T (duration)
        sample_rate: int = None,
        guidance: torch.Tensor = None,  # B, T, D
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, mu, log_var, kl_loss = self.encode(audio_data)
        proj_loss = 0.0
        if self.distill and self.training:
            guidance_lengths = [g.shape[0] for g in guidance]
            z_lens = [zi.shape[0] for zi in z]
            target_lengths = torch.tensor(guidance_lengths, device=z.device)
            z_lengths = torch.tensor(z_lens, device=z.device)
            z_mask = make_pad_mask(
                z_lengths, max_len=torch.max(target_lengths)
            )  # 16 150, padding的部分是1
            g_mask = make_pad_mask(target_lengths, max_len=torch.max(z_lengths))

            proj_g, olens = self.projectors(guidance, target_lengths, z_lengths)
            bsz, seq_len, distill_dim = (
                proj_g.shape[0],
                proj_g.shape[1],
                proj_g.shape[2],
            )
            for i, (pi, gi) in enumerate(zip(z, proj_g)):
                cos_sim = F.cosine_similarity(pi, gi, dim=-1)  # 150, 1024  # 150, 1024

                proj_loss += masked_mean(-cos_sim, ~g_mask[i])
            proj_loss = proj_loss / bsz

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "vae/kl_loss": kl_loss,
            "vae/proj_loss": proj_loss,
        }
