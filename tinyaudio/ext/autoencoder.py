"""TA-VAE: a 64-channel continuous audio VAE at 44.1 kHz (hop 1024, ~43 Hz)."""
import json
from pathlib import Path

import torch
import torch.nn as nn

from tinyaudio.model.sequence_config import resolve_vae_dir
from tinyaudio.model.utils.distributions import DiagonalGaussianDistribution

_CONF_DIR = Path(__file__).resolve().parent / 'dac' / 'conf'


def _load_vae(ckpt_dir: Path, use_ema: bool) -> nn.Module:
    from tinyaudio.ext.dac.model.dac import DAC

    with (ckpt_dir / 'metainfo.json').open('r', encoding='utf-8') as f:
        metainfo = json.load(f)
    dac_cfg = dict(metainfo['DAC'])
    bigvgan_conf = dac_cfg.get('bigvgan_conf')
    if isinstance(bigvgan_conf, str):
        dac_cfg['bigvgan_conf'] = str(_CONF_DIR / Path(bigvgan_conf).name)

    if use_ema:
        ckpt = torch.load(ckpt_dir / 'dac' / 'ema_state_dict.pth', map_location='cpu', weights_only=True)
        if any(k.startswith('ema_model.') for k in ckpt):
            ckpt = {k.removeprefix('ema_model.'): v for k, v in ckpt.items() if k.startswith('ema_model.')}
    else:
        ckpt = torch.load(ckpt_dir / 'dac' / 'weights.pth', map_location='cpu', weights_only=True)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith('projectors')}

    model = DAC(**dac_cfg)
    if hasattr(model, 'projectors'):
        del model.projectors
    model.load_state_dict(ckpt, strict=False)
    return model.eval()


class AutoEncoderModule(nn.Module):

    def __init__(self,
                 *,
                 vae_ckpt_path: str,
                 model_tag: str = 'latest',
                 use_ema: bool = True,
                 need_vae_encoder: bool = True):
        super().__init__()
        self.need_vae_encoder = need_vae_encoder
        self.vae = _load_vae(resolve_vae_dir(vae_ckpt_path, model_tag), use_ema)
        self.vocoder = nn.Identity()  # the VAE decodes directly to waveform

        for param in self.parameters():
            param.requires_grad = False
        if not need_vae_encoder:
            # decoder-only inference
            for attr in ('encoder', 'pre_block', 'fc_mu', 'fc_var', 'projectors'):
                if hasattr(self.vae, attr):
                    delattr(self.vae, attr)

    @torch.inference_mode()
    def encode_waveform(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        if not self.need_vae_encoder:
            raise RuntimeError('VAE encoder is disabled (need_vae_encoder=False).')
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f'Expected waveform shape [B, C, T], got {tuple(x.shape)}')
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        param = next(self.vae.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        audio_data = self.vae.preprocess(x, self.vae.sample_rate)
        _, mu, log_var, _ = self.vae.encode(audio_data)
        params = torch.cat([mu.transpose(1, 2), log_var.transpose(1, 2)], dim=1)
        return DiagonalGaussianDistribution(params)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, T); the VAE decoder expects (B, T, C)
        return self.vae.decode(z.transpose(1, 2))

    @torch.inference_mode()
    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return spec
