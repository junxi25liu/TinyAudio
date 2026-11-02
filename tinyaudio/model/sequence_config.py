import dataclasses
import json
import math
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class SequenceConfig:
    duration: float
    sampling_rate: int
    hop_length: int

    @property
    def latent_seq_len(self) -> int:
        return int(math.ceil(self.duration * self.sampling_rate / self.hop_length))

    @property
    def num_audio_frames(self) -> int:
        # an integer number of latents
        return self.latent_seq_len * self.hop_length


def resolve_vae_dir(ckpt_path: str, model_tag: str = 'latest') -> Path:
    base = Path(ckpt_path).expanduser()
    for candidate in (base, base / model_tag):
        if (candidate / 'metainfo.json').exists() and (candidate / 'dac').exists():
            return candidate
    raise FileNotFoundError(f'Invalid TA-VAE checkpoint path: {ckpt_path}')


def vae_hop_length(ckpt_path: str, model_tag: str = 'latest') -> int:
    ckpt_dir = resolve_vae_dir(ckpt_path, model_tag)
    with (ckpt_dir / 'metainfo.json').open('r', encoding='utf-8') as f:
        metainfo = json.load(f)
    rates = metainfo.get('DAC', {}).get('encoder_rates')
    if not rates:
        raise ValueError(f'Cannot infer hop length from {ckpt_dir / "metainfo.json"}')
    return int(math.prod(rates))


def resolve_sequence_config_from_cfg(cfg: Any, *, duration: float | None = None) -> SequenceConfig:
    resolved_duration = float(duration if duration is not None else cfg.get('audio_max_length', 10))
    hop_length = vae_hop_length(cfg.vae_ckpt, str(cfg.get('vae_model_tag', 'latest')))
    return SequenceConfig(duration=resolved_duration,
                          sampling_rate=int(cfg.audio_sample_rate),
                          hop_length=hop_length)
