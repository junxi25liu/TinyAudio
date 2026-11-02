"""High-level TinyAudio inference with automatic checkpoint download.

Example:
    from tinyaudio import TinyAudioPipeline

    pipe = TinyAudioPipeline.from_pretrained("tinyaudio_mf")
    audio = pipe("A dog is barking while a car passes by")
    pipe.save(audio[0], "dog.wav")

Checkpoints are looked up in the local repository first. Missing files are downloaded
from the Hugging Face Hub and cached in the standard Hugging Face cache directory.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
from omegaconf import DictConfig

from tinyaudio.eval_utils import generate_fm, generate_mf, load_text_condition, sync_empty_conditions
from tinyaudio.model.flow_matching import FlowMatching
from tinyaudio.model.mean_flow import MeanFlow
from tinyaudio.model.networks import build_model_from_cfg, is_meanflow_model
from tinyaudio.model.sequence_config import resolve_sequence_config_from_cfg
from tinyaudio.model.utils.features_utils import FeaturesUtils

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / 'config'
HF_REPO_ID = 'junxiliu/TinyAudio'

VARIANTS = {
    'tinyaudio': dict(checkpoint='weights/tinyaudio.pth', num_steps=25),
    'tinyaudio_mf': dict(checkpoint='weights/tinyaudio_mf.pth', num_steps=1),
}


def resolve_checkpoint(path: Union[str, Path],
                       *,
                       repo_id: str = HF_REPO_ID,
                       revision: Optional[str] = None,
                       local_files_only: bool = False) -> Path:
    """Return a local path for a checkpoint file or folder, downloading it if needed.

    Absolute or existing paths are used as-is. Relative paths are resolved against the
    repository root, then fetched from ``repo_id`` with the same relative layout.
    """
    path = Path(path).expanduser()
    if path.exists():
        return path.resolve()
    if path.is_absolute():
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    local = REPO_ROOT / path
    if local.exists():
        return local

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    rel = path.as_posix()
    # Include the Hub configuration in real downloads for standard download statistics.
    kwargs = dict(repo_id=repo_id, revision=revision, allow_patterns=['config.json', rel, f'{rel}/*'])
    try:
        root = snapshot_download(local_files_only=True, **kwargs)
        target = Path(root) / rel
        if not target.exists():
            raise LocalEntryNotFoundError(rel)
    except LocalEntryNotFoundError:
        if local_files_only:
            raise FileNotFoundError(f'{rel} is not cached; run once without offline mode to download it')
        log.info(f'Downloading {rel} from https://huggingface.co/{repo_id}')
        root = snapshot_download(**kwargs)
    target = Path(root) / rel
    if not target.exists():
        raise FileNotFoundError(f'{rel} is not available in https://huggingface.co/{repo_id}')
    return target


def resolve_config_checkpoints(cfg: DictConfig,
                               keys: tuple[str, ...] = ('text_encoder_path', 'text_encoder_ckpt', 'vae_ckpt',
                                                        'weights'),
                               **fetch) -> DictConfig:
    """Replace checkpoint paths in ``cfg`` with local paths, downloading missing ones in place."""
    from omegaconf import open_dict

    with open_dict(cfg):
        for key in keys:
            value = cfg.get(key, None)
            if value:
                cfg[key] = str(resolve_checkpoint(value, **fetch))
    return cfg


def download_checkpoints(variants: tuple[str, ...] = tuple(VARIANTS),
                         *,
                         repo_id: str = HF_REPO_ID,
                         revision: Optional[str] = None) -> None:
    """Pre-download the checkpoints needed by ``variants`` into the Hugging Face cache."""
    for name in ('weights/ta_clap_ettin32m.pt', 'weights/ta_vae', *(VARIANTS[v]['checkpoint'] for v in variants)):
        resolve_checkpoint(name, repo_id=repo_id, revision=revision)


def _default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _compose_config(config_name: str) -> DictConfig:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base='1.3.2'):
        return compose(config_name=config_name)


class TinyAudioPipeline:
    """Text encoder, TA-DiT generator and TA-VAE decoder wrapped for text-to-audio inference."""

    def __init__(self, cfg: DictConfig, net: torch.nn.Module, feature_utils: FeaturesUtils, *, device: str,
                 dtype: torch.dtype, duration: float) -> None:
        self.cfg = cfg
        self.net = net
        self.feature_utils = feature_utils
        self.device = device
        self.dtype = dtype
        self.use_meanflow = is_meanflow_model(cfg.model)
        self.default_num_steps = VARIANTS.get(cfg.model, {}).get('num_steps', 25)
        self.default_cfg_strength = float(cfg.cfg_strength)
        self.duration = None
        self.set_duration(duration)

    @classmethod
    def from_pretrained(cls,
                        variant: str = 'tinyaudio_mf',
                        *,
                        model_path: Optional[Union[str, Path]] = None,
                        config_name: Optional[str] = None,
                        device: Optional[str] = None,
                        dtype: Optional[torch.dtype] = None,
                        duration: float = 10.0,
                        repo_id: str = HF_REPO_ID,
                        revision: Optional[str] = None,
                        local_files_only: bool = False,
                        text_encoder_attn_impl: Optional[str] = None) -> 'TinyAudioPipeline':
        """Build a pipeline, downloading any missing checkpoints from the Hugging Face Hub.

        Args:
            variant: ``tinyaudio_mf`` (MeanFlow, 1 step) or ``tinyaudio`` (flow matching, 25 steps).
            model_path: generator weights; defaults to the released checkpoint of ``variant``.
            device: ``cuda``, ``mps`` or ``cpu``; picked automatically when omitted.
            dtype: defaults to bfloat16 on CUDA and float32 elsewhere.
            local_files_only: never access the network; use cached or local files only.
        """
        if variant not in VARIANTS and model_path is None:
            raise ValueError(f'Unknown variant {variant!r}; choose from {list(VARIANTS)}')
        cfg = _compose_config(config_name or variant)
        fetch = dict(repo_id=repo_id, revision=revision, local_files_only=local_files_only)

        cfg.text_encoder_path = str(resolve_checkpoint(cfg.text_encoder_path, **fetch))
        cfg.text_encoder_ckpt = str(resolve_checkpoint(cfg.text_encoder_ckpt, **fetch))
        cfg.vae_ckpt = str(resolve_checkpoint(cfg.vae_ckpt, **fetch))
        model_path = resolve_checkpoint(model_path or VARIANTS[cfg.model]['checkpoint'], **fetch)

        device = device or _default_device()
        if dtype is None:
            dtype = torch.bfloat16 if device == 'cuda' else torch.float32

        net = build_model_from_cfg(cfg).to(device, dtype).eval()
        net.load_weights(torch.load(model_path, map_location=device, weights_only=True))
        load_text_condition(net, cfg.text_encoder_ckpt)
        log.info(f'Loaded {cfg.model} from {model_path}')

        feature_utils = FeaturesUtils(vae_ckpt=cfg.vae_ckpt,
                                      vae_model_tag=cfg.vae_model_tag,
                                      vae_use_ema=cfg.vae_use_ema,
                                      enable_conditions=True,
                                      text_encoder_path=cfg.text_encoder_path,
                                      text_encoder_weights_path=cfg.text_encoder_ckpt,
                                      text_encoder_attn_implementation=text_encoder_attn_impl or
                                      cfg.get('text_encoder_attn_implementation', None),
                                      need_vae_encoder=False).to(device, dtype).eval()
        sync_empty_conditions(net, feature_utils)
        return cls(cfg, net, feature_utils, device=device, dtype=dtype, duration=duration)

    @property
    def sampling_rate(self) -> int:
        return self.seq_cfg.sampling_rate

    def set_duration(self, duration: float) -> None:
        if duration == self.duration:
            return
        self.seq_cfg = resolve_sequence_config_from_cfg(self.cfg, duration=duration)
        self.net.update_seq_lengths(self.seq_cfg.latent_seq_len)
        self.duration = duration

    def _sampler(self, num_steps: int, sampling_method: Optional[str], flow_shift: Optional[float]):
        if self.use_meanflow:
            return MeanFlow(steps=num_steps)
        sampling = self.cfg.sampling
        return FlowMatching(min_sigma=float(sampling.get('min_sigma', 0.0)),
                            inference_mode=str(sampling_method or sampling.get('method', 'euler')),
                            num_steps=num_steps,
                            reverse_flow=bool(sampling.get('reverse_flow', True)),
                            shift=float(flow_shift if flow_shift is not None else sampling.get('shift', 1.0)))

    @torch.inference_mode()
    def __call__(self,
                 prompt: Union[str, list[str]],
                 *,
                 negative_prompt: str = '',
                 num_steps: Optional[int] = None,
                 cfg_strength: Optional[float] = None,
                 duration: Optional[float] = None,
                 seed: Optional[int] = 42,
                 generator: Optional[torch.Generator] = None,
                 cfg_interval_min: float = 0.0,
                 cfg_interval_max: float = 1.0,
                 sampling_method: Optional[str] = None,
                 flow_shift: Optional[float] = None) -> torch.Tensor:
        """Generate audio for one prompt or a batch of prompts.

        Returns a float32 CPU tensor of shape ``(batch, channels, samples)`` at ``self.sampling_rate``.
        """
        texts = [prompt] if isinstance(prompt, str) else list(prompt)
        if duration is not None:
            self.set_duration(duration)
        if generator is None:
            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)

        num_steps = num_steps if num_steps is not None else self.default_num_steps
        common = dict(negative_text=[negative_prompt] * len(texts),
                      feature_utils=self.feature_utils,
                      net=self.net,
                      rng=generator,
                      cfg_strength=cfg_strength if cfg_strength is not None else self.default_cfg_strength)
        sampler = self._sampler(num_steps, sampling_method, flow_shift)
        if self.use_meanflow:
            audio = generate_mf(texts,
                                mf=sampler,
                                cfg_interval_min=cfg_interval_min,
                                cfg_interval_max=cfg_interval_max,
                                **common)
        else:
            audio = generate_fm(texts, fm=sampler, **common)

        audio = audio.detach().float().cpu()
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        return audio

    def save(self, audio: torch.Tensor, path: Union[str, Path]) -> Path:
        """Save one waveform of shape ``(channels, samples)`` or ``(samples,)`` as a WAV file."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        audio = audio.detach().float().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save(str(path), audio, self.sampling_rate)
        return path
