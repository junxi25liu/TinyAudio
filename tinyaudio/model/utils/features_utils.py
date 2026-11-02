import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from tinyaudio.ext.autoencoder import AutoEncoderModule
from tinyaudio.model.utils.distributions import DiagonalGaussianDistribution

log = logging.getLogger()

TEXT_SEQ_LEN = 77
DEFAULT_TEXT_ENCODER = 'jhu-clsp/ettin-encoder-32m'


def load_text_encoder_checkpoint(path: str) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


class FeaturesUtils(nn.Module):
    """Frozen text encoder (TA-CLAP) and audio VAE (TA-VAE)."""

    def __init__(self,
                 *,
                 vae_ckpt: Optional[str] = None,
                 vae_model_tag: str = 'latest',
                 vae_use_ema: bool = True,
                 enable_conditions: bool = True,
                 text_encoder_path: str = DEFAULT_TEXT_ENCODER,
                 text_encoder_weights_path: Optional[str] = None,
                 text_encoder_attn_implementation: Optional[str] = None,
                 need_vae_encoder: bool = True):
        super().__init__()

        if enable_conditions:
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            attn_impl = text_encoder_attn_implementation or 'sdpa'
            model_path = Path(text_encoder_path)
            has_local_weights = model_path.is_dir() and any(
                (model_path / name).is_file() for name in ('pytorch_model.bin', 'model.safetensors'))
            if model_path.is_dir() and not has_local_weights and text_encoder_weights_path:
                # the TA-CLAP checkpoint holds the complete encoder state
                config = AutoConfig.from_pretrained(text_encoder_path)
                self.text_encoder = AutoModel.from_config(config, attn_implementation=attn_impl).eval()
            else:
                self.text_encoder = AutoModel.from_pretrained(text_encoder_path,
                                                              attn_implementation=attn_impl).eval()
            if text_encoder_weights_path:
                self._load_text_encoder_weights(text_encoder_weights_path)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            self.text_encoder = None
            self.tokenizer = None

        if vae_ckpt is not None:
            self.tod = AutoEncoderModule(vae_ckpt_path=vae_ckpt,
                                         model_tag=vae_model_tag,
                                         use_ema=vae_use_ema,
                                         need_vae_encoder=need_vae_encoder)
        else:
            self.tod = None

    def _load_text_encoder_weights(self, weights_path: str) -> None:
        payload = load_text_encoder_checkpoint(weights_path)
        if not isinstance(payload, dict) or 'encoder_state_dict' not in payload:
            raise ValueError(f'Unsupported text encoder checkpoint format: {weights_path}')
        missing, unexpected = self.text_encoder.load_state_dict(payload['encoder_state_dict'], strict=False)
        log.info('Loaded TA-CLAP text encoder from %s (missing=%d unexpected=%d)', weights_path, len(missing),
                 len(unexpected))

    def compile(self):
        self.encode_text = torch.compile(self.encode_text)
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.no_grad()
    def encode_text(self, text: list[str], return_padding_mask: bool = False):
        assert self.text_encoder is not None, 'Text encoder is not loaded'
        tokens = self.tokenizer(text,
                                max_length=TEXT_SEQ_LEN,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_padding_mask = attention_mask.eq(0)
        mask = attention_mask.unsqueeze(-1).to(text_features.dtype)
        text_features_c = (text_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        if return_padding_mask:
            return text_features, text_features_c, text_padding_mask
        return text_features, text_features_c

    @torch.no_grad()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.encode_waveform(x)

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, C) latent sequence
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z.transpose(1, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
