import logging
from typing import Optional

import torch
from colorlog import ColoredFormatter

from tinyaudio.model.flow_matching import FlowMatching
from tinyaudio.model.mean_flow import MeanFlow
from tinyaudio.model.networks import TinyAudio, TinyAudioMF
from tinyaudio.model.utils.features_utils import FeaturesUtils, load_text_encoder_checkpoint

log = logging.getLogger()


def _encode_prompts(feature_utils: FeaturesUtils, net: TinyAudio, text: Optional[list[str]], bs: int):
    if text is not None:
        return feature_utils.encode_text(text, return_padding_mask=True)
    text_f, text_f_c = net.get_empty_string_sequence(bs)
    return text_f, text_f_c, None


def _prepare(text, negative_text, feature_utils, net):
    bs = len(text) if text is not None else len(negative_text)
    text_f, text_f_c, text_mask = _encode_prompts(feature_utils, net, text, bs)
    conditions = net.preprocess_conditions(text_f, text_f_c, text_padding_mask=text_mask)
    if negative_text is not None:
        assert len(negative_text) == bs
        negative_features = feature_utils.encode_text(negative_text, return_padding_mask=True)
        empty_conditions = net.get_empty_conditions(bs, negative_text_features=negative_features)
    else:
        empty_conditions = net.get_empty_conditions(bs)
    return bs, conditions, empty_conditions


def _decode(feature_utils: FeaturesUtils, net: TinyAudio, x1: torch.Tensor) -> torch.Tensor:
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    return feature_utils.vocode(spec)


def generate_fm(text: Optional[list[str]],
                *,
                negative_text: Optional[list[str]] = None,
                feature_utils: FeaturesUtils,
                net: TinyAudio,
                fm: FlowMatching,
                rng: torch.Generator,
                cfg_strength: float,
                latent_seq_len: Optional[int] = None) -> torch.Tensor:
    """Generate audio with the flow-matching model."""
    bs, conditions, empty_conditions = _prepare(text, negative_text, feature_utils, net)
    seq_len = latent_seq_len if latent_seq_len is not None else net.latent_seq_len
    x0 = torch.randn(bs,
                     seq_len,
                     net.latent_dim,
                     device=feature_utils.device,
                     dtype=feature_utils.dtype,
                     generator=rng)

    def cfg_ode_wrapper(t, x):
        return net.ode_wrapper(t, x, conditions, empty_conditions, cfg_strength)

    return _decode(feature_utils, net, fm.to_data(cfg_ode_wrapper, x0))


def generate_mf(text: Optional[list[str]],
                *,
                negative_text: Optional[list[str]] = None,
                feature_utils: FeaturesUtils,
                net: TinyAudioMF,
                mf: MeanFlow,
                rng: torch.Generator,
                cfg_strength: float,
                cfg_interval_min: Optional[float] = 0.0,
                cfg_interval_max: Optional[float] = 1.0,
                latent_seq_len: Optional[int] = None) -> torch.Tensor:
    """Generate audio with the MeanFlow model (one or a few steps)."""
    bs, conditions, empty_conditions = _prepare(text, negative_text, feature_utils, net)
    seq_len = latent_seq_len if latent_seq_len is not None else net.latent_seq_len
    x0 = torch.randn(bs,
                     seq_len,
                     net.latent_dim,
                     device=feature_utils.device,
                     dtype=feature_utils.dtype,
                     generator=rng)
    use_interval = cfg_interval_min is not None or cfg_interval_max is not None

    def cfg_ode_wrapper(t, r, x):
        return net.ode_wrapper(t,
                               r,
                               x,
                               conditions,
                               empty_conditions,
                               cfg_strength,
                               omega=cfg_strength if use_interval else None,
                               t_min=cfg_interval_min,
                               t_max=cfg_interval_max)

    return _decode(feature_utils, net, mf.to_data(cfg_ode_wrapper, x0))


def load_text_condition(net: TinyAudio, ckpt_path: str) -> None:
    """Load the text-side condition parameters that were aligned together with TA-CLAP
    (learned padding token and token-pooling weights)."""
    payload = load_text_encoder_checkpoint(ckpt_path)
    dev_state = payload['dev_condition_state_dict']
    current_state = net.state_dict()
    load_state = {k: v.to(device=net.device) for k, v in dev_state.items() if k in current_state}
    net.load_state_dict(load_state, strict=False)
    log.info(f'Loaded text condition parameters: {sorted(load_state.keys())}')


@torch.no_grad()
def sync_empty_conditions(net: TinyAudio, feature_utils: FeaturesUtils) -> None:
    """Store the encoding of the empty prompt (used as the unconditional branch) in the network."""
    feat, feat_c, padding_mask = feature_utils.encode_text([''], return_padding_mask=True)
    net.empty_string_feat.copy_(feat[0].to(device=net.empty_string_feat.device, dtype=net.empty_string_feat.dtype))
    net.empty_string_feat_c.copy_(feat_c[0].to(device=net.empty_string_feat_c.device,
                                               dtype=net.empty_string_feat_c.dtype))
    padding_mask = padding_mask[0].to(device=net.device, dtype=torch.bool).view(-1)
    if net._empty_string_padding_mask.shape == padding_mask.shape:
        net._empty_string_padding_mask.copy_(padding_mask)
    else:
        net._empty_string_padding_mask = padding_mask


LOGFORMAT = '[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s'


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)
