from typing import Optional

from nitrous_ema import PostHocEMA
from omegaconf import DictConfig

from tinyaudio.model.networks import build_model_from_cfg


def synthesize_ema(cfg: DictConfig, sigma: float, step: Optional[int]):
    # the network must be built with the same arguments as in training
    net = build_model_from_cfg(cfg)
    emas = PostHocEMA(net,
                      sigma_rels=cfg.ema.sigma_rels,
                      update_every=cfg.ema.update_every,
                      checkpoint_every_num_steps=cfg.ema.checkpoint_every,
                      checkpoint_folder=cfg.ema.checkpoint_folder)
    synthesized_ema = emas.synthesize_ema_model(sigma_rel=sigma, step=step, device='cpu')
    return synthesized_ema.ema_model.state_dict()
