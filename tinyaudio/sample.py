"""Generate audio for an evaluation set (default: AudioCaps test) and compute metrics."""
import json
import logging
import random

import numpy as np
import torch
import torch.distributed as distributed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from tinyaudio.data.data_setup import setup_eval_dataset
from tinyaudio.runner import build_runner
from tinyaudio.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from tinyaudio.utils.logger import TensorboardLogger


def sample(cfg: DictConfig):
    run_dir = HydraConfig.get().run.dir
    log = TensorboardLogger(cfg.exp_id, run_dir, logging.getLogger(), is_rank0=(local_rank == 0))
    info_if_rank_zero(log, f'Number of GPUs: {world_size}')

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    runner = build_runner(cfg, log=log, run_path=run_dir, for_training=False).enter_val()

    # weights: explicit `weights=...`, else the synthesized EMA, else the last training weights
    weights = cfg.get('weights', None) or runner.get_final_ema_weight_path() or runner.get_latest_weight_path()
    if weights is None:
        raise FileNotFoundError('No weights found. Pass weights=/path/to/model.pth')
    info_if_rank_zero(log, f'Using weights: {weights}')
    runner.load_weights(weights)

    _, loader, data_cfg = setup_eval_dataset(cfg)
    with open_dict(data_cfg):
        data_cfg.output_subdir = None
        if cfg.get('output_name', None) is not None:
            data_cfg.tag = f'{data_cfg.tag}-{cfg.output_name}'

    audio_dir = None
    for it, data in enumerate(tqdm(loader, disable=local_rank != 0)):
        audio_dir = runner.inference_pass(data, it, data_cfg)
    distributed.barrier()
    info_if_rank_zero(log, f'Generation finished: {audio_dir}')

    metrics = runner.eval(audio_dir, 0, data_cfg)
    if local_rank == 0 and metrics is not None:
        out = f'{run_dir}/{data_cfg.tag}-metrics.json'
        with open(out, 'w') as f:
            json.dump(metrics, f, indent=4)
        log.info(f'Metrics saved to {out}')
    return metrics
