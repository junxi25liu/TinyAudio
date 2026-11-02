"""Evaluate a trained model on the evaluation set (default: AudioCaps test).

    torchrun --standalone --nproc_per_node=1 eval.py --config-name tinyaudio_mf \
        exp_id=eval_tinyaudio_mf weights=./weights/tinyaudio_mf.pth
"""
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import logging
from datetime import timedelta

import hydra
import torch
import torch.distributed as distributed
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record

from tinyaudio.model.sequence_config import resolve_sequence_config_from_cfg
from tinyaudio.pipeline import resolve_config_checkpoints
from tinyaudio.sample import sample
from tinyaudio.utils.dist_utils import local_rank, world_size

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


@record
@hydra.main(version_base='1.3.2', config_path='config', config_name='tinyaudio_mf')
def main(cfg: DictConfig):
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')

    resolve_config_checkpoints(cfg)
    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = resolve_sequence_config_from_cfg(cfg).latent_seq_len

    sample(cfg)
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == '__main__':
    main()
