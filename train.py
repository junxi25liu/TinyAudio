"""Train TinyAudio / TinyAudio-MF.

    torchrun --standalone --nproc_per_node=8 train.py --config-name tinyaudio_mf exp_id=my_mf_run
    torchrun --standalone --nproc_per_node=8 train.py --config-name tinyaudio exp_id=my_fm_run
"""
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import logging
import math
import random
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

from tinyaudio.data.data_setup import setup_eval_dataset, setup_training_datasets
from tinyaudio.model.sequence_config import resolve_sequence_config_from_cfg
from tinyaudio.pipeline import resolve_config_checkpoints
from tinyaudio.runner import build_runner
from tinyaudio.sample import sample
from tinyaudio.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from tinyaudio.utils.logger import TensorboardLogger
from tinyaudio.utils.synthesize_ema import synthesize_ema

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')


@record
@hydra.main(version_base='1.3.2', config_path='config', config_name='tinyaudio_mf')
def train(cfg: DictConfig):
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    distributed_setup()
    run_dir = HydraConfig.get().run.dir

    resolve_config_checkpoints(cfg)
    seq_cfg = resolve_sequence_config_from_cfg(cfg)
    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len

    log = TensorboardLogger(cfg.exp_id, run_dir, logging.getLogger(), is_rank0=(local_rank == 0))
    info_if_rank_zero(log, f'Configuration: {cfg}')
    info_if_rank_zero(log, f'Number of GPUs: {world_size}')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if not cfg.get('use_dynamic_batch', False):
        cfg.batch_size //= world_size
        info_if_rank_zero(log, f'Batch size (per GPU): {cfg.batch_size}')
    total_iterations = cfg.num_iterations

    dataset, sampler, loader = setup_training_datasets(cfg)
    info_if_rank_zero(log, f'Number of training samples: {len(dataset):,d}')
    info_if_rank_zero(log, f'Number of training batches: {len(loader)}')

    do_eval = cfg.get('do_eval', True)
    if do_eval:
        eval_dataset, eval_loader, eval_cfg = setup_eval_dataset(cfg)
        info_if_rank_zero(log, f'Number of eval samples: {len(eval_dataset)}')

    latent_mean = torch.load(cfg.latent_mean, weights_only=True)
    latent_std = torch.load(cfg.latent_std, weights_only=True)

    trainer = build_runner(cfg,
                           log=log,
                           run_path=run_dir,
                           for_training=True,
                           latent_mean=latent_mean,
                           latent_std=latent_std).enter_train()
    eval_rng_clone = trainer.rng.graphsafe_get_state()

    # resume: explicit checkpoint > latest checkpoint in run_dir > initial weights > scratch
    if cfg.checkpoint is not None:
        curr_iter = trainer.load_checkpoint(cfg.checkpoint)
    elif (checkpoint := trainer.get_latest_checkpoint_path()) is not None:
        curr_iter = trainer.load_checkpoint(checkpoint)
    else:
        curr_iter = 0
        if cfg.weights is not None:
            trainer.load_weights(cfg.weights)
        else:
            info_if_rank_zero(log, 'Training from scratch')

    total_epoch = math.ceil(total_iterations / len(loader))
    current_epoch = curr_iter // len(loader)
    info_if_rank_zero(log, f'Approximately {total_epoch - current_epoch} epochs to go.')

    try:
        np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)
        while curr_iter < total_iterations:
            sampler.set_epoch(current_epoch)
            current_epoch += 1

            trainer.enter_train()
            trainer.log.data_timer.start()
            for data in loader:
                if data is None:
                    continue
                trainer.train_pass(data, curr_iter)

                if do_eval and (curr_iter + 1) % cfg.eval_interval == 0:
                    save_eval = (curr_iter + 1) % cfg.save_eval_interval == 0
                    train_rng_snapshot = trainer.rng.graphsafe_get_state()
                    trainer.rng.graphsafe_set_state(eval_rng_clone)
                    info_if_rank_zero(log, f'Iteration {curr_iter}: generating eval samples')
                    audio_dir = None
                    for eval_data in tqdm(eval_loader, disable=local_rank != 0):
                        audio_dir = trainer.inference_pass(eval_data, curr_iter, eval_cfg, save_eval=save_eval)
                    distributed.barrier()
                    trainer.rng.graphsafe_set_state(train_rng_snapshot)
                    trainer.eval(audio_dir, curr_iter, eval_cfg)
                    trainer.enter_train()

                curr_iter += 1
                if curr_iter >= total_iterations:
                    break
    except Exception as e:
        log.error(f'Error at iteration {curr_iter}: {e}')
        raise

    trainer.save_checkpoint(curr_iter)
    trainer.save_weights(curr_iter, save_copy=False)
    distributed.barrier()

    del trainer
    torch.cuda.empty_cache()

    if local_rank == 0 and cfg.ema.enable:
        sigma = cfg.ema.default_output_sigma
        log.info(f'Synthesizing EMA with sigma={sigma}')
        state_dict = synthesize_ema(cfg, sigma, step=None)
        save_path = Path(run_dir) / f'{cfg.exp_id}_ema_final.pth'
        torch.save(state_dict, save_path)
        log.info(f'Synthesized EMA saved to {save_path}')
    distributed.barrier()

    if do_eval:
        sample(cfg)

    log.complete()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == '__main__':
    train()
