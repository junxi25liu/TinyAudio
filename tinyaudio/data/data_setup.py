import logging
import random
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from tinyaudio.data.audio_dataset import AudioCaptionDataset, DistributedDynamicBatchSampler, TextOnlyDataset
from tinyaudio.utils.dist_utils import local_rank

log = logging.getLogger()


def worker_init_fn(worker_id: int):
    # re-seed every worker so that random augmentations differ across workers and ranks
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def error_avoidance_collate(batch):
    return default_collate([x for x in batch if x is not None])


def construct_loader(dataset: Dataset,
                     batch_size: int,
                     num_workers: int,
                     *,
                     shuffle: bool = True,
                     drop_last: bool = True,
                     pin_memory: bool = False,
                     collate_fn=None,
                     persistent_workers: Optional[bool] = None,
                     prefetch_factor: Optional[int] = None,
                     use_dynamic_batch: bool = False,
                     frames_threshold: float = 0,
                     max_samples: int = 0,
                     random_seed: Optional[int] = None,
                     drop_residual: bool = False):
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    kwargs = {}
    if num_workers > 0 and prefetch_factor is not None:
        kwargs['prefetch_factor'] = prefetch_factor
    collate_fn = collate_fn or error_avoidance_collate

    if use_dynamic_batch:
        sampler = DistributedDynamicBatchSampler(SequentialSampler(dataset),
                                                 frames_threshold=frames_threshold,
                                                 max_samples=max_samples,
                                                 random_seed=random_seed,
                                                 drop_residual=drop_residual,
                                                 drop_last=True)
        loader = DataLoader(dataset,
                            batch_sampler=sampler,
                            num_workers=num_workers,
                            worker_init_fn=worker_init_fn,
                            persistent_workers=persistent_workers,
                            pin_memory=pin_memory,
                            collate_fn=collate_fn,
                            **kwargs)
        return sampler, loader

    sampler = DistributedSampler(dataset, rank=local_rank, shuffle=shuffle)
    loader = DataLoader(dataset,
                        batch_size,
                        sampler=sampler,
                        num_workers=num_workers,
                        worker_init_fn=worker_init_fn,
                        drop_last=drop_last,
                        persistent_workers=persistent_workers,
                        pin_memory=pin_memory,
                        collate_fn=collate_fn,
                        **kwargs)
    return sampler, loader


def setup_training_datasets(cfg: DictConfig):
    dataset = AudioCaptionDataset(datasets=cfg.data.datasets,
                                  sample_rate=cfg.audio_sample_rate,
                                  max_length=cfg.audio_max_length,
                                  use_dynamic_batch=cfg.get('use_dynamic_batch', False))
    sampler, loader = construct_loader(dataset,
                                       cfg.batch_size,
                                       cfg.num_workers,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=cfg.pin_memory,
                                       collate_fn=dataset.collate_fn,
                                       prefetch_factor=cfg.get('prefetch_factor', None),
                                       use_dynamic_batch=cfg.get('use_dynamic_batch', False),
                                       frames_threshold=cfg.get('frames_threshold', 0),
                                       max_samples=cfg.get('max_samples', 0),
                                       random_seed=cfg.seed,
                                       drop_residual=cfg.get('drop_residual', False))
    return dataset, sampler, loader


def setup_eval_dataset(cfg: DictConfig, data_cfg: Optional[DictConfig] = None):
    """Caption-only evaluation set (default: cfg.data.AudioCaps_test)."""
    data_cfg = data_cfg or cfg.data.AudioCaps_test
    dataset = TextOnlyDataset(data_cfg.jsonl)
    _, loader = construct_loader(dataset,
                                 cfg.eval_batch_size,
                                 cfg.num_workers,
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=cfg.pin_memory,
                                 persistent_workers=False,
                                 prefetch_factor=cfg.get('prefetch_factor', None))
    return dataset, loader, data_cfg
