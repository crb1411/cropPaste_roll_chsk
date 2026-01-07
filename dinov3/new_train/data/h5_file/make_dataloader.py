import torch


from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, default_collate, Sampler
from typing import Optional
import logging
from .h5_dataset import H5Dataset, worker_init_fn
from enum import Enum
from .sampler import DistributedFirstNSequentialSampler, RandomSamplerWithReplacement


logger = logging.getLogger("dinov2")



def make_dataloader(
        ds, 
        batch_size=256, 
        num_workers=8, 
        worker_init_fn=worker_init_fn,
        sampler=None,
        collate_fn=default_collate,
    ):
    if sampler is None:
        sampler = DistributedFirstNSequentialSampler(ds)
    if collate_fn is None:
        collate_fn = default_collate

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    return loader


