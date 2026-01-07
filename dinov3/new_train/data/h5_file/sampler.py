import math
import torch

 
from torch.utils.data import Sampler
import numpy as np
import itertools


class DistributedFirstNSequentialSampler(Sampler[int]):
    """
    在 DDP 下，对前 N 个样本做“顺序 + 切片”采样。
    - 不做全局 shuffle（避免 O(N) 内存/时间）
    - 不物化大列表；用 range 切片，内存友好
    - 支持 drop_last / padding（默认不丢弃）
    """
    def __init__(self, dataset, N=None, drop_last=False,
                 num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                num_replicas = 1
            else:
                num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                rank = 0
            else:
                rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.dataset_len = len(dataset) if N is None else min(N, len(dataset))
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        if self.drop_last:
            # 每个 rank 的样本数向下取整
            self.num_samples = self.dataset_len // self.num_replicas
        else:
            # 每个 rank 的样本数向上取整（需要 padding）
            self.num_samples = int(math.ceil(self.dataset_len / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # 顺序：全局索引 0..dataset_len-1
        # 切片给当前 rank
        start = self.rank * self.num_samples
        end   = start + self.num_samples

        # 主体部分：真正的数据范围
        real_start = start
        real_end   = min(end, self.dataset_len)

        # 先产出真实范围
        for idx in range(real_start, real_end):
            yield idx

        # 如需 padding（不丢弃时，凑满 num_samples）
        if not self.drop_last and real_end - real_start < self.num_samples:
            needed = self.num_samples - (real_end - real_start)
            # 从开头循环补齐（与 DistributedSampler 的语义保持一致）
            pad_idx = 0
            while needed > 0:
                yield pad_idx
                pad_idx += 1
                if pad_idx >= self.dataset_len:
                    pad_idx = 0
                needed -= 1

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        # 顺序采样不需要根据 epoch 改变，但保留接口以兼容训练循环
        pass


class RandomSamplerWithReplacement(Sampler[int]):
    """
    分布式随机采样器，支持有放回/无放回采样。
    - 每个 epoch 会重新生成随机序列
    - 每个 rank 自动分配 disjoint 子集
    """
    def __init__(
        self,
        dataset,
        seed=87,
        generator=None,
        replacement=True,
        shuffle=True,
        num_replicas=None,
        rank=None,
    ):
        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                num_replicas = 1
            else:
                num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                rank = 0
            else:
                rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.generator = generator if generator is not None else torch.Generator()
        self.replacement = replacement
        self._shuffle = shuffle
        self._epoch = 0

        # 每个 rank 负责的样本数
        self.num_samples = int(math.ceil(self.dataset_len / self.num_replicas))

    def __iter__(self):
        # 为当前 epoch 和 rank 生成确定性随机数种子
        seed = self.seed + self._epoch * self.num_replicas + self.rank
        rng = np.random.default_rng(seed)

        indices = np.arange(self.dataset_len)

        if self._shuffle:
            iterable = rng.choice(
                indices, self.num_samples, replace=self.replacement
            )
        else:
            # 顺序 or 重复 pad
            full = np.tile(indices, (self.num_samples + self.dataset_len - 1) // self.dataset_len)
            iterable = full[: self.num_samples]

        # 下一轮 epoch
        self._epoch += 1

        return iter(iterable.tolist())

    def __len__(self):
        return self.num_samples
