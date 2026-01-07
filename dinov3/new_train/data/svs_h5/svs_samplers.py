import itertools
import torch


from torch.utils.data import Sampler
from typing import Optional
from enum import Enum

class SamplerType_WSI(Enum):
    """
    GLOBAL_CHUNK: 小数据（<=1e7）
    BLOCK_CHUNK: 中数据（1e7~1e9）
    HASH_CHUNK: 超大数据（>=1e9）
    """
    GLOBAL_CHUNK = 0
    BLOCK_CHUNK = 1
    HASH_CHUNK= 2


# ========== 分布式 rank / world_size 工具 ==========
class distributed:
    @staticmethod
    def get_global_rank():
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return 0
        return torch.distributed.get_rank()

    @staticmethod
    def get_global_size():
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return 1
        return torch.distributed.get_world_size()

# ========== 工具函数 ==========
def hash_permute(idx: int, seed: int, num_chunks: int) -> int:
    """伪随机置换函数：输入 idx → 输出一个 0~num_chunks-1 的乱序编号"""
    x = (idx + seed * 1013904223) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
    return x % num_chunks

# ========== 采样器 ==========
class ShardedChunkedInfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        chunk_size: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        shuffle_mode: str = "global",  # "global", "block", "hash"
        block_size: int = 10**6,       # 用于 block shuffle
    ):
        """
        Args:
            sample_count: 数据集大小
            chunk_size: 每个 chunk 的大小
            shuffle: 是否打乱 chunk 顺序
            seed: 全局 seed
            start: rank 偏移 (默认 = global_rank)
            step: world_size (默认 = global_size)
            advance: 跳过多少个样本 (断点续训用)
            shuffle_mode: 打乱模式
                - "global": 一次 randperm 整个 num_chunks (小规模用)
                - "block" : 分块 randperm，适合中等规模
                - "hash"  : 哈希伪随机置换，适合超大规模 (10B)
            block_size: block 模式的块大小
        """
        self._sample_count = sample_count
        self._chunk_size = chunk_size
        self._shuffle = shuffle
        self._seed = seed
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        print(f"Sampler start={self._start}, step={self._step}")
        self._advance = advance
        self._iter_count = 0
        self._shuffle_mode = shuffle_mode
        self._block_size = block_size

    def __iter__(self):
        # 处理 advance（断点续训）
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        iterator = (
            self._shuffled_chunk_iterator()
            if self._shuffle else self._sequential_chunk_iterator()
        )
        yield from itertools.islice(iterator, self._advance, None)

    def _sequential_chunk_iterator(self):
        """顺序 chunk 输出"""
        while True:
            num_chunks = (self._sample_count + self._chunk_size - 1) // self._chunk_size
            for chunk_id in range(self._start, num_chunks, self._step):
                start = chunk_id * self._chunk_size
                end = min(start + self._chunk_size, self._sample_count)
                yield from range(start, end)
            self._iter_count += 1

    def _shuffled_chunk_iterator(self):
        """根据 shuffle_mode 选择不同策略"""
        num_chunks = (self._sample_count + self._chunk_size - 1) // self._chunk_size
        generator = torch.Generator()

        while True:
            generator.manual_seed(self._seed + self._iter_count)

            if self._shuffle_mode == "global":
                # 一次性打乱所有 chunk
                perm = torch.randperm(num_chunks, generator=generator)
                for idx in range(self._start, num_chunks, self._step):
                    chunk_id = perm[idx].item()
                    start = chunk_id * self._chunk_size
                    end = min(start + self._chunk_size, self._sample_count)
                    yield from range(start, end)

            elif self._shuffle_mode == "block":
                # 分块打乱
                for block_start in range(0, num_chunks, self._block_size):
                    block_end = min(block_start + self._block_size, num_chunks)
                    perm = torch.randperm(block_end - block_start, generator=generator) + block_start
                    for idx in range(self._start, len(perm), self._step):
                        chunk_id = perm[idx].item()
                        start = chunk_id * self._chunk_size
                        end = min(start + self._chunk_size, self._sample_count)
                        yield from range(start, end)

            elif self._shuffle_mode == "hash":
                # 哈希伪随机置换
                for idx in range(self._start, num_chunks, self._step):
                    chunk_id = hash_permute(idx + self._iter_count * num_chunks,
                                            self._seed, num_chunks)
                    start = chunk_id * self._chunk_size
                    end = min(start + self._chunk_size, self._sample_count)
                    yield from range(start, end)

            else:
                raise ValueError(f"Unknown shuffle_mode={self._shuffle_mode}")

            self._iter_count += 1
            
            
if __name__ == "__main__":
    # 小数据（<=1e7）
    sampler = ShardedChunkedInfiniteSampler(
        sample_count=10_000_000,
        chunk_size=1000,
        shuffle=True,
        shuffle_mode="global"
    )

    # 中数据（1e7~1e9）
    sampler = ShardedChunkedInfiniteSampler(
        sample_count=1_000_000_000,
        chunk_size=1000,
        shuffle=True,
        shuffle_mode="block",
        block_size=10**6
    )

    # 超大数据（>=1e9）
    sampler = ShardedChunkedInfiniteSampler(
        sample_count=10_000_000_000,
        chunk_size=1000,
        shuffle=True,
        shuffle_mode="hash"
    )

