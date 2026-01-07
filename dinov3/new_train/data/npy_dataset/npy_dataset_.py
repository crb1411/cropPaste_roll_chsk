import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
PATCH_H = 224
PATCH_W = 224
PATCH_C = 3
PATCH_BYTES = PATCH_H * PATCH_W * PATCH_C

SHARD_SIZE_BYTES = 2 * 1024**4
SHARD_CAP = int((SHARD_SIZE_BYTES // PATCH_BYTES) * 0.98)

INDEX_DTYPE = np.dtype([
    ("h5_id", np.int32),
    ("patch_id", np.int32),
    ("global_idx", np.int64)
])


class ShardedPatchDataset(Dataset):
    def __init__(self, shard_dir, index_path, transform=None):
        self.shard_dir = shard_dir
        self.transform = transform
        self.index_path = index_path

        
        self.index = None
        self.shard_paths = None
        self._shards = None

    def _ensure_data(self):
        if self.index is None:
            # 加载索引 (mmap)
            self.index = np.memmap(self.index_path, dtype=INDEX_DTYPE, mode="r")
            self.shard_paths = sorted(
                [os.path.join(self.shard_dir, f) for f in os.listdir(self.shard_dir) if f.startswith("shard_")]
            )
            assert len(self.shard_paths) > 0, "No shards found!"
            self._shards = [None] * len(self.shard_paths)
    def __len__(self):
        self._ensure_data()
        return self.index.shape[0]

    def _get_shard(self, shard_id):
        self._ensure_data()
        """获取 (lazy load) memmap handle"""
        if self._shards[shard_id] is None:
            path = self.shard_paths[shard_id]
            self._shards[shard_id] = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                shape=(SHARD_CAP, PATCH_H, PATCH_W, PATCH_C)
            )
        return self._shards[shard_id]

    def __getitem__(self, idx):
        self._ensure_data()
        rec = self.index[idx]
        gid = rec["global_idx"]

        shard_id = gid // SHARD_CAP
        local_id = gid % SHARD_CAP

        shard = self._get_shard(shard_id)
        patch = shard[local_id]

        patch = Image.fromarray(patch).convert("RGB")

        if self.transform:
            patch = self.transform(patch)

        return patch, None


class ShardedPatchDataset(Dataset):
    """
    支持传入单个或多个数据集：

    - shard_dirs: str 或 [str, str, ...]
    - index_paths: str 或 [str, str, ...]

    会把多个数据集在样本维度上拼接成一个大 Dataset。
    注意：不同数据集里的 h5_id 可能会重复（都从 0 开始），
    如果你需要全局唯一的 h5_id，可以后面再单独处理。
    """
    def __init__(self, shard_dirs, index_paths, transform=None):
        # 统一成列表
        if isinstance(shard_dirs, str):
            shard_dirs = [shard_dirs]
        if isinstance(index_paths, str):
            index_paths = [index_paths]

        assert len(shard_dirs) == len(index_paths), \
            "shard_dirs 和 index_paths 数量必须一致"

        self.transform = transform
        self.datasets = []   # 每个子数据集的信息
        self.cum_sizes = []  # 全局前缀和，用于定位 idx 属于哪个子数据集
        total = 0

        for sd, ip in zip(shard_dirs, index_paths):
            # 1) 加载 index（mmap）
            index = np.memmap(ip, dtype=INDEX_DTYPE, mode="r")
            size = index.shape[0]
            if size == 0:
                continue

            # 2) 找到该数据集的所有 shard 文件
            shard_paths = sorted(
                [os.path.join(sd, f)
                 for f in os.listdir(sd)
                 if f.startswith("shard_")]
            )
            assert len(shard_paths) > 0, f"No shards found in {sd}"

            # 3) 为该数据集创建 mmap 缓存列表（延迟加载）
            shards_cache = [None] * len(shard_paths)

            self.datasets.append({
                "index": index,
                "shard_paths": shard_paths,
                "shards": shards_cache,
            })

            total += size
            self.cum_sizes.append(total)

        assert len(self.datasets) > 0, "No valid datasets found."
        self.total = total

    def __len__(self):
        return self.total

    def _locate_dataset(self, idx):
        """
        给定全局 idx，找到属于哪个子数据集，以及在该子数据集内的局部 idx
        """
        # 手写一个小循环，数据集数量一般很少（2~几个），不用 bisect 也行
        prev = 0
        for ds_id, end in enumerate(self.cum_sizes):
            if idx < end:
                local_idx = idx - prev
                return ds_id, local_idx
            prev = end
        # 理论上不会到这里
        raise IndexError(f"Index out of range: {idx}")

    def _get_shard(self, ds_id, shard_id):
        """
        获取指定子数据集的指定 shard 的 memmap（lazy load）
        """
        ds = self.datasets[ds_id]
        if ds["shards"][shard_id] is None:
            path = ds["shard_paths"][shard_id]
            ds["shards"][shard_id] = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                shape=(SHARD_CAP, PATCH_H, PATCH_W, PATCH_C)
            )
        return ds["shards"][shard_id]

    def __getitem__(self, idx):
        # 1) 找到属于哪个子数据集
        ds_id, local_idx = self._locate_dataset(idx)
        ds = self.datasets[ds_id]

        # 2) 从对应 index 中取记录
        rec = ds["index"][local_idx]
        gid = rec["global_idx"]

        shard_id = gid // SHARD_CAP
        local_id = gid % SHARD_CAP

        shard = self._get_shard(ds_id, shard_id)
        patch = shard[local_id]

        patch = Image.fromarray(patch).convert("RGB")

        if self.transform:
            patch = self.transform(patch)

        return patch, rec



if __name__ == "__main__":
    shard_dir = ['/mnt/local08/data_new/shard', '/mnt/local10/data_new/shard']
    index_path = ["/mnt/local08/data_new/big_index.npy", "/mnt/local10/data_new/shard/big_index.npy"]
    dataset = ShardedPatchDataset(
            shard_dir=shard_dir,
            index_path=index_path
        )

