import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict
from PIL import Image
import logging
logger = logging.getLogger('dinov3')

PATCH_H = 224
PATCH_W = 224
PATCH_C = 3
PATCH_BYTES = PATCH_H * PATCH_W * PATCH_C

SHARD_SIZE_BYTES = 2 * 1024**4
SHARD_CAP = int((SHARD_SIZE_BYTES // PATCH_BYTES) * 0.98)

INDEX_DTYPE = np.dtype([
    ("h5_id", np.int32),
    ("patch_id", np.int32),
    ("global_idx", np.int64),
])


class ShardedPatchDataset(Dataset):
    """
    支持两种用法：

    1）单个数据集：
        ShardedPatchDataset("/path/to/shards", "/path/to/big_index.npy")

    2）多个数据集合并：
        ShardedPatchDataset(
            ["/ds1/shards", "/ds2/shards"],
            ["/ds1/big_index.npy", "/ds2/big_index.npy"],
        )
    """
    def __init__(self, shard_dir, index_path, transform=None, patch_h=None, patch_w=None):
        # 统一成列表
        if isinstance(shard_dir, str):
            shard_dir = [shard_dir]
        if isinstance(index_path, str):
            index_path = [index_path]

        assert len(shard_dir) == len(index_path), \
            "shard_dir 和 index_path 的数量必须一致"

        self.shard_dirs = shard_dir
        self.index_paths = index_path
        self.transform = transform

        self.bad_map = defaultdict(set)   # bad_idx -> {used_replacement_idx}
        self.max_retry = 10               # 防止死循环


        # 延迟初始化
        self.datasets = None      # 每个子数据集的 {index, shard_paths, shards_cache}
        self.cum_sizes = None     # 全局前缀和
        self.total = None
        
        self.patch_h = 224 if patch_h is None else patch_h
        self.patch_w = 224 if patch_w is None else patch_w
        if self.patch_h == self.patch_w and self.patch_h == 224:
            self.shard_cap = SHARD_CAP
        else:
            self.shard_cap = int((SHARD_SIZE_BYTES // (self.patch_h * self.patch_w * PATCH_C)) * 0.98)

    # ---------- 延迟加载所有 index & shards 基本信息 ----------
    def _ensure_data(self):
        if self.datasets is not None:
            return

        datasets = []
        cum_sizes = []
        total = 0

        for sd, ip in zip(self.shard_dirs, self.index_paths):
            # 1) 加载索引 (mmap)
            index = np.memmap(ip, dtype=INDEX_DTYPE, mode="r")
            size = index.shape[0]
            if size == 0:
                continue

            # 2) 找出该数据集的所有 shard 文件
            shard_paths = sorted(
                os.path.join(sd, f)
                for f in os.listdir(sd)
                if f.startswith("shard_") and f.endswith('.npy')
            )
            assert len(shard_paths) > 0, f"No shards found in {sd}"

            # 3) 为该数据集创建 memmap 缓存列表（lazy load）
            shards_cache = [None] * len(shard_paths)

            datasets.append({
                "index": index,
                "shard_paths": shard_paths,
                "shards": shards_cache,
            })

            total += size
            cum_sizes.append(total)

        assert len(datasets) > 0, "No valid datasets found."

        self.datasets = datasets
        self.cum_sizes = cum_sizes
        self.total = total

    def __len__(self):
        self._ensure_data()
        return self.total

    # ---------- 工具：定位 idx 属于哪个子数据集 ----------
    def _locate_dataset(self, idx):
        self._ensure_data()
        prev = 0
        for ds_id, end in enumerate(self.cum_sizes):
            if idx < end:
                local_idx = idx - prev
                return ds_id, local_idx
            prev = end
        raise IndexError(f"Index out of range: {idx}")

    # ---------- 工具：获取某个数据集的 shard memmap ----------
    
    def _get_shard(self, ds_id, shard_id):
        self._ensure_data()
        ds = self.datasets[ds_id]

        if ds["shards"][shard_id] is None:
            path = ds["shard_paths"][shard_id]

            # === 关键：根据真实文件大小算可用 patch 数 ===
            file_size = os.path.getsize(path)
            n_patches = file_size // (self.patch_h * self.patch_w * 3)

            ds["shards"][shard_id] = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                shape=(n_patches, self.patch_h, self.patch_w, 3),
            )
            # print(f'shard_{shard_id} len is: {len(ds["shards"][shard_id])}')

        return ds["shards"][shard_id]


    def __getitem_old__(self, idx):
        # 1) 找到属于哪个子数据集 & 子数据集内的局部 idx
        ds_id, local_idx = self._locate_dataset(idx)
        ds = self.datasets[ds_id]

        # 2) 取出该子数据集 index 记录
        rec = ds["index"][local_idx]
        gid = rec["global_idx"]

        shard_id = gid // self.shard_cap
        local_id = gid % self.shard_cap

        shard = self._get_shard(ds_id, shard_id)
        max_local_id = len(shard)
        if local_id < max_local_id:
            patch = shard[local_id]
        else:
            patch = shard[local_id - max_local_id]
            logger.error(f'npydataset patches out index: global_index_{idx}->subdata_{local_idx}->shared_{shard_id}->local_id_{local_id}')


        patch = Image.fromarray(patch).convert("RGB")

        if self.transform is not None:
            patch = self.transform(patch)

        return patch, None
    def __getitem__(self, idx):
        self._ensure_data()

        orig_idx = idx
        tried = 0

        while True:
            ds_id, local_idx = self._locate_dataset(idx)
            ds = self.datasets[ds_id]
            rec = ds["index"][local_idx]
            gid = rec["global_idx"]

            shard_id = gid // self.shard_cap
            local_id = gid % self.shard_cap

            shard = self._get_shard(ds_id, shard_id)
            max_local_id = len(shard)

            if local_id < max_local_id:
                patch = shard[local_id]
            else:
                patch = shard[local_id - max_local_id]
                logger.error(
                    f"npydataset OOB: global={idx} shard={shard_id} local={local_id}"
                )

            img = Image.fromarray(patch).convert("RGB")

            # ====== 黑图检测 ======
            if not is_black_patch(img):
                if self.transform is not None:
                    img = self.transform(img)
                return img, None

            # ====== 坏样本处理 ======
            tried += 1
            self.bad_map[orig_idx].add(idx)

            if tried >= self.max_retry:
                logger.error(
                    f"[BAD_PATCH] idx={orig_idx} retry>{self.max_retry}, return anyway"
                )
                if self.transform is not None:
                    img = self.transform(img)
                return img, None

            # 从全局随机采样一个新 index（避免重复）
            while True:
                new_idx = random.randint(0, self.total - 1)
                if new_idx not in self.bad_map[orig_idx]:
                    break

            idx = new_idx




def is_black_patch(pil_img, eps=2, ratio=0.99):
    """
    eps: 像素阈值（0~255）
    ratio: 黑像素比例
    """
    arr = np.asarray(pil_img)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    black = arr < eps
    return black.mean() > ratio


def save_image_rgb_png(img, save_path: str):
    """
    img: PIL.Image.Image，期望是 RGB（如果不是会自动转）
    save_path: 以 .png 结尾
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(img)}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path, format="PNG", compress_level=3)  # 0~9，3 是速度/体积折中


if __name__ == "__main__":
    # '/mnt/local08/data_new/shard', "/mnt/local08/data_new/big_index.npy",
    shard_dir = ['/mnt/local10/data_new/shard_datawheel']
    index_path = ["/mnt/local10/data_new/shard_datawheel/big_index.npy"]
    # shard_dir = ['/mnt/local08/data_new/shard']
    # index_path = ["/mnt/local08/data_new/big_index.npy"]
    dataset = ShardedPatchDataset(
            shard_dir=shard_dir,
            index_path=index_path
        )
    os.makedirs(os.path.join(shard_dir[0], 'test_imgs'), exist_ok=True)
    idxs = [0, 14316556//2, 14316557*2-1, 14316559, 14316558, 33144601, 33144601//2]
    for idx in idxs:
        dataset[idx][0].save(os.path.join(shard_dir[0], 'test_imgs', f'img_{idx}.png'), format="PNG")
    
    # print(len(dataset))
    # print(dataset[14316556][0])
    # print(dataset[33144601//2][0])
    # print(dataset[2][0])
