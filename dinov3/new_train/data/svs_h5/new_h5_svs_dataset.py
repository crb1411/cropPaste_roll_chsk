#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSI Patch Dataset (OpenSlide) — memmap 版
- svs_patch.npy: 结构化或普通二维，列/字段为 [sdpc_id, patch_size, x, y, level]
- path.txt: 每行一个绝对路径（行号=sdpc_id），或“两列: <sdpc_id> <path>”
"""
import logging 
import os
import time
from datetime import datetime
from collections import OrderedDict
from typing import Optional, Dict, Any, List

import numpy as np

from PIL import Image
from tqdm import tqdm
import torch

 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


from ...utils.log_create import creat_subdir
from .svs_samplers import SamplerType_WSI, ShardedChunkedInfiniteSampler
from .show_img_augmented import save_index
from .slide_lru import SlideLRU

logger = logging.getLogger("dinov3")

# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[4].parent / "opensdpc"))
# import opensdpc as openslide
import openslide
# 建议：提升重复邻近读时的命中


DTYPE = np.dtype([
    ("h5_id", np.int32),
    ("x",     np.int32),
    ("y",     np.int32),
    ("patch", np.int16),
])


class PathAccessor:
    def __init__(
            self, 
            paths_raw, 
            suffix: str="/mnt/data/svs_data",
        ):
        self.paths_raw = paths_raw
        self.suffix = suffix

    def __getitem__(self, index):
        return os.path.join(self.suffix, self.paths_raw[index])
    def __len__(self):
        return len(self.paths_raw)


# =========================
# 工具：读取 path.txt
# =========================
def load_paths_txt(path_txt: str) -> List[Optional[str]]:
    """
    读取 path.txt：
    - 单列：每行即路径（行号=sdpc_id）
    """
    paths: List[Optional[str]] = []
    with open(path_txt, "r", encoding="utf-8") as f:
        paths_raw = [os.path.basename(ln.rstrip("\n")).replace('.h5', '.svs') for ln in f]
        paths = PathAccessor(paths_raw, suffix='/mnt/local09/svs_data')
    return paths


# =========================
# 工具：以 memmap 方式打开 npy 索引
# =========================
def open_patch_index_memmap(npy_path: str):
    """
    用 np.load(mmap_mode='r') 打开 .npy（零拷贝映射）。
    兼容：
      - 结构化：字段名包含 h5_id/patch_size/x/y/level（忽略大小写和细微变体）
      - 普通二维：(N, >=5)，列顺序同上
    返回 (arr, is_structured, field_names_dict)
    """
    data = np.load(npy_path, mmap_mode="r")
    return data


def _get_worker_id() -> int:
    try:
        info = torch.utils.data.get_worker_info()
        return info.id if info is not None else -1
    except Exception:
        return -1



def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    # 兜底：torchrun 也会注环境变量
    return int(os.environ.get("RANK", -1))

# =========================
# Dataset（memmap 索引）
# =========================
class WsiPatchDataset(Dataset):
    """
    返回：
      {
        "image": Tensor[C,H,W] (0..1),
        "path": str,
        "sdpc_id": int, "x": int, "y": int, "level": int, "patch_size": int,
        "index": int
      }
    """
    def __init__(self,
                 patch_npy: str = '/mnt/local09/train/crb/npu_adp/data_npu/svs_index/index.npy',
                 path_txt: str='/mnt/local09/train/crb/npu_adp/data_npu/svs_index/path.txt',
                 pil_transform=None,
                 tensor_transform=None,
                 lru_max_open: int = 32,
                 log_dir: Optional[str] = '/mnt/data/train/crb/train_out/train_test/log_dir',
                 advance: Optional[int] = None,
                 show_max_num: int = 5,
                 fix_size=None,
                 return_dic: bool=False,
                 shared_val=None,
            ):
        super().__init__()
        # 关键：以 memmap 方式打开，不将索引载入内存
        self.patch_npy = patch_npy
        self.path_txt = path_txt
        self._data = None
        self._paths: List[Optional[str]] = None

        self.pil_transform = pil_transform
        self.tensor_transform = tensor_transform
        self._lru_max_open = lru_max_open
        self.log_dir = log_dir 
        self._slide_cache: Optional[SlideLRU] = None  # 每个 worker 自己初始化
        self.advance = advance
        self.rank = get_rank()
        self.return_dic = return_dic
        self.show_max_num = show_max_num
        self.showed_num = 0 
        self.fix_size = fix_size
        self.shared_val = shared_val
        print(f"dataset init success, rank: {self.rank}")

    def _ensure_data(self):
        if self._data is None:
            self._data = open_patch_index_memmap(self.patch_npy)
            self._paths: List[Optional[str]] = load_paths_txt(self.path_txt)

    def __len__(self) -> int:
        self._ensure_data()
        return int(self._data.shape[0]) if not self.advance else len(self._data) - self.advance

    def _ensure_cache(self):
        if self._slide_cache is None:
            wid = _get_worker_id()
            self._slide_cache = SlideLRU(
                max_open=self._lru_max_open,
                shared_val=self.shared_val,
                open_fn=None,                    # 默认 openslide.OpenSlide
            )

    def _get_row(self, i: int):
        if self.advance: i += self.advance
        row = self._data[i]
        sid = int(row["h5_id"])
        s = int(row["patch"])
        x = int(row["x"])
        y = int(row["y"])
        return sid, s, x, y
    
    def _getslice(self, sl: slice):
        """处理切片：内部仍然调用 __getitem__"""
        return [self[i] for i in range(*sl.indices(len(self)))]
    
    def __getitem__(self, index: int) -> Optional[Dict[str, Any]]:
        self._ensure_data()
        if isinstance(index, slice):
            return self._getslice(index)
        self._ensure_cache()

        sid, s, x, y = self._get_row(index)
        svs_path = self._paths[sid]
        assert svs_path not in (None, ""), f"Missing path for sdpc_id={sid}"

        try:
            slide = self._slide_cache.get(svs_path)
            # todo: process x,y (s,s) 越界
            if self.fix_size is None:
                
                img: Image.Image = slide.read_region((x, y), 0, (s, s)).convert("RGB")
            else:
                img: Image.Image = slide.read_region((x, y), 0, (self.fix_size, self.fix_size)).convert("RGB")
            if self.pil_transform is not None:
                img = self.pil_transform(img)

            if self.tensor_transform is not None:
                img = self.tensor_transform(img)
            else:
                img = wsi_transform(img)
            img_dic = {
                "image": img,
                "path": svs_path,
                "sdpc_id": sid,
                "x": x, "y": y, "level": 0, "patch_size": s if self.fix_size is None else self.fix_size,
                "index": index,
            }
            return  img_dic if self.return_dic else img, None
        except Exception:
            # 坏 patch 直接丢弃；collate 跳过 None
            return None
        
    def generate_augument_img(self):
        augument_dir = creat_subdir(
            base_dir=self.log_dir,
            prefix='augument',
            time=True
        )
        flag = self.return_dic
        self.return_dic = True
        for index in range(self.show_max_num):
            img_dic= self[index]
            img = img_dic["image"]
            index_dir = os.path.join(augument_dir, f"index_{index}")
            os.makedirs(index_dir, exist_ok=True)
            save_index(
                img=img,
                out_dir=index_dir,
                prefix='augmentation'
            )
        logger.info(f'index: 0-{self.show_max_num-1} augmentations saved to {augument_dir}')
        self.return_dic = flag

# =========================
# Collate & Loader & Probe
# =========================
def collate_skip_none(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    meta = {k: [b[k] for b in batch] for k in batch[0] if k != "image"}
    return {"image": imgs, **meta}

def dict_collate_fn(batch):
    """
    batch: list[dict[str, Any]]
    例如:
    [
        {"image": tensor1, "path": "a.svs", "x": 12, "y": 55},
        {"image": tensor2, "path": "b.svs", "x": 15, "y": 60},
        ...
    ]
    """
    if not isinstance(batch[0], dict):
        return batch
    collated = {}

    # 获取所有的 key
    keys = batch[0].keys()
    for k in keys:
        values = [b[k] for b in batch]
        # 根据值类型不同分别处理
        if torch.is_tensor(values[0]):
            collated[k] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            collated[k] = torch.tensor(values)
        else:
            # 字符串、路径、元组等不做堆叠
            collated[k] = values
    return collated



wsi_transform = v2.Compose([    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) not in ("spawn", "forkserver"):
    mp.set_start_method("spawn", force=True)

def make_data_loader_wsi(
        dataset,
        batch_size=64,
        num_workers: int=16,
        shuffle=True,
        seed: int = 87,  # TODO: Fix this -- cfg.train.seed
        sampler_type = SamplerType_WSI.HASH_CHUNK,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        chunked: bool = True,
        chunk_size: int = 30,
        collate_fn = None,
    ):
    sampler_advance = int(sampler_advance)
    if sampler_type == SamplerType_WSI.GLOBAL_CHUNK:
        sampler = ShardedChunkedInfiniteSampler(
        seed=seed,
        sample_count=len(dataset),
        advance=sampler_advance,
        chunk_size=chunk_size if chunked else 1,
        shuffle=shuffle,
        shuffle_mode="global"
    )
    elif sampler_type == SamplerType_WSI.BLOCK_CHUNK:
        # 中数据（1e7~1e9）
        sampler = ShardedChunkedInfiniteSampler(
            sample_count=len(dataset),
            seed=seed,
            advance=sampler_advance,
            chunk_size=chunk_size if chunked else 1,
            shuffle=shuffle,
            shuffle_mode="block",
            block_size=10**6
        )
    elif sampler_type == SamplerType_WSI.HASH_CHUNK:
        # 超大数据（>=1e9）
        sampler = ShardedChunkedInfiniteSampler(
            sample_count=len(dataset),
            seed=seed,
            advance=sampler_advance,
            chunk_size=chunk_size if chunked else 1,
            shuffle=shuffle,
            shuffle_mode="hash"
        )
    logger.info(f'len: {len(dataset)}, sampler_type: {sampler_type}')
    if collate_fn is None:
        collate_fn = dict_collate_fn
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        multiprocessing_context="spawn",
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return loader




def quick_probe(loader: DataLoader, steps: int = 200):
    """轻量吞吐探针（不打印每批，避免干扰）"""
    from itertools import islice
    t0 = time.perf_counter(); n = 0
    key = True
    if steps > 0:
        for b in tqdm(islice(loader, steps)):
            if key:
                print(b['image'].shape)
                key = False
            if b is None:
                continue
            n += int(b["image"].shape[0])
    else:
        for b in tqdm(loader):
            if key:
                print(b['image'].shape)
                key = False
            if b is None:
                continue
            n += int(b["image"].shape[0])
    t1 = time.perf_counter()
    sec = max(1e-9, t1 - t0)
    print(f"[probe] {n/sec:.1f} patch/s  ({1000*sec/max(1,n):.3f} ms/patch) over {n} patches")



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("WSI Patch Dataset (memmap) probe")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=32)
    args = ap.parse_args()

    ds = WsiPatchDataset(
        pil_transform=None,
        tensor_transform=None,
        return_dic=False
    )

    loader = make_data_loader_wsi(
        dataset=ds
    )
    print_max_time, print_time = 5, 0
    for d in loader:
        if isinstance(d, dict):
            print(f'index: {d["index"]}')
        else:
            print(len(d))
        print_time += 1
        if print_time >= print_max_time:
            break
        pass

