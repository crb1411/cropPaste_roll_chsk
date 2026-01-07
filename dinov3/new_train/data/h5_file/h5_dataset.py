import os
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

from collections import OrderedDict
from torch.utils.data import DataLoader
from typing import Optional
import random
import logging 
from PIL import Image
import traceback
from dinov3.new_train.data.h5_file.file_lock_h5 import append_success_log, ensure_chunk1

logger = logging.getLogger('dinov3')
# ==== 索引 dtype 定义 ====
INDEX_DTYPE = np.dtype([
    ("h5_id", np.int32),
    ("patch_id", np.int16)
])

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # 拿到这个 worker 对应的 Dataset
    dataset.cache = H5Cache(max_open=dataset.max_open)  # 每个 worker 自己一个 cache


class H5Cache:
    def __init__(self, max_open=1000):
        self.cache = OrderedDict()
        self.max_open = max_open
        

    def get(self, path):
        # 如果文件已在缓存，就更新位置（标记为最近使用）
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        # 否则打开新文件
        if len(self.cache) >= self.max_open:
            old_path, old_file = self.cache.popitem(last=False)  # 移除最久没用的
            old_file.close()
        
        try:
            f = h5py.File(path, "r", swmr=True, libver="latest")
            self.cache[path] = f
            return f
        except Exception as e:
            logger.error(f"Failed to open file {path}: {e}")
            return None

from torchvision.transforms import v2
wsi_transform = v2.Compose([    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

def worker_init_fn_h5(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.cache = None
    dataset.bad_h5_files = set()
    dataset.bad_indices = set()


class H5Dataset_NoCache(Dataset):
    def __init__(
            self, 
            index_path ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index.npy',          # npy 索引 (structured npy: h5_id, patch_id)
            files_txt ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index_files.txt',
            dataset_key: str = "patches", 
            transform=None, 
            max_open: int = 32,
            subdata_advance: Optional[int] = None,
            chunk_tras = False,
        ):
        """
        Args:
            index_npy (str): npy 索引文件，dtype=[("h5_id",np.int32),("patch_id",np.int32)]
            files_txt (str): 保存所有 h5 路径的 txt 文件
            dataset_key (str): HDF5 数据集的 key
        """
        self.files_txt = files_txt
        self.index_path = index_path
        self.subdata_advance = subdata_advance
        self.dataset_key = dataset_key
        self.transform = transform
        self.max_open = max_open
        self.cache = None
        self.index = None
        self.file_path_list = None
        self.bad_h5_files: set[str] = None
        # 坏的 index（__getitem__ 的 idx）
        self.bad_indices: set[int] = None
        self.chunk_tras = chunk_tras

    def _ensure_data(self):
        if self.index is None:
            self.bad_h5_files: set[str] = set()
            # 坏的 index（__getitem__ 的 idx）
            self.bad_indices: set[int] = set()
            with open(self.files_txt, "r", encoding="utf-8") as f:
                self.file_path_list = [line.strip() for line in f if line.strip()]
            full_index = np.memmap(self.index_path, dtype=INDEX_DTYPE, mode="r")

            if self.subdata_advance is not None and 0 < self.subdata_advance < len(full_index):
                self.index = full_index[self.subdata_advance:]
            else:
                self.index = full_index

    def __len__(self):
        self._ensure_data()
        return len(self.index)
    # ====== 新增：从全量 index 中随机挑选一个“好的” idx ======
    def _sample_good_index(self) -> int:
        n = len(self.index)

        # 尝试若干次随机采样，避开坏 index
        max_random_try = 50
        for _ in range(max_random_try):
            new_idx = random.randint(0, n - 1)
            if new_idx not in self.bad_indices:
                return new_idx

        # 如果随机尝试仍然没找到（说明坏的比例已经很高了），
        # 做一次线性扫描，找到第一个没被标坏的 index
        for new_idx in range(n):
            if new_idx not in self.bad_indices:
                return new_idx

        # 兜底：如果真的所有都坏了，就直接报错
        raise RuntimeError("H5Dataset: all indices seem to be bad, no valid sample left.")

    def __getitem__(self, idx):
        self._ensure_data()
        rec = self.index[idx]
        file_idx = int(rec["h5_id"])
        patch_idx = int(rec["patch_id"])
        
        file_path = self.file_path_list[file_idx]
        if file_path in self.bad_h5_files:
            self.bad_indices.add(idx)
            idx = self._sample_good_index()
            return self.__getitem__(idx)
        try:
            if self.chunk_tras:
                ensure_chunk1(file_path=file_path)
            with h5py.File(file_path, mode="r", swmr=True) as h5_data:
                patch = Image.fromarray(h5_data[self.dataset_key][patch_idx]).convert("RGB")
            if self.transform is not None:
                patch = self.transform(patch)
            else:
                patch = wsi_transform(patch)
            return patch, None
        except Exception as e:
            self.bad_indices.add(idx)
            self.bad_h5_files.add(file_path)
            err = traceback.format_exc()
            logger.info("H5Dataset: error reading file %s, patch %d, error: %s", file_path, patch_idx, err)
            logger.info("H5Dataset: bad sample at index %d, file %s, patch %d", idx, file_path, patch_idx)
            idx_new = self._sample_good_index()
            return self.__getitem__(idx_new)


class H5Dataset(Dataset):
    def __init__(
            self, 
            index_path ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index.npy',          # npy 索引 (structured npy: h5_id, patch_id)
            files_txt ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index_files.txt',
            dataset_key: str = "patches", 
            transform=None, 
            max_open: int = 3824,
            subdata_advance: Optional[int] = None,
            chunk_tras=False,
        ):
        """
        Args:
            index_npy (str): npy 索引文件，dtype=[("h5_id",np.int32),("patch_id",np.int32)]
            files_txt (str): 保存所有 h5 路径的 txt 文件
            dataset_key (str): HDF5 数据集的 key
        """
        #  用 memmap 方式加载索引
        
        # 读文件路径
        self.files_txt = files_txt
        self.index_path = index_path
        self.subdata_advance = subdata_advance
        self.dataset_key = dataset_key
        self.transform = transform
        self.max_open = max_open
        self.cache = None
        self.file_path_list = None
        self.index = None
        # ====== 新增：坏样本缓存 ======
        # 坏的 h5 文件路径
        self.bad_h5_files: set[str] = None
        # 坏的 index（__getitem__ 的 idx）
        self.bad_indices: set[int] = None
        self.chunk_tras = chunk_tras

    def _ensure_data(self):
        if self.index is None:
            # ====== 新增：坏样本缓存 ======
            # 坏的 h5 文件路径
            self.bad_h5_files: set[str] = set()
            # 坏的 index（__getitem__ 的 idx）
            self.bad_indices: set[int] = set()
            with open(self.files_txt, "r", encoding="utf-8") as f:
                self.file_path_list = [line.strip() for line in f if line.strip()]
            full_index = np.memmap(self.index_path, dtype=INDEX_DTYPE, mode="r")

            #  如果 subdata_advance 不为 None，就做切片
            if self.subdata_advance is not None and 0 < self.subdata_advance < len(full_index):
                self.index = full_index[self.subdata_advance:]
            else:
                self.index = full_index

    def _ensure_cache(self):
        if self.cache is None:
            self.cache = H5Cache(max_open=self.max_open)

    def __len__(self):
        self._ensure_data()
        return len(self.index)
    # ====== 新增：从全量 index 中随机挑选一个“好的” idx ======
    def _sample_good_index(self) -> int:
        n = len(self.index)

        # 尝试若干次随机采样，避开坏 index
        max_random_try = 50
        for _ in range(max_random_try):
            new_idx = random.randint(0, n - 1)
            if new_idx not in self.bad_indices:
                return new_idx

        # 如果随机尝试仍然没找到（说明坏的比例已经很高了），
        # 做一次线性扫描，找到第一个没被标坏的 index
        for new_idx in range(n):
            if new_idx not in self.bad_indices:
                return new_idx

        # 兜底：如果真的所有都坏了，就直接报错
        raise RuntimeError("H5Dataset: all indices seem to be bad, no valid sample left.")

    def __getitem__(self, idx):
        self._ensure_data()
        rec = self.index[idx]
        file_idx = int(rec["h5_id"])
        patch_idx = int(rec["patch_id"])
        

        file_path = self.file_path_list[file_idx]
        self._ensure_cache()
        if file_path in self.bad_h5_files:
            self.bad_indices.add(idx)
            idx = self._sample_good_index()
            return self.__getitem__(idx)
        try:
            h5f = self.cache.get(file_path)
            patch = Image.fromarray(h5f[self.dataset_key][patch_idx]).convert("RGB")
            if self.transform is not None:
                patch = self.transform(patch)
            else:
                patch = wsi_transform(patch)
            return patch, None
        except Exception as e:
            self.bad_indices.add(idx)
            self.bad_h5_files.add(file_path)
            logger.info("H5Dataset: bad sample at index %d, file %s, patch %d", idx, file_path, patch_idx)
            idx_new = self._sample_good_index()
            return self.__getitem__(idx_new)

class Path_suffix:
    def __init__(self, file_path: str, suffix: str = "", patches_per_file: int = 2000,
                 advance: Optional[int] = None):
        with open(file_path, "r", encoding="utf-8") as f:

            self.file_path_list = [line.strip() for line in f if line.strip()]
            if advance is not None:
                self.file_path_list = self.file_path_list[advance:]
                print(f"advance {advance} files")
                print(f"total {len(self.file_path_list)} files")
                if len(self.file_path_list) >0:
                    print('The first 10 files:', self.file_path_list[:min(10, len(self.file_path_list))])
        self.suffix = suffix
        self.patches_per_file = patches_per_file
        
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.file_path_list):
            raise IndexError("Index out of range")
        path_h5 = os.path.join(self.suffix, self.file_path_list[idx])
        if not os.path.exists(path_h5):
            raise FileNotFoundError(f"File not found: {path_h5}")
        return path_h5

    def __len__(self):
        return len(self.file_path_list) 



if __name__ == "__main__": 
    ds = H5Dataset(
            index_path ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index.npy',          # npy 索引 (structured npy: h5_id, patch_id)
            files_txt ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index_files.txt',
            dataset_key = "patches", 
            transform=None, 
            max_open = 32,
            subdata_advance = None,
        )
    print(len(ds))
    print(ds[0])
    
