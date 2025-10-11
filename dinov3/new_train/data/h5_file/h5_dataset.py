import os
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch 
from collections import OrderedDict
from torch.utils.data import DataLoader
from typing import Optional

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
        
        
        f = h5py.File(path, "r", swmr=True, libver="latest")
        self.cache[path] = f
        return f
from torchvision import transforms
wsi_transform = transforms.Compose([          
    transforms.ToTensor(),
])

class H5Dataset(Dataset):
    def __init__(
            self, 
            index_path: str,          # npy 索引 (structured npy: h5_id, patch_id)
            files_txt: str,          # 保存 h5 文件路径
            dataset_key: str = "patches", 
            transform=None, 
            max_open: int = 1000,
            subdata_advance: Optional[int] = None,
        ):
        """
        Args:
            index_npy (str): npy 索引文件，dtype=[("h5_id",np.int32),("patch_id",np.int32)]
            files_txt (str): 保存所有 h5 路径的 txt 文件
            dataset_key (str): HDF5 数据集的 key
        """
        #  用 memmap 方式加载索引
        full_index = np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")

        #  如果 subdata_advance 不为 None，就做切片
        if subdata_advance is not None and 0 < subdata_advance < len(full_index):
            self.index = full_index[subdata_advance:]
        else:
            self.index = full_index
        # 读文件路径
        with open(files_txt, "r", encoding="utf-8") as f:
            self.file_path_list = [line.strip() for line in f if line.strip()]

        self.dataset_key = dataset_key
        self.transform = transform
        self.max_open = max_open
        self.cache = H5Cache(max_open=max_open)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec = self.index[idx]
        file_idx = int(rec["h5_id"])
        patch_idx = int(rec["patch_id"])

        file_path = self.file_path_list[file_idx]
        h5f = self.cache.get(file_path)

        patch = h5f[self.dataset_key][patch_idx]
        if self.transform is not None:
            patch = self.transform(patch)
        else:
            
            patch = wsi_transform(patch)
        return patch


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
            file_path="file_path.txt", 
            dataset_key="patches_224", 
            patches_per_file=2000, 
            transform=None, 
            max_open=1000
        )
    loader = DataLoader(
        ds,
        batch_size=...,
        num_workers=8,
        worker_init_fn=worker_init_fn,
        shuffle=True
    )
