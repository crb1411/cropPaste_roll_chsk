import os
from tqdm import tqdm

import h5py
import numpy as np
from multiprocessing import Pool, cpu_count


def list_h5_files(root_dir, out_txt):
    with open(out_txt, "w", encoding="utf-8") as f, tqdm(desc="Scanning .h5 files", unit="file", total=None) as pbar:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(".h5"):
                    abs_path = os.path.abspath(os.path.join(dirpath, fname))
                    f.write(abs_path + "\n")
                    pbar.update(1)   # 每找到一个文件就更新进度条
    print(f"Done List saved to {out_txt}")




# ==== 索引 dtype 定义 ====
INDEX_DTYPE = np.dtype([
    ("h5_id", np.int32),
    ("patch_id", np.int16)
])

bad_h5 = []
def count_patches(path):
    try:
        with h5py.File(path, "r") as f:
            return (path, len(f["patches"]), None)   # (路径, 计数, 错误)
    except Exception as e:
        return (path, None, repr(e))

def scan_h5_dir(file_path, num_workers=None):
    """多进程扫描所有 h5 文件"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    if file_path.endswith('.txt'):
        with open(file_path, "r", encoding="utf-8") as f:
            file_list = [line.strip() for line in f.readlines()]
    else:
        import pandas as pd
        file_list = pd.read_csv(file_path)['h5_path'].to_list()
        file_list = [os.path.join('/mnt/unicom', h5_ph) for h5_ph in file_list]

    print(f'len h5: {len(file_list)}')

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(count_patches, file_list), total=len(file_list)))

    ok   = [(p, n) for p, n, err in results if n is not None]
    file_list, patch_counts = zip(*ok) if ok else ([], [])
    bad  = [(p, err) for p, n, err in results if n is None]
    print(f"Found {len(ok)} h5 files, {len(bad)} bad h5 files")
    # bad_h5, _ = zip(*[(f, p) for f, p in zip(file_list, patch_counts) if p is None])

    # print(f"Found {len(file_list)} h5 files, {len(bad_h5)} bad h5 files")
    # 更新 file_list 
    if len(bad)>0:
        file_path_end = file_path.split(".")[-1]
        print(f"bad saved -> {file_path.replace(f'.{file_path_end}', '_bad_data.txt')}")
        with open(file_path.replace(f'.{file_path_end}', '_bad_data.txt'), "w", encoding="utf-8") as f:
            for p, err in bad:
                f.write(f"{p}\t{err}\n")
    return file_list, patch_counts

def _write_index_chunk(args):
    file_idx, n_patches, offset, mmap_path = args
    mmap = np.memmap(mmap_path, dtype=INDEX_DTYPE, mode="r+")
    # 批量生成数据
    arr = np.empty(n_patches, dtype=INDEX_DTYPE)
    arr["h5_id"] = file_idx
    arr["patch_id"] = np.arange(n_patches, dtype=np.int32)
    # 一次性写入
    mmap[offset:offset+n_patches] = arr
    del mmap  # 自动刷盘

def build_index_memmap(file_list, patch_counts, out_npy="index.npy", num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = sum(patch_counts)
    print(f"Total patches: {total:,}")
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    # ==== 预分配 memmap ====
    mmap = np.memmap(out_npy, dtype=INDEX_DTYPE, mode="w+", shape=(total,))
    del mmap  # 关闭句柄，子进程才能访问

    # ==== 准备任务 ====
    tasks, offset = [], 0
    for file_idx, n_patches in enumerate(patch_counts):
        tasks.append((file_idx, n_patches, offset, out_npy))
        offset += n_patches

    # ==== 多进程写入 ====
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(_write_index_chunk, tasks), total=len(tasks)))

    # ==== 保存文件列表 ====
    with open(out_npy.replace(".npy", "_files.txt"), "w") as f:
        for path in file_list:
            f.write(path + "\n")

    print(f"Index saved to {out_npy}, dtype={INDEX_DTYPE}, shape=({total},)")




    
if __name__ == "__main__":
    # list_h5_files("/mnt/unicom/bundles", "/mnt/crb/vl/test_code/h5_file/h5_list.txt")
    # Step1: 扫描所有 h5 文件，得到每个文件 patch 数

    file_list, patch_counts = scan_h5_dir("/mnt/local09/train/crb/data/h5_data_unicom_nas/finish_h5.txt", num_workers=128)

    # # Step2: 建立全局索引 (memmap)
    build_index_memmap(file_list, patch_counts, out_npy="/mnt/local09/train/crb/data/h5_data_unicom_nas_250k/h5_patch_index.npy", num_workers=128)

    # Step3: 读取索引
    index_path = "/mnt/local09/train/crb/data/h5_data_unicom_nas_250k/h5_patch_index.npy"
    index = np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")
    print(index.shape, index.dtype)
    print(index[:5])  # [(0,0) (0,1) (0,2) ...]
    print(index["h5_id"][:5], index["patch_id"][:5])

