import os
from tqdm import tqdm

import numpy as np
from multiprocessing import Pool, cpu_count


# ==== 索引 dtype 定义 ====
INDEX_DTYPE = np.dtype([
    ("h5_id", np.int32),      # 这里名字保持一致，不改接口
    ("patch_id", np.int32)
])


def list_npy_files(root_dir, out_txt):
    """遍历目录下所有 .npy 文件并保存到 txt 中"""
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f, tqdm(desc="Scanning .npy files") as pbar:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(".npy"):
                    abs_path = os.path.abspath(os.path.join(dirpath, fname))
                    f.write(abs_path + "\n")
                    pbar.update(1)
    print(f"Done! List saved to {out_txt}")


# =============================================================
# 单个 NPY 文件的 patch 数统计器：返回 (path, count, err)
# =============================================================
def count_patches_npy(path):
    try:
        arr = np.load(path, mmap_mode="r")   # mmap 不会加载全量
        return (path, arr.shape[0], None)
    except Exception as e:
        return (path, None, repr(e))


# =============================================================
# 多进程扫描所有 NPY 文件
# 接口保持一致：返回 (file_list, patch_counts)
# =============================================================
def scan_npy_dir(file_path, num_workers=None):
    """
    输入 file_path 既可以是 txt，也可以是 CSV（与原版本保持一致）
    返回:
        file_list: List[str]
        patch_counts: List[int]
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # ---- 加载文件列表 ----
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            file_list = [line.strip() for line in f.readlines()]
    else:
        import pandas as pd
        file_list = pd.read_csv(file_path)["h5_path"].to_list()
        # 可选处理（保持行为一致）
        file_list = [os.path.join('/mnt/unicom', p) for p in file_list]

    print(f"len npy files: {len(file_list)}")

    # ---- 多进程统计 ----
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(count_patches_npy, file_list), total=len(file_list)))

    # ---- 分类 ----
    ok = [(p, n) for p, n, err in results if n is not None]
    file_list, patch_counts = zip(*ok) if ok else ([], [])

    bad = [(p, err) for p, n, err in results if n is None]
    print(f"Found {len(ok)} valid npy files, {len(bad)} bad npy files")

    if len(bad) > 0:
        out_path = file_path.replace(".csv", "_bad_data.txt").replace(".txt", "_bad_data.txt")
        print(f"bad saved -> {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            for p, err in bad:
                f.write(f"{p}\t{err}\n")

    return list(file_list), list(patch_counts)


# =============================================================
# 与原代码一致，无需改动
# =============================================================
def _write_index_chunk(args):
    file_idx, n_patches, offset, mmap_path = args
    mmap = np.memmap(mmap_path, dtype=INDEX_DTYPE, mode="r+")
    arr = np.empty(n_patches, dtype=INDEX_DTYPE)

    arr["h5_id"] = file_idx
    arr["patch_id"] = np.arange(n_patches, dtype=np.int32)

    mmap[offset:offset + n_patches] = arr
    del mmap


def build_index_memmap(file_list, patch_counts, out_npy="index.npy", num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = sum(patch_counts)
    print(f"Total patches: {total:,}")

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)

    # ---- 预分配 memmap ----
    mmap = np.memmap(out_npy, dtype=INDEX_DTYPE, mode="w+", shape=(total,))
    del mmap

    # ---- 分发任务 ----
    tasks = []
    offset = 0
    for file_idx, n_patches in enumerate(patch_counts):
        tasks.append((file_idx, n_patches, offset, out_npy))
        offset += n_patches

    # ---- 多进程写入 ----
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(_write_index_chunk, tasks), total=len(tasks)))

    # ---- 保存 NP 文件路径列表 ----
    with open(out_npy.replace(".npy", "_files.txt"), "w") as f:
        for p in file_list:
            f.write(p + "\n")

    print(f"Index saved → {out_npy}, dtype={INDEX_DTYPE}, shape=({total},)")


# =============================================================
# 主函数
# =============================================================
if __name__ == "__main__":
    # file_list, patch_counts = scan_npy_dir("/path/to/npy_list.txt", num_workers=128)
    list_npy_files(root_dir='/mnt/local08/data_new', 
                   out_txt="/mnt/local09/train/crb/data/h5_data_local08/npy_all.txt")
    file_list, patch_counts = scan_npy_dir(
        "/mnt/local09/train/crb/data/h5_data_local08/npy_all.txt",
        num_workers=128
    )

    build_index_memmap(
        file_list,
        patch_counts,
        out_npy="/mnt/local09/train/crb/data/h5_data_local08/npy_patch_index.npy",
        num_workers=128
    )

    index = np.memmap("/mnt/local09/train/crb/data/h5_data_local08/npy_patch_index.npy",
                      dtype=INDEX_DTYPE, mode="r")

    print(index.shape, index.dtype)
    print(index[:5])
    print(index["h5_id"][:5], index["patch_id"][:5])
