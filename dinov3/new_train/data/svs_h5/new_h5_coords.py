#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大规模 H5 coords 抽样 → 生成 path.txt 与 index.npy（memmap 可并行写）
- 默认递归扫描 --root 下的 .h5；也可用 --list 传入已存在的路径列表（每行一个）
- 每个 h5 的 coords 形状假设为 (N, 2) 或 (N, >=2)，取前两列为 (x, y)
- 随机种子可设定，保证可复现
"""

import os
import sys
import argparse
import traceback
from typing import List, Tuple, Dict

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import h5py
import multiprocessing as mp


DTYPE = np.dtype([
    ("h5_id", np.int32),
    ("x",     np.int32),
    ("y",     np.int32),
    ("patch", np.int16),
])


def find_h5_files(root: str) -> List[str]:
    files = []
    for r, _, fs in os.walk(root):
        for name in fs:
            if name.lower().endswith(".h5") or name.lower().endswith(".hdf5"):
                files.append(os.path.abspath(os.path.join(r, name)))
    files.sort()
    return files


def read_list_file(list_txt: str) -> List[str]:
    paths = []
    with open(list_txt, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(os.path.abspath(p))
    return paths


def _probe_len_one(args) -> Tuple[int, int]:
    """
    返回 (index_in_list, N_coords)。若失败，返回 (index_in_list, 0)。
    """
    idx, path, dataset_key = args
    try:
        with h5py.File(path, "r") as f:
            if dataset_key not in f:
                return idx, 0
            dset = f[dataset_key]
            n = int(dset.shape[0])
            return idx, n
    except Exception:
        return idx, 0


def _normalize_patch_probs(patch_dict: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    sizes = np.array(list(patch_dict.keys()), dtype=np.int32)
    probs = np.array(list(patch_dict.values()), dtype=np.float64)
    if np.any(probs < 0):
        raise ValueError("patch 概率不能为负数")
    s = probs.sum()
    if s <= 0:
        raise ValueError("patch 概率之和必须为正")
    probs = probs / s
    return sizes, probs


def _worker_sample_and_write(args) -> Tuple[int, int, str]:
    """
    在子进程中执行：
      - 打开 h5，随机采样 idx
      - 提取 coords[idx][:,0:2] 作为 x,y
      - 采样 patch
      - 打开 index.npy memmap（r+），写入 [start:start+k]
    返回 (h5_id, 写入行数, 错误信息或空串)
    """
    (h5_id, path, dataset_key, k, npy_path, total_rows,
     start, patch_sizes, patch_probs, seed) = args
    if k <= 0:
        return h5_id, 0, ""

    try:
        # 独立 RNG：与 h5_id 绑定
        rng = np.random.default_rng(int(seed) + int(h5_id) * 1000003)

        with h5py.File(path, "r") as f:
            dset = f[dataset_key]
            n = int(dset.shape[0])
            if k >= n:
                idx = np.arange(n, dtype=np.int64)
            else:
                idx = rng.choice(n, size=k, replace=False)
            idx.sort()  # 提升 HDF5 随机行访问效率

            # 仅取 x,y 两列
            coords = dset[idx]
            if coords.ndim == 1:
                # 假如 coords 是 (N,) 的对象/引用，尝试展开
                coords = np.stack(coords, axis=0)
            if coords.shape[1] < 2:
                raise ValueError(f"{path}::{dataset_key} 第二维 < 2，无法提取 (x,y)")

            xy = coords[:, :2]
            # 若为浮点，取 floor/round；这里用 round
            if not np.issubdtype(xy.dtype, np.integer):
                xy = np.rint(xy).astype(np.int32, copy=False)
            else:
                xy = xy.astype(np.int32, copy=False)

            patches = rng.choice(patch_sizes, size=xy.shape[0], replace=True, p=patch_probs).astype(np.int32)

        # 打开 .npy memmap 并写入切片
        mm = open_memmap(npy_path, mode="r+", dtype=DTYPE, shape=(total_rows,))
        out = np.empty(xy.shape[0], dtype=DTYPE)
        out["h5_id"] = np.int32(h5_id)
        out["x"] = xy[:, 0]
        out["y"] = xy[:, 1]
        out["patch"] = patches

        mm[start:start + xy.shape[0]] = out  # 并行写不同切片，互不冲突
        del mm  # 及时 flush
        return h5_id, int(xy.shape[0]), ""
    except Exception as e:
        err = f"[h5_id={h5_id}] {path} ERROR: {repr(e)}\n{traceback.format_exc(limit=1)}"
        return h5_id, 0, err


def build_index(
    paths: List[str],
    out_dir: str,
    dataset_key: str = "coords",
    sample_per_file: int = 2000,
    patch_dict: Dict[int, float] = None,
    seed: int = 123,
    num_workers: int = None,
) -> Tuple[str, str]:
    """
    主流程：两阶段
      1) 并行探测每个 h5 的 coords 行数，计算每文件采样量 k_i 与写入偏移 start_i
      2) 并行读取/采样并写入 memmap .npy
    返回 (path_txt, index_npy)
    """
    os.makedirs(out_dir, exist_ok=True)
    if patch_dict is None:
        patch_dict = {224: 0.3, 336: 0.3, 448: 0.4}
    patch_sizes, patch_probs = _normalize_patch_probs(patch_dict)

    path_txt = os.path.join(out_dir, "path.txt")
    index_npy = os.path.join(out_dir, "index.npy")
    probe_log = os.path.join(out_dir, "probe_errors.log")
    write_log = os.path.join(out_dir, "write_errors.log")

    # 写 path.txt（行号即 h5_id）
    with open(path_txt, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")

    # 阶段 1：并行探测 coords 行数
    print("==> Probing coords length for each H5 ...")
    nproc = num_workers or max(1, min(mp.cpu_count(), 32))
    probe_args = [(i, p, dataset_key) for i, p in enumerate(paths)]
    lengths = [0] * len(paths)
    errors = []

    with mp.Pool(processes=nproc) as pool:
        for i, n in tqdm(pool.imap_unordered(_probe_len_one, probe_args),
                         total=len(probe_args), desc="Probe", dynamic_ncols=True):
            lengths[i] = n

    # 记录探测失败（长度为 0 且文件存在 coords? 可能本来就是 0 或不存在）
    with open(probe_log, "w", encoding="utf-8") as f:
        for i, (p, n) in enumerate(zip(paths, lengths)):
            if n <= 0:
                f.write(f"[WARN] h5_id={i} path={p} coords_len={n}\n")

    # 计算每文件采样量 k_i 与写入偏移 start_i
    counts = [min(n, sample_per_file) if n > 0 else 0 for n in lengths]
    starts = [0] * len(counts)
    total = 0
    for i, c in enumerate(counts):
        starts[i] = total
        total += c
    total_rows = int(total)

    print(f"==> Total rows to write: {total_rows:,d}")
    if total_rows == 0:
        print("No rows to write. Exiting.")
        # 创建一个空壳 index.npy 以方便后续流水线
        _ = open_memmap(index_npy, mode="w+", dtype=DTYPE, shape=(0,))
        return path_txt, index_npy

    # 先创建目标 .npy（带头部），并初始化（可不写入数据）
    mm = open_memmap(index_npy, mode="w+", dtype=DTYPE, shape=(total_rows,))
    # 可选：初始化为 -1 方便后验检查
    mm["h5_id"][:] = -1
    mm["x"][:] = 0
    mm["y"][:] = 0
    mm["patch"][:] = 0
    del mm  # 释放，以便子进程 r+ 访问

    # 阶段 2：并行采样与写入
    print("==> Sampling coords & writing index.npy in parallel ...")
    worker_args = []
    for i, p in enumerate(paths):
        k = counts[i]
        if k <= 0:
            continue
        worker_args.append((
            i, p, dataset_key, k, index_npy, total_rows, starts[i],
            patch_sizes, patch_probs, int(seed),
        ))

    wrote_rows = 0
    with mp.Pool(processes=nproc) as pool, open(write_log, "w", encoding="utf-8") as flog:
        for h5_id, wrote, err in tqdm(pool.imap_unordered(_worker_sample_and_write, worker_args),
                                      total=len(worker_args), desc="Write", dynamic_ncols=True):
            wrote_rows += wrote
            if err:
                flog.write(err + "\n")

    print(f"==> Done. Planned rows={total_rows:,d}, actually written={wrote_rows:,d}")
    if wrote_rows != total_rows:
        print("WARNING: actual written rows != planned rows，请检查 write_errors.log 中的错误。")

    return path_txt, index_npy


def parse_patch_dict(s: str) -> Dict[int, float]:
    """
    解析形如 "224:0.3,336:0.3,448:0.4" 的字符串
    """
    out = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(":")
        out[int(k)] = float(v)
    return out


def main():
    parser = argparse.ArgumentParser(description="抽样 H5 coords → 生成 path.txt & index.npy")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", type=str, help="递归扫描 .h5 的根目录")
    g.add_argument("--list", type=str, help="包含 h5 绝对路径的 txt 文件（每行一个）")

    parser.add_argument("--out-dir", type=str, required=True, help="输出目录（将写入 path.txt 与 index.npy）")
    parser.add_argument("--dataset-key", type=str, default="coords", help="H5 中 coords 数据集的键名，默认 'coords'")
    parser.add_argument("--per-file", type=int, default=2500, help="每个 h5 采样数量上限，默认 2500")
    parser.add_argument("--patch", type=str, default="224:0.3,336:0.2,448:0.3,1024:0.2",
                        help='patch 概率字典，如 "224:0.3,336:0.3,448:0.4"')
    parser.add_argument("--seed", type=int, default=123, help="随机种子")
    parser.add_argument("--workers", type=int, default=32, help="并行进程数（默认 0=自动<=32）")

    args = parser.parse_args()

    if args.list:
        paths = read_list_file(args.list)
    else:
        paths = find_h5_files(args.root)

    if len(paths) == 0:
        print("未找到任何 .h5 文件", file=sys.stderr)
        sys.exit(1)

    patch_dict = parse_patch_dict(args.patch)
    num_workers = None if args.workers <= 0 else args.workers

    print(f"Found {len(paths)} h5 files.")
    path_txt, index_npy = build_index(
        paths=paths,
        out_dir=args.out_dir,
        dataset_key=args.dataset_key,
        sample_per_file=args.per_file,
        patch_dict=patch_dict,
        seed=args.seed,
        num_workers=num_workers,
    )

    print(f"\nOutputs:\n  path.txt -> {path_txt}\n  index.npy -> {index_npy}")
    print("用法：后续可用 numpy.load(index.npy, mmap_mode='r') 读取；通过 path[h5_id], x, y, patch 唯一定位。")


if __name__ == "__main__":
    main()
