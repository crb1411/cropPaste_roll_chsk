#!/usr/bin/env python3
import os
import csv
import zlib
import numpy as np
import h5py
import openslide
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import accumulate

# ============================================================
# 常量
# ============================================================
PATCH_H = 224
PATCH_W = 224
PATCH_C = 3
PATCH_BYTES = PATCH_H * PATCH_W * PATCH_C

SHARD_SIZE_BYTES = 2 * 1024**4   # 2TB
SHARD_CAP = int((SHARD_SIZE_BYTES // PATCH_BYTES) * 0.98)

INDEX_DTYPE = np.dtype([
    ("h5_id", np.int32),
    ("patch_id", np.int32),
    ("global_idx", np.int64),
])

# ============================================================
# CSV
# ============================================================
def read_csv(csv_path):
    h5s, svss = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            h5s.append(r["h5_file_path"])
            svss.append(r["wsi_path"])
    return h5s, svss




# ============================================================
# big_index
# ============================================================
def build_big_index(n_keep, out_dir):
    total = sum(n_keep)
    print(f"[INFO] total patches = {total:,}")

    if total == 0:
        raise RuntimeError("No valid patches found.")

    index_path = out_dir / "big_index.npy"
    index = np.memmap(index_path, dtype=INDEX_DTYPE, mode="w+", shape=(total,))
    offsets = list(accumulate([0] + n_keep))[:-1]

    cur = 0
    for h5_id, k in enumerate(n_keep):
        if k == 0:
            continue
        index[cur:cur+k]["h5_id"] = h5_id
        index[cur:cur+k]["patch_id"] = np.arange(k, dtype=np.int32)
        index[cur:cur+k]["global_idx"] = np.arange(cur, cur+k, dtype=np.int64)
        cur += k

    index.flush()
    del index
    return index_path, offsets


# ============================================================
# shard utils
# ============================================================
def open_shard(path):
    shape = (SHARD_CAP, PATCH_H, PATCH_W, PATCH_C)
    if not path.exists():
        mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
        del mm
    return np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)


# ============================================================
# shard 内单个 H5 任务（子进程）
# ============================================================
def process_one_h5(args):
    h5_id, h5_path, svs_path, coords, K, base = args

    seed = zlib.crc32(h5_path.encode()) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(coords), size=min(K, len(coords)), replace=False)

    slide = openslide.OpenSlide(svs_path)
    patches = np.empty((len(sel), PATCH_H, PATCH_W, PATCH_C), dtype=np.uint8)

    for j, ci in enumerate(sel):
        x, y = coords[ci]
        patch = slide.read_region((int(x), int(y)), 0, (PATCH_W, PATCH_H))
        patches[j] = np.asarray(patch)[..., :3]

    slide.close()
    return base, patches


# ============================================================
# shard worker（shard 内并行）
# ============================================================
def shard_worker(shard_id, h5_ids, h5s, svss, coords_cache, offsets, out_dir, K, workers):
    shard_path = out_dir / f"shard_{shard_id:05d}.npy"
    done_flag = shard_path.with_suffix(".npy.done")

    if done_flag.exists():
        print(f"[INFO] shard {shard_id} done, skip")
        return

    shard_mem = open_shard(shard_path)

    jobs = []
    for h5_id in h5_ids:
        coords = coords_cache[h5_id]
        if coords is None or len(coords) == 0:
            continue
        base = offsets[h5_id] % SHARD_CAP
        jobs.append((
            h5_id,
            h5s[h5_id],
            svss[h5_id],
            coords,
            K,
            base,
        ))

    if jobs:
        with Pool(workers) as pool:
            for base, patches in tqdm(
                pool.imap_unordered(process_one_h5, jobs),
                total=len(jobs),
                desc=f"shard {shard_id}"
            ):
                shard_mem[base: base + len(patches)] = patches

    shard_mem.flush()
    del shard_mem
    done_flag.write_text("done\n")


# ============================================================
# main
# ============================================================
def build_dataset(csv_path, out_dir, K, workers):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5s, svss = read_csv(csv_path)
    print(f"[INFO] H5 files = {len(h5s)}")

    # map_out
    map_out = out_dir / "h5id_map.txt"
    with open(map_out, "w") as f:
        f.write("h5_id\th5_path\tsvs_path\n")
        for i, (h, s) in enumerate(zip(h5s, svss)):
            f.write(f"{i}\t{h}\t{s}\n")

    n_keep, coords_cache = scan_h5(h5s, K)
    index_path, offsets = build_big_index(n_keep, out_dir)

    total = sum(n_keep)
    n_shards = (total + SHARD_CAP - 1) // SHARD_CAP
    print(f"[INFO] shards = {n_shards}")

    shard_tasks = [[] for _ in range(n_shards)]
    for i, off in enumerate(offsets):
        if n_keep[i] > 0:
            shard_tasks[off // SHARD_CAP].append(i)

    # shard 数通常很小，直接串行 shard
    for sid in range(n_shards):
        if not shard_tasks[sid]:
            continue
        shard_worker(
            sid,
            shard_tasks[sid],
            h5s,
            svss,
            coords_cache,
            offsets,
            out_dir,
            K,
            workers,
        )





# ============================================================
# H5 scan
# ============================================================
def scan_one_h5(args):
    h5_path, K = args
    try:
        with h5py.File(h5_path, "r") as f:
            if "coords" not in f:
                return 0, None
            coords = f["coords"][:]
    except Exception:
        return 0, None

    if coords.ndim != 2 or coords.shape[1] != 2:
        return 0, None

    k = min(len(coords), K)
    return k, coords


def scan_h5(h5s, K, workers=None):
    if workers is None:
        workers = min(cpu_count(), 32)

    n_keep = [0] * len(h5s)
    coords_cache = [None] * len(h5s)

    args = [(p, K) for p in h5s]

    with Pool(workers) as pool:
        for i, (k, coords) in enumerate(
            tqdm(
                pool.imap(scan_one_h5, args),
                total=len(h5s),
                desc="scan h5 (mp)",
            )
        ):
            n_keep[i] = k
            coords_cache[i] = coords

    return n_keep, coords_cache
# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default='/mnt/local10/data_new/tcga_rundata/patches_output/h5_wsi_index.csv')
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--out_dir", default='/mnt/local10/data_new/shard_tcga')
    parser.add_argument("--index_out", default='/mnt/local10/data_new/shard_tcga/big_index.npy')
    parser.add_argument("--map_out", default='/mnt/local10/data_new/shard_tcga/svs_to_npypath.txt')
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    build_dataset(
        csv_path=args.csv,
        out_dir=args.out_dir,
        K=args.k,
        workers=args.workers,
    )
