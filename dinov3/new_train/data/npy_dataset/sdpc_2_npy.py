#!/usr/bin/env python3
import os
import csv
import zlib
import time
import numpy as np
import h5py
import sys
sys.path.append('/mnt/crb/code/opensdpc')
import opensdpc_old as openslide

from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import accumulate
from typing import List, Tuple

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
# done helpers（唯一的跨机协调）
# ============================================================
def make_done_path(done_root, svs_path, *, segment=None):
    """
    done_root/path/to/wsi.svs.{start}-{end}.done
    """
    rel = Path(svs_path).as_posix().lstrip("/").replace(":", "_")
    suffix = ""
    if segment is not None:
        start, end = segment
        suffix = f".{start:09d}-{end:09d}"
    p = Path(done_root) / f"{rel}{suffix}.done"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _try_acquire_lock(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

# ============================================================
# CSV
# ============================================================


def read_csv(csv_path: str, *, dedup_by: str = "pair", max_wsi=None) -> Tuple[List[str], List[str]]:
    """
    dedup_by:
      - "pair": 按 (h5_path, wsi_path) 这一对去重（推荐）
      - "h5":   只按 h5_path 去重（同一 h5 多行只保留第一次）
    """
    h5s: List[str] = []
    svss: List[str] = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            h5_path = (r.get("h5_file_path") or "").strip()
            wsi_path = (r.get("wsi_path") or "").strip()  # 若列名是 svs_path 改这里

            if not h5_path or not wsi_path:
                continue

            # 可选：规范化路径，减少“同一路径不同写法”导致的重复
            h5_path = os.path.abspath(h5_path)
            wsi_path = os.path.abspath(wsi_path)

            # 可选：存在性检查
            if not os.path.isfile(h5_path) or not os.path.isfile(wsi_path):
                continue

            key = (h5_path, wsi_path) if dedup_by == "pair" else h5_path

            # 关键点：保序去重（只跳过重复，不做排序/重排）
            if key in seen:
                continue
            seen.add(key)
            if max_wsi is not None and len(svss) >= max_wsi:
                break

            h5s.append(h5_path)
            svss.append(wsi_path)

    return h5s, svss

# ============================================================
# scan H5（coords cache，多机安全）
# ============================================================
def _scan_one_h5(args):
    idx, h5_path, K, cache_dir = args
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    coords_path = cache_dir / f"{idx:05d}.coords.npy"
    if coords_path.exists():
        try:
            coords = np.load(coords_path, allow_pickle=False)
            return idx, min(len(coords), K), coords
        except Exception:
            pass

    try:
        with h5py.File(h5_path, "r") as f:
            if "coords" not in f:
                return idx, 0, None
            coords = f["coords"][:]
    except Exception:
        return idx, 0, None

    if coords is None or len(coords) == 0:
        return idx, 0, None

    tmp = coords_path.with_suffix(".tmp.npy")
    np.save(tmp, coords)
    os.replace(tmp, coords_path)

    return idx, min(len(coords), K), coords


def scan_h5(h5s, K, workers, cache_dir):
    n_keep = [0] * len(h5s)
    coords_cache = [None] * len(h5s)

    with Pool(workers) as pool:
        for idx, k, coords in tqdm(
            pool.imap_unordered(
                _scan_one_h5,
                ((i, p, K, cache_dir) for i, p in enumerate(h5s))
            ),
            total=len(h5s),
            desc="scan h5",
        ):
            n_keep[idx] = k
            coords_cache[idx] = coords

    return n_keep, coords_cache

# ============================================================
# big_index
# ============================================================
def build_big_index(n_keep, out_dir):
    total = sum(n_keep)
    offsets = list(accumulate([0] + n_keep))[:-1]

    index_path = out_dir / "big_index.npy"
    if index_path.exists():
        return index_path, offsets

    index = np.memmap(
        index_path, dtype=INDEX_DTYPE, mode="w+", shape=(total,)
    )

    cur = 0
    for h5_id, k in enumerate(n_keep):
        if k == 0:
            continue
        index[cur:cur+k]["h5_id"] = h5_id
        index[cur:cur+k]["patch_id"] = np.arange(k, dtype=np.int32)
        index[cur:cur+k]["global_idx"] = np.arange(cur, cur+k, dtype=np.int64)
        cur += k

    index.flush()
    return index_path, offsets

# ============================================================
# shard utils
# ============================================================
def open_shard(path, patch_count=SHARD_CAP):
    if patch_count <= 0:
        raise ValueError(f"patch_count must be > 0, got {patch_count}")
    shape = (patch_count, PATCH_H, PATCH_W, PATCH_C)
    expected_bytes = patch_count * PATCH_BYTES
    if not path.exists():
        mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
        del mm
    else:
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            os.truncate(path, expected_bytes)
    return np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)

# ============================================================
# 单个 WSI 处理（带锁 + 异常保护）
# ============================================================
def process_one_wsi(args):
    (
        h5_id,
        h5_path,
        svs_path,
        coords,
        K,
        base,
        done_root,
        patch_start,
        patch_end,
    ) = args

    done_path = make_done_path(done_root, svs_path, segment=(patch_start, patch_end))
    processing_path = done_path.with_suffix(".processing")

    # 已完成，直接跳过
    if done_path.exists():
        return ("skip", h5_id, base, None, None, done_path, h5_path, svs_path)

    # === 关键：防止并发打开同一个 WSI ===
    if not _try_acquire_lock(processing_path):
        return ("skip", h5_id, base, None, None, done_path, h5_path, svs_path)

    slide = None
    slide = None
    try:
        # ===== 真正干活 =====
        k = min(K, len(coords))
        if patch_start >= k:
            return ("empty", h5_id, base, np.empty((0, PATCH_H, PATCH_W, PATCH_C), dtype=np.uint8), None, done_path, h5_path, svs_path)
        patch_end = min(patch_end, k)

        seed = zlib.crc32(h5_path.encode()) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(coords), size=k, replace=False)
        sel = sel[patch_start:patch_end]

        patches = np.empty((len(sel), PATCH_H, PATCH_W, PATCH_C), dtype=np.uint8)
        slide = openslide.open_slide(svs_path)

        for j, ci in enumerate(sel):
            x, y = coords[ci]
            patch = slide.read_region((int(x), int(y)), 0, (PATCH_W, PATCH_H))
            patches[j] = np.asarray(patch)[..., :3]

        return ("ok", h5_id, base, patches, None, done_path, h5_path, svs_path)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return ("error", h5_id, base, None, err, done_path, h5_path, svs_path)

    finally:
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass
        # 无论成功 / 异常，都释放 processing
        _safe_unlink(processing_path)


# ============================================================
# shard worker（真正的多机线性加速点）
# ============================================================
def shard_worker_old(
    shard_id,
    h5_ids,
    h5s,
    svss,
    coords_cache,
    offsets,
    data_root,
    K,
    workers,
    done_root,
):
    shard_path = data_root / f"shard_{shard_id:05d}.npy"
    shard_mem = open_shard(shard_path, patch_count=SHARD_CAP)

    jobs = []
    for h5_id in h5_ids:
        coords = coords_cache[h5_id]
        if coords is None or len(coords) == 0:
            continue

        base = offsets[h5_id] % SHARD_CAP
        jobs.append(
            (
                h5_id,
                h5s[h5_id],
                svss[h5_id],
                coords,
                K,
                base,
                done_root,
                0,
                min(K, len(coords)),
            )
        )

    with Pool(workers) as pool:
        for ret in tqdm(
            pool.imap_unordered(process_one_wsi, jobs),
            total=len(jobs),
            desc=f"shard {shard_id}",
        ):
            if ret is None:
                continue
            status, h5_id, base, patches, err, done_path, h5_path, svs_path = ret
            if status not in ("ok", "empty"):
                if status == "error":
                    print(f"[WARN] shard {shard_id} h5_id={h5_id} error: {err} ({h5_path} | {svs_path})")
                continue
            end = base + len(patches)
            if end <= SHARD_CAP:
                shard_mem[base:end] = patches
                done_path.write_text("done\n")
            else:
                n1 = SHARD_CAP - base
                shard_mem[base:SHARD_CAP] = patches[:n1]
                print(f"[WARN] shard {shard_id} overflow {end}->{SHARD_CAP} ({h5_path} | {svs_path})")

    shard_mem.flush()

def shard_worker(shard_id, tasks, h5s, svss, coords_cache, out_dir, K, workers, done_root, shard_patch_count):
    shard_path = out_dir / f"shard_{shard_id:05d}.npy"
    done_flag = shard_path.with_suffix(".npy.done")
    processing_flag = shard_path.with_suffix(".npy.processing")
    error_log = shard_path.with_suffix(".npy.errors.log")

    if done_flag.exists():
        if not _try_acquire_lock(processing_flag):
            print(f"[INFO] shard {shard_id} done, skip")
            return
        try:
            mm = open_shard(shard_path, patch_count=shard_patch_count)
            del mm
        finally:
            _safe_unlink(processing_flag)
        print(f"[INFO] shard {shard_id} done, skip")
        return

    if not _try_acquire_lock(processing_flag):
        print(f"[INFO] shard {shard_id} is processing elsewhere, skip")
        return

    shard_mem = open_shard(shard_path, patch_count=shard_patch_count)
    had_error = False

    jobs = []
    for h5_id, base, patch_start, patch_end in tasks:
        coords = coords_cache[h5_id]
        if coords is None or len(coords) == 0:
            continue
        jobs.append((
            h5_id,
            h5s[h5_id],
            svss[h5_id],
            coords,
            K,
            base,
            done_root,
            patch_start,
            patch_end,
        ))

    try:
        if jobs:
            with Pool(workers) as pool:
                for ret in tqdm(
                    pool.imap_unordered(process_one_wsi, jobs),
                    total=len(jobs),
                    desc=f"shard {shard_id}",
                ):
                    if ret is None:
                        continue
                    status, h5_id, base, patches, err, done_path, h5_path, svs_path = ret
                    if status not in ("ok", "empty"):
                        if status == "error":
                            had_error = True
                            msg = f"h5_id={h5_id} error: {err} ({h5_path} | {svs_path})"
                            print(f"[WARN] shard {shard_id} {msg}")
                            with open(error_log, "a") as f:
                                f.write(msg + "\n")
                        continue
                    end = base + len(patches)
                    if end <= SHARD_CAP:
                        shard_mem[base:end] = patches
                        done_path.write_text("done\n")
                    else:
                        had_error = True
                        n1 = SHARD_CAP - base
                        shard_mem[base:SHARD_CAP] = patches[:n1]
                        msg = f"overflow {end}->{SHARD_CAP} ({h5_path} | {svs_path})"
                        print(f"[WARN] shard {shard_id} {msg}")
                        with open(error_log, "a") as f:
                            f.write(msg + "\n")
    finally:
        shard_mem.flush()
        del shard_mem
        if not had_error:
            done_flag.write_text("done\n")
        _safe_unlink(processing_flag)

def process_one_h5(args):
    h5_id, h5_path, svs_path, coords, K, base = args

    seed = zlib.crc32(h5_path.encode()) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(coords), size=min(K, len(coords)), replace=False)

    slide = openslide.open_slide(svs_path)
    patches = np.empty((len(sel), PATCH_H, PATCH_W, PATCH_C), dtype=np.uint8)

    for j, ci in enumerate(sel):
        x, y = coords[ci]
        patch = slide.read_region((int(x), int(y)), 0, (PATCH_W, PATCH_H))
        patches[j] = np.asarray(patch)[..., :3]

    slide.close()
    return base, patches
# ============================================================
# main
# ============================================================
def build_dataset(csv_path, out_dir, K, workers, shared_dir=None, max_wsi=None):
    out_dir = Path(out_dir)
    shared_dir = Path(shared_dir or out_dir)

    data_root = shared_dir
    data_root.mkdir(parents=True, exist_ok=True)
    done_root = data_root / "wsi_done"
    done_root.mkdir(parents=True, exist_ok=True)
    scan_cache_dir = data_root / "scan_cache"
    scan_cache_dir.mkdir(parents=True, exist_ok=True)

    h5s, svss = read_csv(csv_path, max_wsi=max_wsi)
    print(f"[INFO] H5 files = {len(h5s)}")

    n_keep, coords_cache = scan_h5(h5s, K, workers, scan_cache_dir)
    index_path, offsets = build_big_index(n_keep, out_dir)

    total = sum(n_keep)
    n_shards = (total + SHARD_CAP - 1) // SHARD_CAP
    print(f"[INFO] shards = {n_shards}")

    shard_tasks = [[] for _ in range(n_shards)]
    for h5_id, start in enumerate(offsets):
        k = n_keep[h5_id]
        if k <= 0:
            continue
        end = start + k
        seg_start = start
        while seg_start < end:
            shard_id = seg_start // SHARD_CAP
            shard_end = min(end, (shard_id + 1) * SHARD_CAP)
            patch_start = seg_start - start
            patch_end = shard_end - start
            base = seg_start % SHARD_CAP
            if shard_id >= len(shard_tasks):
                shard_tasks.extend([] for _ in range(shard_id + 1 - len(shard_tasks)))
            shard_tasks[shard_id].append((h5_id, base, patch_start, patch_end))
            seg_start = shard_end
    n_shards = len(shard_tasks)
    shard_patch_counts = [
        sum(patch_end - patch_start for _, _, patch_start, patch_end in tasks)
        for tasks in shard_tasks
    ]

    # shard 数通常很小，直接串行 shard
    for sid in range(n_shards):
        if not shard_tasks[sid]:
            continue
        shard_patch_count = shard_patch_counts[sid]
        if shard_patch_count <= 0:
            continue
        shard_worker(
            sid,
            shard_tasks[sid],
            h5s,
            svss,
            coords_cache,
            out_dir,
            K,
            workers,
            done_root,
            shard_patch_count,
        )

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default='/mnt/crb/work/runcode/copy_svs/datawheel_rundata/patches_output/h5_wsi_index.csv')
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--out_dir", default='/mnt/crb/work/runcode/copy_svs/datawheel_rundata/shared_datawheel_test')
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--shared", help="shared dir across machines (default = --out)")
    args = parser.parse_args()

    build_dataset(
        csv_path=args.csv,
        out_dir=args.out_dir,
        K=args.k,
        workers=args.workers,
        shared_dir=args.shared,
        max_wsi=200
    )
