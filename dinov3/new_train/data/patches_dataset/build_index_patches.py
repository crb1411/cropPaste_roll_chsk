#!/usr/bin/env python
import argparse
from pathlib import Path
import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

Image.MAX_IMAGE_PIXELS = None
PATCH_SIZE = 224
MAX_PAD_RATIO = 0.5
RANDOM_SHIFT = True
RANDOM_SEED = None


def load_image_paths(list_path: Path):
    list_path = list_path.expanduser().resolve()
    img_paths = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p and os.path.isfile(p):
                img_paths.append(p)
    return img_paths


# -------------------------
#  Worker：处理一张图片
# -------------------------
def process_one_image(args):
    img_idx, img_path, patch_size, max_pad_ratio, random_shift, base_seed = args

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            W, H = img.size
    except Exception:
        return []  # 跳过此图

    rng = np.random.default_rng(None if base_seed is None else base_seed + img_idx)
    min_keep_area = patch_size * patch_size * (1.0 - max_pad_ratio)
    min_keep_side = patch_size * (1.0 - max_pad_ratio)

    entries = []

    def build_positions(length):
        if length <= patch_size:
            return [0]
        n_full = length // patch_size
        remainder = length - n_full * patch_size
        positions = [i * patch_size for i in range(n_full)]
        if remainder >= min_keep_side:
            positions.append(n_full * patch_size)
            return positions
        if remainder > 0 and random_shift and n_full > 0:
            shift = int(rng.integers(0, remainder + 1))
            positions = [pos + shift for pos in positions]
        return positions

    x_positions = build_positions(W)
    y_positions = build_positions(H)
    strict_pad_limit = W >= patch_size and H >= patch_size
    for y in y_positions:
        h_tile = min(patch_size, H - y)
        for x in x_positions:
            w_tile = min(patch_size, W - x)
            if strict_pad_limit and (w_tile * h_tile) < min_keep_area:
                continue
            entries.append((img_idx, y, x, h_tile, w_tile))

    return entries


# -------------------------
# 多进程 build_index
# -------------------------
def build_index_mp(
    img_paths,
    patch_size=PATCH_SIZE,
    num_workers=None,
    max_pad_ratio=MAX_PAD_RATIO,
    random_shift=RANDOM_SHIFT,
    seed=RANDOM_SEED,
):

    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    print(f"[INFO] 使用 {num_workers} 个进程处理图像...")

    tasks = [
        (i, img_paths[i], patch_size, max_pad_ratio, random_shift, seed)
        for i in range(len(img_paths))
    ]

    all_entries = []

    with Pool(num_workers) as pool:
        for i, entries in enumerate(pool.imap_unordered(process_one_image, tasks, chunksize=32)):
            all_entries.extend(entries)
            if (i + 1) % 1000 == 0:
                print(f"[INFO] 已完成 {i+1}/{len(img_paths)} 张图像")

    print(f"[DONE] 总 patch 数: {len(all_entries)}")

    # 转 structured numpy
    dtype = np.dtype([
        ("img_idx", "int32"),
        ("y", "int32"),
        ("x", "int32"),
        ("h", "int16"),
        ("w", "int16"),
    ])

    index_arr = np.zeros(len(all_entries), dtype=dtype)
    for i, (img_idx, y, x, h, w) in enumerate(all_entries):
        index_arr[i] = (img_idx, y, x, h, w)

    return index_arr


if __name__ == "__main__":
    img_txt = "/mnt/local09/train/crb/data/img_data_v1225/img_list_v2.txt"
    output_npy = "/mnt/local09/train/crb/data/img_data_v1225/patch_index.npy"
    patch_size = 224

    img_paths = load_image_paths(Path(img_txt))
    print(f"[INFO] 从 {img_txt} 读取图像 {len(img_paths)} 张")

    index_arr = build_index_mp(img_paths, patch_size=patch_size, num_workers=64)

    out_path = Path(output_npy).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_path, index_arr)
    print(f"[DONE] 索引写入 {out_path}")

    # 保存 img list
    txt_path = out_path.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as f:
        for p in img_paths:
            f.write(p + "\n")
    print(f"[DONE] 路径写入 {txt_path}")
