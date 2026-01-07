#!/usr/bin/env python3
import os
import multiprocessing as mp
from pathlib import Path

import numpy as np


DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _collect_in_dir(args) -> list[str]:
    base_dir, exts, follow_links = args
    paths = []
    for dirpath, _, filenames in os.walk(base_dir, followlinks=follow_links):
        for name in filenames:
            if name.lower().endswith(exts):
                full_path = os.path.abspath(os.path.join(dirpath, name))
                paths.append(full_path)
    return paths


def collect_image_paths(
    root: Path,
    exts: tuple,
    follow_links: bool,
    num_workers: int | None = None,
    chunk_size: int = 8,
) -> list[str]:
    paths = []
    root = Path(root)
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    root_files = [
        os.path.abspath(os.path.join(root, name))
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name)) and name.lower().endswith(exts)
    ]

    if not subdirs:
        paths = _collect_in_dir((root, exts, follow_links))
        paths.extend(root_files)
        paths.sort()
        return paths

    if num_workers is None:
        num_workers = max(mp.cpu_count() - 1, 1)

    tasks = [(p, exts, follow_links) for p in subdirs]
    if num_workers <= 1:
        for task in tasks:
            paths.extend(_collect_in_dir(task))
    else:
        with mp.Pool(processes=num_workers) as pool:
            for sub_paths in pool.imap_unordered(_collect_in_dir, tasks, chunksize=chunk_size):
                paths.extend(sub_paths)
    paths.extend(root_files)
    paths.sort()
    return paths


def _normalize_exts(extensions) -> tuple:
    if isinstance(extensions, str):
        parts = [ext.strip().lower() for ext in extensions.split(",") if ext.strip()]
        return tuple(parts)
    return tuple(ext.lower() for ext in extensions)


def build_imagenet_index(
    root: Path,
    output_npy: Path | None = None,
    output_txt: Path | None = None,
    extensions=DEFAULT_EXTS,
    follow_links: bool = False,
    num_workers: int | None = None,
    chunk_size: int = 8,
) -> tuple[Path, Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    exts = _normalize_exts(extensions)
    if not exts:
        raise ValueError("No valid extensions provided")

    paths = collect_image_paths(
        root,
        exts,
        follow_links,
        num_workers=num_workers,
        chunk_size=chunk_size,
    )
    if not paths:
        raise RuntimeError(f"No images found under {root}")

    output_npy = Path(output_npy) if output_npy else root / "imagenet_paths.npy"
    output_txt = Path(output_txt) if output_txt else root / "imagenet_paths.txt"
    output_npy = output_npy.expanduser().resolve()
    output_txt = output_txt.expanduser().resolve()
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_npy, np.array(paths, dtype=np.str_))
    with output_txt.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")

    print(f"[DONE] {len(paths)} images")
    print(f"[DONE] Saved npy: {output_npy}")
    print(f"[DONE] Saved txt: {output_txt}")
    return output_npy, output_txt

ROOT_DIR = '/mnt/data/imagenet_1k'
OUTPUT_NPY = '/mnt/data/imagenet_1k/data_new/imagenet_1k.npy'
OUTPUT_TXT = '/mnt/data/imagenet_1k/data_new/imagenet_1k.txt'
EXTENSIONS = DEFAULT_EXTS
FOLLOW_LINKS = False
NUM_WORKERS = 12
CHUNK_SIZE = 8
if __name__ == "__main__":
    if ROOT_DIR is None:
        raise ValueError("Set ROOT_DIR to the folder containing extracted images.")
    build_imagenet_index(
        root=ROOT_DIR,
        output_npy=OUTPUT_NPY,
        output_txt=OUTPUT_TXT,
        extensions=EXTENSIONS,
        follow_links=FOLLOW_LINKS,
        num_workers=NUM_WORKERS,
        chunk_size=CHUNK_SIZE,
    )
