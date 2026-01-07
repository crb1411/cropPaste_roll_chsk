#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
Image.MAX_IMAGE_PIXELS = None

def is_image_openable(path: Path) -> str | None:
    """尝试用 PIL 打开，确定是不是有效图像文件。
    返回 str(path) 或 None（失败）。
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return str(path.resolve())
    except Exception:
        return None


def iter_dirs_from_args(inputs):
    for p in inputs:
        p = Path(p).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".txt":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    d = Path(line.strip()).expanduser().resolve()
                    if d.is_dir():
                        yield d
        elif p.is_dir():
            yield p
        else:
            print(f"[WARN] 路径无效或不是目录/txt: {p}")


def scan_images(dirs, output_txt: Path, num_workers: int = None):
    output_txt = output_txt.expanduser().resolve()
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    # 默认 worker 数
    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    # 收集所有候选文件
    all_files = []
    for root_dir in dirs:
        root_dir = root_dir.expanduser().resolve()
        print(f"[INFO] 扫描目录: {root_dir}")
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                fp = Path(dirpath) / name
                if fp.is_file():
                    all_files.append(fp)

    print(f"[INFO] 共发现 {len(all_files)} 个文件，开始多进程验证...")

    results = []
    with Pool(num_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(is_image_openable, all_files, chunksize=64)):
            if r is not None:
                results.append(r)
            if (i + 1) % 2000 == 0:
                print(f"[INFO] 已处理 {i+1}/{len(all_files)} 文件")

    print(f"[DONE] 共找到 {len(results)} 张可用图像")

    with output_txt.open("w", encoding="utf-8") as f:
        for p in results:
            f.write(p + "\n")

    print(f"[DONE] 输出写入 {output_txt}")


if __name__ == "__main__":
    dirs = ["/mnt/local08/opensource_datasets/panda"]
    output_txt = Path("/mnt/local09/train/crb/data/img_data_v1225/img_list_v2.txt")
    dirs = [Path(dir_) for dir_ in dirs]
    scan_images(dirs, output_txt, num_workers=32)
