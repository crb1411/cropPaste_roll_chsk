#!/usr/bin/env python3
import os
from pathlib import Path
import tarfile



def _is_within_directory(base_dir: str, target_path: str) -> bool:
    base_dir = os.path.abspath(base_dir)
    target_path = os.path.abspath(target_path)
    return os.path.commonpath([base_dir]) == os.path.commonpath([base_dir, target_path])


def _safe_extract(tar: tarfile.TarFile, dest_dir: Path) -> None:
    dest_str = str(dest_dir)
    for member in tar.getmembers():
        member_path = os.path.join(dest_str, member.name)
        if not _is_within_directory(dest_str, member_path):
            raise RuntimeError(f"Unsafe path in tar: {member.name}")
    tar.extractall(dest_str)


def _collect_tars(tar_dir: Path, suffix: str) -> list[Path]:
    suffix = suffix.lower()
    return sorted(
        p for p in tar_dir.iterdir()
        if p.is_file() and p.name.lower().endswith(suffix)
    )


def extract_one(tar_path: Path, force: bool) -> tuple[bool, Path]:
    out_dir = tar_path.with_suffix("")
    if out_dir.exists() and not force:
        return False, out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        _safe_extract(tar, out_dir)
    return True, out_dir


def extract_imagenet_tars(tar_dir: Path, suffix: str = ".tar", force: bool = False) -> None:
    tar_dir = Path(tar_dir).expanduser().resolve()
    if not tar_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {tar_dir}")

    tar_paths = _collect_tars(tar_dir, suffix)
    if not tar_paths:
        raise RuntimeError(f"No tar files found in {tar_dir} with suffix {suffix}")

    for idx, tar_path in enumerate(tar_paths, start=1):
        ok, out_dir = extract_one(tar_path, force)
        status = "DONE" if ok else "SKIP"
        print(f"[{status}] {idx}/{len(tar_paths)} {tar_path.name} -> {out_dir.name}")

TAR_DIR = '/mnt/data/imagenet_1k/ILSVRC2012_img_train'
SUFFIX = ".tar"
FORCE = False

if __name__ == "__main__":
    if TAR_DIR is None:
        raise ValueError("Set TAR_DIR to the folder containing tar files.")
    extract_imagenet_tars(TAR_DIR, suffix=SUFFIX, force=FORCE)