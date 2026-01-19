#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

from dinov3.data.datasets import ImageNet

logger = logging.getLogger("dinov3.imagenet_extra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare ImageNet extras for log_regression")
    parser.add_argument("--root", required=True, help="ImageNet root (contains train/ and val/)")
    parser.add_argument("--extra", default=None, help="Extra cache root (default: root)")
    parser.add_argument(
        "--devkit-root",
        default=None,
        help="Path to ILSVRC2012_devkit_t12 (to build labels.txt if missing)",
    )
    parser.add_argument("--force-labels", action="store_true", help="Overwrite labels.txt if it exists")
    parser.add_argument("--skip-val", action="store_true", help="Skip val split extra")
    parser.add_argument("--skip-train", action="store_true", help="Skip train split extra")
    parser.add_argument(
        "--prepare-val",
        action="store_true",
        help="Prepare val class folders with symlinks if val is flat",
    )
    parser.add_argument(
        "--val-src",
        default=None,
        help="Val image directory (default: root/ILSVRC2012_img_val)",
    )
    return parser.parse_args()


def load_meta_mapping(devkit_root: Path) -> dict[str, str]:
    meta_path = devkit_root / "data" / "meta.mat"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.mat not found: {meta_path}")
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("scipy is required to parse meta.mat") from exc
    mat = loadmat(meta_path, struct_as_record=False, squeeze_me=True)
    synsets = mat.get("synsets")
    if synsets is None:
        raise ValueError(f"synsets not found in meta.mat: {meta_path}")
    mapping = {}
    for syn in synsets:
        num_children = getattr(syn, "num_children", None)
        if num_children is None or int(num_children) != 0:
            continue
        wnid = getattr(syn, "WNID", None)
        words = getattr(syn, "words", None)
        if wnid is None:
            continue
        if isinstance(wnid, bytes):
            wnid = wnid.decode("utf-8")
        if isinstance(words, bytes):
            words = words.decode("utf-8")
        mapping[str(wnid)] = str(words) if words is not None else str(wnid)
    if not mapping:
        raise ValueError(f"failed to parse mapping from meta.mat: {meta_path}")
    return mapping


def load_id_to_wnid(devkit_root: Path) -> dict[int, str]:
    meta_path = devkit_root / "data" / "meta.mat"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.mat not found: {meta_path}")
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("scipy is required to parse meta.mat") from exc
    mat = loadmat(meta_path, struct_as_record=False, squeeze_me=True)
    synsets = mat.get("synsets")
    if synsets is None:
        raise ValueError(f"synsets not found in meta.mat: {meta_path}")
    mapping = {}
    for syn in synsets:
        num_children = getattr(syn, "num_children", None)
        if num_children is None or int(num_children) != 0:
            continue
        wnid = getattr(syn, "WNID", None)
        ilsvrc_id = getattr(syn, "ILSVRC2012_ID", None)
        if wnid is None or ilsvrc_id is None:
            continue
        if isinstance(wnid, bytes):
            wnid = wnid.decode("utf-8")
        mapping[int(ilsvrc_id)] = str(wnid)
    if not mapping:
        raise ValueError(f"failed to parse id->wnid mapping from meta.mat: {meta_path}")
    return mapping


def _val_has_class_dirs(val_dir: Path) -> bool:
    if not val_dir.exists():
        return False
    for entry in val_dir.iterdir():
        if entry.is_dir():
            return True
    return False


def prepare_val_symlinks(root: Path, devkit_root: Path, val_src: Path) -> None:
    val_dir = root / "val"
    if not val_dir.exists():
        os.symlink(val_src, val_dir)
        logger.info("created val symlink: %s -> %s", val_dir, val_src)
    if _val_has_class_dirs(val_dir):
        logger.info("val already has class directories: %s", val_dir)
        return

    labels_path = devkit_root / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not labels_path.is_file():
        raise FileNotFoundError(f"validation labels not found: {labels_path}")
    id_to_wnid = load_id_to_wnid(devkit_root)
    val_dir.mkdir(parents=True, exist_ok=True)

    with labels_path.open("r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    if not labels:
        raise ValueError(f"empty validation labels: {labels_path}")

    for idx, label in enumerate(labels, start=1):
        wnid = id_to_wnid.get(label)
        if wnid is None:
            raise KeyError(f"missing wnid for label {label}")
        class_dir = val_dir / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        filename = f"ILSVRC2012_val_{idx:08d}.JPEG"
        src = val_src / filename
        dst = class_dir / filename
        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(f"val image missing: {src}")
        os.symlink(src, dst)


def ensure_labels_txt(root: Path, devkit_root: Path | None, force: bool) -> None:
    labels_path = root / "labels.txt"
    if labels_path.exists() and not force:
        logger.info("labels.txt exists: %s", labels_path)
        return
    if devkit_root is None:
        raise RuntimeError("labels.txt missing and --devkit-root not provided")
    mapping = load_meta_mapping(devkit_root)
    train_dir = root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train directory not found: {train_dir}")
    wnids = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not wnids:
        raise ValueError(f"no class folders found in: {train_dir}")
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for wnid in wnids:
            class_name = mapping.get(wnid, wnid)
            writer.writerow([wnid, class_name])
    logger.info("wrote labels.txt: %s", labels_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    extra = Path(args.extra).expanduser().resolve() if args.extra else root
    devkit_root = Path(args.devkit_root).expanduser().resolve() if args.devkit_root else None

    ensure_labels_txt(root, devkit_root, args.force_labels)

    if args.prepare_val:
        if devkit_root is None:
            raise RuntimeError("--prepare-val requires --devkit-root")
        val_src = Path(args.val_src).expanduser().resolve() if args.val_src else root / "ILSVRC2012_img_val"
        prepare_val_symlinks(root, devkit_root, val_src)

    if not args.skip_train:
        logger.info("dumping train extras to %s", extra)
        dataset = ImageNet(split=ImageNet.Split.TRAIN, root=str(root), extra=str(extra))
        dataset.dump_extra()
    if not args.skip_val:
        logger.info("dumping val extras to %s", extra)
        dataset = ImageNet(split=ImageNet.Split.VAL, root=str(root), extra=str(extra))
        dataset.dump_extra()


if __name__ == "__main__":
    main()
