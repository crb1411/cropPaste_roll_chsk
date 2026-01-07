"""
Small FSDP-enabled smoke script that runs one forward/backward step
on the png_dataset defined in resize_2/data/dataset.py using
DataAugmentationDINO + SSLAugmentedMetaArchV2.

Launch (single GPU):
  python -m dinov3.train.train_png_fsdp

Multi-GPU (example):
  torchrun --nproc_per_node=4 -m dinov3.train.train_png_fsdp
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from dinov3.configs import setup_config, setup_job
from dinov3.logging import setup_logging
from dinov3.data import MaskingGenerator, collate_data_and_cast
from dinov3.data.augmentations import DataAugmentationDINO
from dinov3.train import SSLMetaArch, SSLAugmentedMetaArch, SSLAugmentedMetaArchV2
from dinov3.train.train import get_args_parser


def _to_device_any(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device_any(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device_any(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device_any(v, device) for v in x)
    return x


def init_dist():
    if dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    else:
        # single process
        dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:23456", rank=0, world_size=1)


def build_dataloader(cfg):
    aug = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
        gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
        local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
        share_color_jitter=cfg.crops.share_color_jitter,
        horizontal_flips=cfg.crops.horizontal_flips,
        mean=cfg.crops.rgb_mean,
        std=cfg.crops.rgb_std,
        use_legacy_augmentor=True,
        legacy_augmentor_switch=None,
    )

    n_tokens = (cfg.crops.global_crops_size // cfg.student.patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(cfg.crops.global_crops_size // cfg.student.patch_size,) * 2,
        max_num_patches=0.5 * n_tokens,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=None,
    )

    # resize_2/data/dataset.py must be importable (ensure PYTHONPATH includes resize_2/)
    sys.path.append(str(REPO_ROOT.parent / "resize_2"))
    from data.dataset import png_dataset  # noqa: WPS433

    dataset = png_dataset(img_transforms=aug)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return loader


def main():
    parser = get_args_parser()
    parser.add_argument("--config", type=str, default='/data/work/git_proj/dinov3/dinov3/configs/train_new/ssl_default_config.yaml', help="Optional path to config yaml (falls back to --config-file)")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Optional checkpoint dir (align with train_svs)")
    args = parser.parse_args()
    try:
        from dinov3.new_train.utils.log_create import creat_subdir
        args.output_dir = creat_subdir(base_dir='/data/work/output_dir/cr_ch_rundata', create=True, time=True)
    except:
        raise ImportError
    # setup config & logging (similar to train.main but simplified)
    if os.path.isfile(args.config):
        args.config_file = args.config

    setup_job(output_dir=args.output_dir, seed=args.seed)
    cfg = setup_config(args, strict_cfg=False)
    setup_logging(
        output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
        name="nan_logger_pngfsdp",
    )

    # force using augmented meta-arch for this smoke
    if not cfg.MODEL.META_ARCHITECTURE:
        cfg.MODEL.META_ARCHITECTURE = "SSLAugmentedMetaArchV2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "SSLAugmentedMetaArch": SSLAugmentedMetaArch,
        "SSLAugmentedMetaArchV2": SSLAugmentedMetaArchV2,
    }.get(cfg.MODEL.META_ARCHITECTURE, SSLAugmentedMetaArchV2)
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        model = meta_arch(cfg)
    # use the meta-arch's built-in distributed prep (DDP/FSDP as configured)
    model.prepare_for_distributed_training()

    loader = build_dataloader(cfg)
    batch = next(iter(loader))
    batch = _to_device_any(batch, device)

    model.train()
    teacher_temp = 0.059
    batch['global_batch_size'] = 8
    loss, logs = model.forward_backward(batch, teacher_temp=teacher_temp, iteration=0)

    if dist.get_rank() == 0:
        print("loss:", loss.item())
        for k, v in logs.items():
            if torch.is_tensor(v):
                v = v.item()
            print(f"{k}: {v}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
