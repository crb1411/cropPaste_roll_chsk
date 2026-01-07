# utils.py
from __future__ import annotations
import os


import json
from datetime import datetime
import pathlib
import numpy as np
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision.transforms import v2, InterpolationMode

def json_default(o):
    # torch 张量 -> Python 基本类型
    if isinstance(o, torch.Tensor):
        # 先转到 CPU，再转 list（标量转成 Python 标量）
        if o.ndim == 0:
            return o.detach().cpu().item()
        return o.detach().cpu().tolist()
    # numpy 类型
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    # 其他常见类型
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    if isinstance(o, pathlib.Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    # 兜底
    return str(o)


def pil_to_01_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float32 [C,H,W], in [0,1]."""
    x = v2.functional.pil_to_tensor(img)                      # uint8
    x = v2.functional.to_dtype(x, torch.float32, scale=True)  # 0..1
    return x



def save_x01_image_safe(x01: torch.Tensor, save_path: str):
    """
    x01: float32 [C,H,W] in [0,1], C in {1,3,4}
    自动推断 PIL 模式并保存。
    """
    if x01.ndim != 3:
        raise ValueError(f"Expect [C,H,W], got {x01.shape}")
    C = x01.shape[0]
    if C not in (1,3,4):
        raise ValueError(f"Unsupported channels: {C}")

    x01 = x01.detach().cpu().clamp(0, 1)
    # 直接交给 to_pil_image（会从 float 自动缩放到 0..255 并推断模式）
    img = v2.functional.to_pil_image(x01)
    img.save(save_path)
    print(f"saved -> {save_path}")
    
def create_dir_time(
    *,
    base_dir: str | Path,
    prefix: str = '',
):
    """
    创建一个以时间命名的子目录，返回路径。
    """
    base_dir = Path(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    now = datetime.now()
    now_str = now.strftime('%Y%m%d_%H%M%S')
    dir_name = f'{prefix}_{now_str}'
    dir_path = base_dir / dir_name
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)



def _denormalize(x: torch.Tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD) -> torch.Tensor:
    if x.ndim != 3:
        return x
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    return x * std + mean
    
def save_index(
    img: dict,
    out_dir: str = "aug_vis",
    prefix: str = "idx",
    norm: str | tuple = "imagenet",   # "imagenet" | "wsi" | (mean, std)
    max_locals: int = 8,
    save_teacher: bool = True
):
    """
    img: 就是 dataset[1] 的返回 dict
    out_dir = create_dir_time(
                base_dir='/data/work/output_dir/aug_out',
                prefix='aug'                   
            )
    save_index(img, out_dir, prefix='augVis')
    """
    os.makedirs(out_dir, exist_ok=True)

    if norm == "imagenet":
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    elif isinstance(norm, (list, tuple)) and len(norm) == 2:
        mean, std = norm
    else:
        raise ValueError(f"unknown norm spec: {norm}")

    def _save_one(t: torch.Tensor, path: str):
        # t: CHW normalized tensor
        t = t.detach().cpu()
        t_denorm = _denormalize(t, mean, std)             # CHW, [0,1]
        save_x01_image_safe(t_denorm, path)                   # 接受 BCHW/CHW，范围[0,1]

    # 1) global crops (student)
    for i, t in enumerate(img.get("global_crops", [])):
        _save_one(t, os.path.join(out_dir, f"{prefix}_global_{i+1}.png"))

    # 2) local crops（只保存前 max_locals 张，避免太多）
    for i, t in enumerate(img.get("local_crops", [])[:max_locals]):
        _save_one(t, os.path.join(out_dir, f"{prefix}_local_{i+1}.png"))

    # 3) teacher crops（通常与 global_crops 相同，这里可选保存）
    if save_teacher:
        for i, t in enumerate(img.get("global_crops_teacher", [])):
            _save_one(t, os.path.join(out_dir, f"{prefix}_teacher_{i+1}.png"))

    if 'raw_image' in img:
        _save_one(img['raw_image'], os.path.join(out_dir, f"{prefix}_raw_img.png"))
        
    if 'legacy_aug' in img:
        aug_dic = img['legacy_aug']
        for k, v in aug_dic.items():
            if k.endswith('_info'):
                with open(f'{out_dir}/legacy_aug_png_{k}_info.json', 'w') as f:
                    json.dump(v, f, indent=4, default=json_default)
            else:
                save_x01_image_safe(v, f'{out_dir}/legacy_aug_png_{k}.png')
    if 'legacy_aug_resized' in img:
        aug_dic = img['legacy_aug_resized']
        for k, v in aug_dic.items():
            if k.endswith('_info'):
                with open(f'{out_dir}/legacy_aug_resized_png_{k}_info.json', 'w') as f:
                    json.dump(v, f, indent=4, default=json_default)
            else:
                save_x01_image_safe(v, f'{out_dir}/legacy_aug_resized_png_{k}.png')

    return out_dir
    
    