#!/usr/bin/env python
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

Image.MAX_IMAGE_PIXELS = None

class PaddedPatchDataset(Dataset):
    """
    输入：
      - patches_index.npy: structured array，字段 (img_idx, y, x, h, w)
      - image_list.txt: 每行一个图像绝对路径
    输出：
      - patch: [3, H, W] (默认 224x224)，已经 pad 好
      - meta: 包含 img_idx, y, x, h, w 等信息（如果你不需要，可以忽略）
    """

    def __init__(
        self,
        index_npy: str,
        image_list_txt: str,
        patch_size: int = 224,
        pad_value: int = 0,
        transform: Optional[Callable] = None,
    ):
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.index_npy = index_npy
        self.image_list_txt = image_list_txt
        self.index = None
        self.img_paths = None

        # 默认 transform：PIL -> Tensor [0,1]
        if transform is None:
            self.transform = T.ToTensor()
        else:
            self.transform = transform
    def _ensure_data(self):
        if self.index is None:
            self.index = np.load(self.index_npy, allow_pickle=False)
            self.img_paths = self._load_image_paths(self.image_list_txt)

    @staticmethod
    def _load_image_paths(list_path: str):
        paths = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    paths.append(p)
        return paths

    def __len__(self) -> int:
        self._ensure_data()
        return len(self.index)

    def _load_and_crop_patch(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        rec = self.index[idx]
        img_idx = int(rec["img_idx"])
        y = int(rec["y"])
        x = int(rec["x"])
        h = int(rec["h"])
        w = int(rec["w"])

        img_path = Path(self.img_paths[img_idx])
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            patch = img.crop((x, y, x + w, y + h))  # (left, upper, right, lower)

        meta = {
            "img_idx": img_idx,
            "img_path": str(img_path),
            "y": y,
            "x": x,
            "h": h,
            "w": w,
        }
        return patch, meta

    def _pad_to_patch_size(self, patch: Image.Image) -> Image.Image:
        """将 patch pad 到 (patch_size, patch_size)，左上对齐。"""
        W, H = patch.size  # (width, height)
        if W == self.patch_size and H == self.patch_size:
            return patch

        # pad 背景颜色: pad_value（0=黑，255=白）
        bg_color = (self.pad_value,) * 3
        canvas = Image.new("RGB", (self.patch_size, self.patch_size), color=bg_color)
        # 左上角贴上去 (0,0)，如果想居中自己改坐标
        canvas.paste(patch, (0, 0))
        return canvas

    def __getitem__(self, idx: int):
        self._ensure_data()
        patch_pil, meta = self._load_and_crop_patch(idx)
        patch_pil = self._pad_to_patch_size(patch_pil)
        patch = self.transform(patch_pil)  # 默认 [C,H,W] float32 0~1

        # 你如果只想要图像，可以 return patch
        # 这里多返回一个 meta，方便你调试/可视化
        return patch, None


# 简单测试用
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = PaddedPatchDataset(
        index_npy="/mnt/local09/train/crb/data/img_data_v1130/patch_index.npy",
        image_list_txt="/mnt/local09/train/crb/data/img_data_v1130/patch_index.txt",
        patch_size=224,
        pad_value=0,
    )
    print("dataset size:", len(ds))
    dl = DataLoader(ds, batch_size=24, shuffle=True, num_workers=4)
    i = 0
    for batch_idx, imgs in enumerate(dl):
        print("batch", batch_idx, imgs.shape)
        i+=1
        if i>10:
            break
