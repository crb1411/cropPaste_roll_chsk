#!/usr/bin/env python3
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

Image.MAX_IMAGE_PIXELS = None

try:
    _RESAMPLE = Image.Resampling.BICUBIC
except AttributeError:
    _RESAMPLE = Image.BICUBIC


class ImageNetResizeDataset(Dataset):
    def __init__(
        self,
        index_npy: str,
        image_list_txt: Optional[str] = None,
        size: int = 224,
        transform: Optional[Callable] = None,
        return_path: bool = False,
    ) -> None:
        self.index_npy = index_npy
        self.image_list_txt = image_list_txt
        self.size = size
        self.return_path = return_path
        self.transform = transform if transform is not None else T.ToTensor()
        self._paths = None

    def _load_paths(self) -> list[str]:
        if self.index_npy and Path(self.index_npy).is_file():
            arr = np.load(self.index_npy, allow_pickle=False)
            return [str(p) for p in arr.tolist()]
        if self.image_list_txt and Path(self.image_list_txt).is_file():
            paths = []
            with open(self.image_list_txt, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip()
                    if p:
                        paths.append(p)
            return paths
        raise FileNotFoundError("index_npy or image_list_txt must exist")

    def _ensure_paths(self) -> None:
        if self._paths is None:
            self._paths = self._load_paths()

    def __len__(self) -> int:
        self._ensure_paths()
        return len(self._paths)

    def __getitem__(self, idx: int):
        self._ensure_paths()
        img_path = self._paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((self.size, self.size), resample=_RESAMPLE)

        img = self.transform(img) if self.transform is not None else img
        if self.return_path:
            return img, img_path
        return img, None


if __name__ == "__main__":
    ds = ImageNetResizeDataset(
        index_npy="/data/data/imagenet_1k/data_new/imagenet_1k.npy",
        size=224,
    )
    print("dataset size:", len(ds))
    sample, _ = ds[0]
    if torch.is_tensor(sample):
        print("sample shape:", tuple(sample.shape))
