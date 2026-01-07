from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch import Tensor
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .augment_pack import (
    ColorJitterAug,
    GridShift16Roll,
    GrayImage,
    BlackImage,
    WhiteImage,
)
from .crop_paste import CropPaste


@dataclass
class AugmentSwitch:
    use_crop: bool = True
    use_color: bool = False
    use_shift: bool = True
    use_white: bool = False
    use_black: bool = False
    use_gray: bool = False
    crop_background_mode: str = "random"  # "fixed" or "random"
    crop_background_value: float = 1.0
    crop_background_min: float = 0.0
    crop_background_max: float = 1.0


class Augmentor:
    def __init__(
        self,
        switches: AugmentSwitch = AugmentSwitch(),
        noise_std: float = 0.0,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        use_normalize: bool = True,
    ):
        self.sw = switches
        self.mean = mean
        self.std = std
        self.use_normalize = use_normalize
        self.tensor_normalize = v2.Normalize(mean=mean, std=std)

        self.aug_dic: OrderedDict[str, Any] = OrderedDict()

        if self.sw.use_crop:
            if self.sw.crop_background_mode == "random":
                background = (self.sw.crop_background_min, self.sw.crop_background_max)
            else:
                background = float(self.sw.crop_background_value)
            self.aug_dic["cropPaste"] = CropPaste(
                resize_scale_h=(0.5, 0.9),
                resize_scale_w=(0.5, 0.9),
                crop_scale=(0.7, 1.0),
                background=background,
                tile=16,
                grid_snap=16,
                antialias=True,
                interpolate_mode="bicubic",
                seed=87,
            )

        if self.sw.use_color:
            self.aug_dic["color"] = ColorJitterAug(j=0.3, hue=0.02)

        if self.sw.use_shift:
            self.aug_dic["shift"] = GridShift16Roll(patch_size=16, max_shift=5, compute_perm=True)

        if self.sw.use_white:
            self.aug_dic["white"] = WhiteImage(noise_std=noise_std)
        if self.sw.use_black:
            self.aug_dic["black"] = BlackImage(noise_std=noise_std)
        if self.sw.use_gray:
            self.aug_dic["gray"] = GrayImage(noise_std=noise_std)

    def _maybe_norm(self, x: Tensor) -> Tensor:
        if not self.use_normalize:
            return x
        return self.tensor_normalize(x)

    def __call__(self, x01: Tensor) -> Dict[str, Any]:
        assert isinstance(x01, torch.Tensor), "input must be Tensor"
        assert x01.ndim == 3 and x01.shape[0] == 3, f"expect [3,H,W], got {x01.shape}"
        assert x01.dtype == torch.float32, f"expect float32, got {x01.dtype}"

        out: Dict[str, Any] = {}
        out["raw"] = self._maybe_norm(x01)

        for name, aug in self.aug_dic.items():
            res = aug(x01)
            if isinstance(res, tuple) and len(res) == 2:
                img, info = res
            else:
                img, info = res, None

            img = self._maybe_norm(img)
            out[name] = img
            if info is not None:
                out[f"{name}_info"] = info

        return out
