from __future__ import annotations

import random
from typing import Dict, Union

import torch
from torch import Tensor
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image


def pil_to_01_tensor(img: Image.Image) -> Tensor:
    x = pil_to_tensor(img)
    return v2.functional.to_dtype(x, torch.float32, scale=True)


class ColorJitterAug:
    def __init__(self, j: float = 0.2, hue: float = 0.02, seed: int = 87):
        random.seed(seed)
        self.jitter = v2.Compose(
            [
                v2.RandomApply(
                    [v2.ColorJitter(
                        brightness=j * 2,
                        contrast=j * 2,
                        saturation=j,
                        hue=hue,
                    )],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

    def __call__(self, x01: Tensor) -> Tensor:
        # x01: float32, [C,H,W], in [0,1]
        return self.jitter(x01)



class GridShift16Roll:
    def __init__(self, patch_size: int = 16, max_shift: int = 5, compute_perm: bool = True):
        self.ps = patch_size
        self.max_shift = max_shift
        self.compute_perm = compute_perm

    @torch.no_grad()
    def __call__(self, x01: Tensor) -> tuple[Tensor, Dict[str, Union[int, Tensor]]]:
        assert x01.ndim == 3 and x01.shape[0] == 3
        _, H, W = x01.shape
        ps = self.ps
        assert H % ps == 0 and W % ps == 0, f"H,W must be multiples of {ps}, got {H},{W}"

        dr = int(torch.randint(-self.max_shift, self.max_shift + 1, ()).item())
        dc = int(torch.randint(-self.max_shift, self.max_shift + 1, ()).item())
        dy, dx = dr * ps, dc * ps
        y = torch.roll(x01, shifts=(dy, dx), dims=(1, 2))
        info: Dict[str, Union[int, Tensor]] = {"dr": dr, "dc": dc, "dy": dy, "dx": dx}

        if self.compute_perm:
            R, C = H // ps, W // ps
            r = torch.arange(R).unsqueeze(1).expand(R, C)
            c = torch.arange(C).unsqueeze(0).expand(R, C)
            r2 = (r + dr) % R
            c2 = (c + dc) % C
            perm_idx = (r2 * C + c2).reshape(-1).long()
            info["perm_idx"] = perm_idx

        return y, info


class WhiteImage:
    def __init__(self, noise_std: float = 0.0):
        self.std = noise_std

    def __call__(self, x01: Tensor) -> Tensor:
        y = torch.ones_like(x01)
        if self.std > 0:
            y = (y + torch.randn_like(y) * self.std).clamp_(0, 1)
        return y


class BlackImage:
    def __init__(self, noise_std: float = 0.0):
        self.std = noise_std

    def __call__(self, x01: Tensor) -> Tensor:
        y = torch.zeros_like(x01)
        if self.std > 0:
            y = (y + torch.randn_like(y) * self.std).clamp_(0, 1)
        return y


class GrayImage:
    def __init__(self, noise_std: float = 0.0):
        self.std = noise_std

    def __call__(self, x01: Tensor) -> Tensor:
        y = torch.full_like(x01, fill_value=0.5)
        if self.std > 0:
            y = (y + torch.randn_like(y) * self.std).clamp_(0, 1)
        return y
