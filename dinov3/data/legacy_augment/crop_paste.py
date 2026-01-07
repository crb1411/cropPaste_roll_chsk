from __future__ import annotations

from typing import Tuple, Optional, Sequence, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import v2


class CropPaste:
    """
    Crop a patch from the input tensor, resize, and paste it back on a canvas.
    """

    def __init__(
        self,
        *,
        resize_scale_h: Tuple[float, float] = (0.5, 0.9),
        resize_scale_w: Tuple[float, float] = (0.5, 0.9),
        crop_scale: Tuple[float, float] = (0.7, 1.0),
        background: float | Sequence[float] = 1.0,
        tile: int = 16,
        grid_snap: Optional[int] = 16,
        antialias: bool = True,
        interpolate_mode: str = "bicubic",
        seed: int = 87,
    ):
        self.resize_scale_h = resize_scale_h
        self.resize_scale_w = resize_scale_w
        self.crop_scale = crop_scale
        self.background = background
        self.tile = tile
        self.grid_snap = grid_snap
        self.antialias = antialias
        self.interpolate_mode = interpolate_mode
        self.rng = torch.Generator().manual_seed(seed)

    @staticmethod
    def _snap(value: int, grid: int) -> int:
        return (value // grid) * grid

    @torch.no_grad()
    def __call__(
        self,
        x01: Tensor,
        *,
        crop_hw: Optional[Tuple[int, int]] = None,
        paste_hw: Optional[Tuple[int, int]] = None,
        pos_xy: Optional[Tuple[int, int]] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor | int | Tuple[int, int]]]:
        assert x01.ndim == 3 and x01.dtype == torch.float32, f"need [C,H,W] float32, got {x01.shape} {x01.dtype}"
        C, H, W = x01.shape
        dev = x01.device
        rng = rng or self.rng

        if crop_hw is None:
            crop_h = torch.empty((), device=dev).uniform_(*self.crop_scale, generator=rng).item()
            crop_w = torch.empty((), device=dev).uniform_(*self.crop_scale, generator=rng).item()
            ch, cw = int(round(H * crop_h)), int(round(W * crop_w))
        else:
            ch, cw = max(1, min(H, int(crop_hw[0]))), max(1, min(W, int(crop_hw[1])))

        max_top = max(0, H - ch)
        max_left = max(0, W - cw)
        top = int(torch.randint(0, max_top + 1, (), device=dev, generator=rng).item())
        left = int(torch.randint(0, max_left + 1, (), device=dev, generator=rng).item())

        crop = x01[:, top : top + ch, left : left + cw]

        if paste_hw is None:
            rh = torch.empty((), device=dev).uniform_(*self.resize_scale_h, generator=rng).item()
            rw = torch.empty((), device=dev).uniform_(*self.resize_scale_w, generator=rng).item()
            ph = max(1, min(H, int(round(H * rh))))
            pw = max(1, min(W, int(round(W * rw))))
        else:
            ph = max(1, min(H, int(paste_hw[0])))
            pw = max(1, min(W, int(paste_hw[1])))

        if self.grid_snap:
            grid = self.grid_snap
            ph = min(H, max(grid, (ph // grid) * grid))
            pw = min(W, max(grid, (pw // grid) * grid))

        crop_rs = F.interpolate(
            crop.unsqueeze(0),
            size=(ph, pw),
            mode=self.interpolate_mode,
            antialias=self.antialias,
            align_corners=False if "linear" in self.interpolate_mode else None,
        ).squeeze(0)

        if isinstance(self.background, (tuple, list)):
            if len(self.background) == 2 and all(isinstance(v, (int, float)) for v in self.background):
                lo, hi = float(self.background[0]), float(self.background[1])
                if hi < lo:
                    lo, hi = hi, lo
                bg_val = torch.empty((), device=dev).uniform_(lo, hi, generator=rng).item()
                canvas = torch.empty_like(x01).fill_(bg_val)
            else:
                bg = torch.tensor(self.background, dtype=x01.dtype, device=dev).view(C, 1, 1)
                canvas = bg.expand(C, H, W).clone()
        else:
            canvas = torch.empty_like(x01).fill_(float(self.background))

        if pos_xy is None:
            max_yt = max(0, H - ph)
            max_xt = max(0, W - pw)
            yt = int(torch.randint(0, max_yt + 1, (), device=dev, generator=rng).item())
            xt = int(torch.randint(0, max_xt + 1, (), device=dev, generator=rng).item())
            if self.grid_snap:
                grid = self.grid_snap
                yt = self._snap(yt, grid)
                xt = self._snap(xt, grid)
        else:
            yt, xt = int(pos_xy[0]), int(pos_xy[1])
            if self.grid_snap:
                grid = self.grid_snap
                yt = self._snap(yt, grid)
                xt = self._snap(xt, grid)
            yt = max(0, min(H - ph, yt))
            xt = max(0, min(W - pw, xt))

        canvas[:, yt : yt + ph, xt : xt + pw] = crop_rs

        covered_pix = torch.zeros((H, W), dtype=torch.bool, device=dev)
        covered_pix[yt : yt + ph, xt : xt + pw] = True

        t = self.tile
        R, Cc = H // t, W // t
        if R > 0 and Cc > 0:
            cov_f = covered_pix.float().unsqueeze(0).unsqueeze(0)
            tile_cov = F.avg_pool2d(cov_f, kernel_size=t, stride=t) * (t * t)
            uncovered_tiles_mask = tile_cov.squeeze().eq(0)
            lin = torch.arange(R * Cc, device=dev).view(R, Cc)
            uncovered_idx = lin[uncovered_tiles_mask].reshape(-1)
        else:
            uncovered_tiles_mask = torch.zeros((0, 0), dtype=torch.bool, device=dev)
            uncovered_idx = torch.zeros((0,), dtype=torch.long, device=dev)

        info = dict(
            crop_top=top,
            crop_left=left,
            crop_h=ch,
            crop_w=cw,
            paste_h=ph,
            paste_w=pw,
            yt=yt,
            xt=xt,
            uncovered_idx=uncovered_idx,
        )
        return canvas, info
