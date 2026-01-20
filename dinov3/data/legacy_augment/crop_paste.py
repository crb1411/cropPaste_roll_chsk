from __future__ import annotations

import math
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
        crop_prob: float = 0.7,
        roll_prob: float = 0.0,
        split_prob: float = 0.0,
        split_count_range: Tuple[int, int] = (1, 1),
        split_shuffle_prob: float = 0.5,
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
        self.crop_prob = float(crop_prob)
        self.roll_prob = float(roll_prob)
        self.split_prob = float(split_prob)
        self.split_count_range = split_count_range
        self.split_shuffle_prob = float(split_shuffle_prob)
        self.background = background
        self.tile = tile
        self.grid_snap = grid_snap
        self.antialias = antialias
        self.interpolate_mode = interpolate_mode
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)

    @staticmethod
    def _snap(value: int, grid: int) -> int:
        return (value // grid) * grid

    @staticmethod
    def _random_partition(total: int, parts: int, *, rng: torch.Generator, device: torch.device) -> list[int]:
        if parts <= 1:
            return [total]
        parts = max(1, min(parts, total))
        if total == parts:
            return [1] * parts
        cuts = torch.randperm(total - 1, generator=rng, device=device)[: parts - 1] + 1
        cuts, _ = torch.sort(cuts)
        edges = torch.cat([torch.tensor([0], device=device), cuts, torch.tensor([total], device=device)])
        sizes = (edges[1:] - edges[:-1]).tolist()
        return [int(s) for s in sizes]

    def _sample_split_grid(
        self,
        *,
        ph: int,
        pw: int,
        rng: torch.Generator,
        device: torch.device,
    ) -> Tuple[int, int]:
        min_blocks, max_blocks = self.split_count_range
        min_blocks = max(1, int(min_blocks))
        max_blocks = max(min_blocks, int(max_blocks))
        max_rows = max(1, min(ph, max_blocks))
        max_cols = max(1, min(pw, max_blocks))
        for _ in range(20):
            rows = int(torch.randint(1, max_rows + 1, (), device=device, generator=rng).item())
            cols = int(torch.randint(1, max_cols + 1, (), device=device, generator=rng).item())
            count = rows * cols
            if min_blocks <= count <= max_blocks:
                return rows, cols
        rows = 1
        cols = min(max_blocks, pw)
        if rows * cols < min_blocks:
            rows = min(min_blocks, ph)
            cols = min(max(1, math.ceil(min_blocks / rows)), pw)
        return rows, cols

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

        do_crop = True
        if crop_hw is None:
            crop_prob = max(0.0, min(1.0, self.crop_prob))
            if crop_prob < 1.0:
                rv = torch.rand((), device=dev, generator=rng).item()
                do_crop = rv < crop_prob
        if crop_hw is None and do_crop:
            crop_h = torch.empty((), device=dev).uniform_(*self.crop_scale, generator=rng).item()
            crop_w = torch.empty((), device=dev).uniform_(*self.crop_scale, generator=rng).item()
            ch, cw = int(round(H * crop_h)), int(round(W * crop_w))
        elif crop_hw is None and not do_crop:
            ch, cw = H, W
        else:
            ch, cw = max(1, min(H, int(crop_hw[0]))), max(1, min(W, int(crop_hw[1])))

        max_top = max(0, H - ch)
        max_left = max(0, W - cw)
        if ch == H and cw == W:
            top = 0
            left = 0
        else:
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

        roll_shift = None
        roll_prob = max(0.0, min(1.0, self.roll_prob))
        if roll_prob > 0.0:
            rv = torch.rand((), device=dev, generator=rng).item()
            if rv < roll_prob:
                shift_y = int(torch.randint(0, max(1, ph), (), device=dev, generator=rng).item())
                shift_x = int(torch.randint(0, max(1, pw), (), device=dev, generator=rng).item())
                crop_rs = torch.roll(crop_rs, shifts=(shift_y, shift_x), dims=(1, 2))
                roll_shift = (shift_y, shift_x)

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

        do_split = False
        split_prob = max(0.0, min(1.0, self.split_prob))
        min_blocks, max_blocks = self.split_count_range
        if split_prob > 0.0 and max_blocks > 1 and ph > 1 and pw > 1:
            rv = torch.rand((), device=dev, generator=rng).item()
            do_split = rv < split_prob

        split_rows = 1
        split_cols = 1
        blocks = [(crop_rs, ph, pw)]
        if do_split:
            split_rows, split_cols = self._sample_split_grid(ph=ph, pw=pw, rng=rng, device=dev)
            if split_rows * split_cols > 1:
                row_sizes = self._random_partition(ph, split_rows, rng=rng, device=dev)
                col_sizes = self._random_partition(pw, split_cols, rng=rng, device=dev)
                blocks = []
                y0 = 0
                for rh in row_sizes:
                    x0 = 0
                    for cw in col_sizes:
                        blocks.append((crop_rs[:, y0 : y0 + rh, x0 : x0 + cw], rh, cw))
                        x0 += cw
                    y0 += rh
                if len(blocks) > 1:
                    p = max(0.0, min(1.0, self.split_shuffle_prob))
                    if p >= 1.0 or torch.rand((), device=dev, generator=rng).item() < p:
                        perm = torch.randperm(len(blocks), generator=rng, device=dev).tolist()
                        blocks = [blocks[i] for i in perm]

        block_boxes = []
        if pos_xy is None:
            for block, bh, bw in blocks:
                max_yt = max(0, H - bh)
                max_xt = max(0, W - bw)
                yt = int(torch.randint(0, max_yt + 1, (), device=dev, generator=rng).item())
                xt = int(torch.randint(0, max_xt + 1, (), device=dev, generator=rng).item())
                if self.grid_snap:
                    grid = self.grid_snap
                    yt = self._snap(yt, grid)
                    xt = self._snap(xt, grid)
                yt = max(0, min(H - bh, yt))
                xt = max(0, min(W - bw, xt))
                canvas[:, yt : yt + bh, xt : xt + bw] = block
                block_boxes.append((yt, xt, bh, bw))
        else:
            yt, xt = int(pos_xy[0]), int(pos_xy[1])
            if self.grid_snap:
                grid = self.grid_snap
                yt = self._snap(yt, grid)
                xt = self._snap(xt, grid)
            yt = max(0, min(H - ph, yt))
            xt = max(0, min(W - pw, xt))
            canvas[:, yt : yt + ph, xt : xt + pw] = crop_rs
            block_boxes.append((yt, xt, ph, pw))

        covered_pix = torch.zeros((H, W), dtype=torch.bool, device=dev)
        for yt, xt, bh, bw in block_boxes:
            covered_pix[yt : yt + bh, xt : xt + bw] = True

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
            yt=block_boxes[0][0] if block_boxes else 0,
            xt=block_boxes[0][1] if block_boxes else 0,
            roll_shift=roll_shift,
            split_rows=split_rows,
            split_cols=split_cols,
            block_boxes=block_boxes,
            uncovered_idx=uncovered_idx,
        )
        return canvas, info

    def __getstate__(self):
        # Rebuild RNG in worker; avoid pickling torch.Generator.
        return dict(
            resize_scale_h=self.resize_scale_h,
            resize_scale_w=self.resize_scale_w,
            crop_scale=self.crop_scale,
            crop_prob=self.crop_prob,
            roll_prob=self.roll_prob,
            split_prob=self.split_prob,
            split_count_range=self.split_count_range,
            split_shuffle_prob=self.split_shuffle_prob,
            background=self.background,
            tile=self.tile,
            grid_snap=self.grid_snap,
            antialias=self.antialias,
            interpolate_mode=self.interpolate_mode,
            seed=self.seed,
        )

    def __setstate__(self, state):
        self.__init__(**state)
