# dinov3/models/ssl_meta_arch_augmented.py

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from dinov3.data.masking import MaskingGenerator
from dinov3.loss import DINOLoss, DINOLoss_skcache, iBOTPatchLoss
from dinov3.new_train.train.ssl_meta_arch import SSLMetaArch
logger = logging.getLogger("dinov3")


# ----------------------------
# small helpers
# ----------------------------

def _soft_ce(student_logits: Tensor, target_prob: Tensor) -> Tensor:
    # logits: [..., K], target_prob: [..., K]
    logp = F.log_softmax(student_logits, dim=-1)
    return -(target_prob * logp).sum(dim=-1).mean()

def _sample_dirichlet_anchor(
    K: int,
    *,
    alpha: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Sample a K-dim Dirichlet soft anchor.
    alpha: concentration (smaller => lower entropy, stronger anchor)
    """
    conc = torch.full((K,), float(alpha), device=device, dtype=dtype)
    return torch.distributions.Dirichlet(conc).sample()


def _gather_tokens(x: Tensor, idx: Tensor) -> Tensor:
    """
    x:   [B, N, ...]
    idx: [B, N] or [N]
    """
    if idx.dim() == 1:
        idx = idx.unsqueeze(0).expand(x.size(0), -1)
    expand = idx.unsqueeze(-1).expand(-1, -1, *x.shape[2:])
    return x.gather(1, expand)


def _invert_perm_idx(perm_idx: Tensor) -> Tensor:
    """
    perm_idx: orig -> shifted
    returns inv_perm: shifted -> orig
    supports [N] or [B,N]
    """
    if perm_idx.dim() == 1:
        N = perm_idx.numel()
        inv = torch.empty_like(perm_idx)
        inv[perm_idx] = torch.arange(N, device=perm_idx.device, dtype=perm_idx.dtype)
        return inv
    B, N = perm_idx.shape
    inv = torch.empty_like(perm_idx)
    ar = torch.arange(N, device=perm_idx.device, dtype=perm_idx.dtype).unsqueeze(0).expand(B, -1)
    inv.scatter_(1, perm_idx, ar)
    return inv


def _cosine_anchor_loss(x: Tensor, anchor: Tensor) -> Tensor:
    x = F.normalize(x, dim=-1)
    a = F.normalize(anchor, dim=-1).unsqueeze(0).expand_as(x)
    return (1.0 - (x * a).sum(dim=-1)).mean()


def _to_cuda_any(x: Any, *, non_blocking: bool = True) -> Any:
    """Recursively move tensors to CUDA; keep python scalars / strings unchanged."""
    if torch.is_tensor(x):
        return x.cuda(non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: _to_cuda_any(v, non_blocking=non_blocking) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cuda_any(v, non_blocking=non_blocking) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_cuda_any(v, non_blocking=non_blocking) for v in x)
    return x


# ============================================================
#  Augmented MetaArch
# ============================================================

class SSLAugmentedCropRoll(SSLMetaArch):
    """
    Adds 3 augmented losses on top of SSLMetaArch:

    1) CropResize loss:
       - head logits space (K=65536) soft-CE between cropPaste views
       - uses legacy_aug_resized["cropPaste"] and legacy_aug["cropPaste"]

    2) PatchShuffle/Roll loss:
       - iBOT-style patch logits soft-CE, requires perm_idx from shift_info
       - uses legacy_aug_resized["shift"] and legacy_aug["shift"]
       - shift_info is list[dict] (collate keeps info as list), each dict has "perm_idx"

    3) Anchor loss (TODO):
       - run in bottleneck space (e.g. 256 dim)
       - we DO NOT have head bottleneck activations exposed, so we add a small projection:
         anchor_proj: Linear(embed_dim -> bottleneck_dim)
       - anchor on CLS (always optional)
       - weak anchor on background patches from cropPaste_info["uncovered_idx"] (optional)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._cached_data: Optional[dict] = None
        self._legacy_outputs: Optional[dict] = None

        # ---- cfg helpers ----
        def _sel(key: str, default):
            v = OmegaConf.select(cfg, key)
            return default if v is None else v

        # ---- weights & temps ----
        self.cropresize_weight = float(
            _sel("legacy_augmentor.crop_resize_loss_weight", _sel("legacy_augmentor.cropresize_weight", 0.0))
        )
        self.cropresize_temp = float(_sel("legacy_augmentor.crop_resize_temp", 0.1))

        self.patchshuffle_weight = float(
            _sel("legacy_augmentor.patch_shuffle_loss_weight", _sel("legacy_augmentor.patchshuffle_weight", 0.0))
        )
        self.patchshuffle_patch_weight = float(
            _sel("legacy_augmentor.patch_shuffle_patch_weight", _sel("legacy_augmentor.patchshuffle_patch_weight", 1.0))
        )
        self.patchshuffle_cls_weight = float(
            _sel("legacy_augmentor.patch_shuffle_cls_weight", _sel("legacy_augmentor.patchshuffle_cls_weight", 1.0))
        )
        self.patchshuffle_temp = float(_sel("legacy_augmentor.patch_shuffle_temp", 0.1))
        self.patchshuffle_out_dim = int(_sel("legacy_augmentor.patchshuffle_out_dim", cfg.ibot.head_n_prototypes))
        patch_prob = OmegaConf.select(cfg, "legacy_augmentor.patch_shuffle_patch_probability")
        if patch_prob is None:
            patch_prob = OmegaConf.select(cfg, "legacy_augmentor.patchshuffle_patch_probability")
        if patch_prob is None:
            legacy_ratio = OmegaConf.select(cfg, "legacy_augmentor.patch_shuffle_patch_ratio")
            if legacy_ratio is None:
                legacy_ratio = OmegaConf.select(cfg, "legacy_augmentor.patchshuffle_patch_ratio")
            if legacy_ratio is None:
                patch_prob = 1.0
            else:
                patch_prob = legacy_ratio
                logger.warning("patch_shuffle_patch_ratio is deprecated; using it as sample probability")
        self.patchshuffle_patch_probability = float(patch_prob)
        self.patchshuffle_patch_probability = max(0.0, min(1.0, self.patchshuffle_patch_probability))
        raw_patch_min_max = OmegaConf.select(cfg, "legacy_augmentor.patch_shuffle_patch_min_max")
        if raw_patch_min_max is None:
            raw_patch_min_max = OmegaConf.select(cfg, "legacy_augmentor.patchshuffle_patch_min_max")
        if raw_patch_min_max is None:
            raw_patch_min_max = OmegaConf.select(cfg, "ibot.mask_ratio_min_max")
        self.patchshuffle_patch_min_max: Optional[Tuple[float, float]] = None
        if raw_patch_min_max is not None:
            try:
                min_v, max_v = float(raw_patch_min_max[0]), float(raw_patch_min_max[1])
                if 0.0 <= min_v <= max_v <= 1.0:
                    self.patchshuffle_patch_min_max = (min_v, max_v)
                else:
                    logger.warning("patch_shuffle_patch_min_max out of range: %s", raw_patch_min_max)
            except Exception:
                logger.warning("invalid patch_shuffle_patch_min_max: %s", raw_patch_min_max)
        if self.patchshuffle_patch_min_max is None:
            self.patchshuffle_patch_min_max = (0.0, 0.0)

        self.anchor_cls_weight = float(_sel("legacy_augmentor.anchor_cls_weight", 0.0))
        self.anchor_bg_patch_weight = float(_sel("legacy_augmentor.anchor_bg_patch_weight", 0.0))

        # ---- anchor space: project to bottleneck dim ----
        bottleneck_dim = int(_sel("legacy_augmentor.anchor_bottleneck_dim", cfg.dino.head_bottleneck_dim))
        self.anchor_proj = nn.Linear(self.embed_dim, bottleneck_dim, bias=False)

        self.anchor = nn.Parameter(torch.randn(bottleneck_dim, dtype=torch.float32))
        self.bg_anchor = nn.Parameter(torch.randn(bottleneck_dim, dtype=torch.float32))

        # patch/grid size for bg mask build (tile=16 matches your CropPaste.tile)
        self.patch_size = int(_sel("legacy_augmentor.patch_size", cfg.student.patch_size))
        self.bg_tile = int(_sel("legacy_augmentor.bg_tile", 16))
        
        dino_use_history = bool(_sel("dino.use_history", False))
        dino_head_cache = bool(_sel("dino.head_cache", False))
        dino_head_blance = bool(_sel("dino.head_blance_prototype", False))
        dino_history_cache_size = int(_sel("dino.history_cache_size", 4096))
        dino_head_cache_size = int(_sel("dino.head_cache_size", 4096))

        def _build_dino_loss():
            if dino_use_history:
                return DINOLoss_skcache(
                    self.dino_out_dim,
                    use_history=True,
                    history_cache_size=dino_history_cache_size,
                    cfg=cfg,
                )
            if dino_head_cache:
                return DINOLoss_skcache(
                    self.dino_out_dim,
                    student_temp=0.1,
                    center_momentum=0.9,
                    use_sinkhorn_queue=True,
                    sk_cache=dino_head_cache_size,
                )
            if dino_head_blance:
                return DINOLoss_skcache(
                    self.dino_out_dim,
                    student_temp=0.1,
                    center_momentum=0.9,
                    use_blance_p=True,
                    blance_alpha=1.0,
                    blance_momentum=0.9,
                )
            return DINOLoss(self.dino_out_dim)

        self.cropresize_loss = _build_dino_loss()
        self.cropresize_loss_resized = _build_dino_loss()
        
        # patch-shuffle: separate centers/heads for cls (DINO) and patch (iBOT) on resized/original
        self.patchshuffle_cls_loss = _build_dino_loss()
        self.patchshuffle_cls_loss_resized = _build_dino_loss()

        ibot_use_history = bool(_sel("ibot.use_history", False))
        ibot_history_cache_size = int(_sel("ibot.history_cache_size", 20000))
        if ibot_use_history:
            self.patchshuffle_patch_loss = iBOTPatchLoss(
                self.patchshuffle_out_dim,
                use_history=True,
                history_cache_size=ibot_history_cache_size,
            )
            self.patchshuffle_patch_loss_resized = iBOTPatchLoss(
                self.patchshuffle_out_dim,
                use_history=True,
                history_cache_size=ibot_history_cache_size,
            )
        else:
            self.patchshuffle_patch_loss = iBOTPatchLoss(self.patchshuffle_out_dim, use_sk_cache=False)
            self.patchshuffle_patch_loss_resized = iBOTPatchLoss(self.patchshuffle_out_dim, use_sk_cache=False)
        
        
        self._batch_meta = {}
        self._patchshuffle_mask_generator: Optional[MaskingGenerator] = None
        self._patchshuffle_mask_grid: Optional[Tuple[int, int]] = None
        self._patchshuffle_mask_ratio_max: Optional[float] = None
        
        # ---- K-dim soft anchor (head space) ----
        self.anchor_k_weight = float(_sel("legacy_augmentor.anchor_k_weight", 0.0))
        self.anchor_k_alpha = float(_sel("legacy_augmentor.anchor_k_alpha", 0.3))

    # ------------------------------------------------------------
    # forward_backward override: cache + move legacy_aug to cuda
    # ------------------------------------------------------------
    def forward_backward(self, data, *, teacher_temp, iteration=0, **ignored_kwargs):
        self._cached_data = data
        self._batch_meta = getattr(self, "_batch_meta", {})
        self._legacy_outputs = {}

        # move legacy aug dicts (including tensors nested inside info list[dict]) to cuda
        if "legacy_aug_resized" in data and isinstance(data["legacy_aug_resized"], dict):
            data["legacy_aug_resized"] = _to_cuda_any(data["legacy_aug_resized"], non_blocking=True)
        if "legacy_aug" in data and isinstance(data["legacy_aug"], dict):
            data["legacy_aug"] = _to_cuda_any(data["legacy_aug"], non_blocking=True)

        try:
            return super().forward_backward(data, teacher_temp=teacher_temp, iteration=iteration, **ignored_kwargs)
        finally:
            self._cached_data = None
            self._legacy_outputs = None

    # ------------------------------------------------------------
    # augmented loss pieces
    # ------------------------------------------------------------
    @torch.no_grad()
    def _stack_info_field(self, info_list: list, field: str) -> Optional[Tensor]:
        """
        info_list: list[dict] length B (your collate keeps info as list)
        returns stacked tensor [B, ...] or None
        """
        if not isinstance(info_list, list) or len(info_list) == 0:
            return None
        v0 = info_list[0]
        if not isinstance(v0, dict) or field not in v0:
            return None
        vals = []
        for d in info_list:
            if not isinstance(d, dict) or field not in d:
                return None
            v = d[field]
            if not torch.is_tensor(v):
                return None
            vals.append(v)
        try:
            return torch.stack(vals, dim=0)
        except Exception:
            return None

    def _get_legacy_mask(self, legacy: Mapping[str, Any], key: str) -> Optional[Tensor]:
        expected_b = self._batch_meta.get("B")
        mask = legacy.get(key)
        if not torch.is_tensor(mask):
            return None
        if mask.dim() != 2:
            logger.warning("legacy %s mask shape: %s", key, mask.shape)
            return None
        if expected_b is not None and mask.shape[0] != expected_b:
            logger.warning("legacy %s mask batch mismatch: %s vs expected %s", key, mask.shape, expected_b)
            return None
        return mask

    def _get_legacy_aug_inputs(self) -> Optional[Tuple[Mapping[str, Any], Mapping[str, Any], Optional[Tensor], Optional[Tensor]]]:
        if self._cached_data is None:
            return None
        legacy_r = self._cached_data.get("legacy_aug_resized", None)
        legacy_o = self._cached_data.get("legacy_aug", None)
        if legacy_r is None and legacy_o is None:
            return None
        if legacy_r is None:
            legacy_r = legacy_o
        if legacy_o is None:
            legacy_o = legacy_r
        if not isinstance(legacy_r, dict) or not isinstance(legacy_o, dict):
            return None
        legacy_r_raw = legacy_r.get("raw")
        legacy_o_raw = legacy_o.get("raw")
        return legacy_r, legacy_o, legacy_r_raw, legacy_o_raw

    def _get_shift_perm_idx(
        self, legacy_r: Mapping[str, Any], legacy_o: Mapping[str, Any]
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if "shift_info" not in legacy_r or "shift_info" not in legacy_o:
            return None
        perm_idx_r = self._stack_info_field(legacy_r["shift_info"], "perm_idx")
        perm_idx_o = self._stack_info_field(legacy_o["shift_info"], "perm_idx")
        if perm_idx_r is None or perm_idx_o is None:
            return None
        expected_b = self._batch_meta.get("B")
        if expected_b is not None:
            if perm_idx_r.shape[0] != expected_b:
                logger.warning("perm_idx_r batch mismatch: %s vs expected %s", perm_idx_r.shape, expected_b)
                return None
            if perm_idx_o.shape[0] != expected_b:
                logger.warning("perm_idx_o batch mismatch: %s vs expected %s", perm_idx_o.shape, expected_b)
                return None
        if perm_idx_r.shape != perm_idx_o.shape:
            logger.warning("perm_idx shapes differ: %s vs %s", perm_idx_r.shape, perm_idx_o.shape)
            return None
        return perm_idx_r, perm_idx_o

    def _infer_patch_grid_hw(self, x: Optional[Tensor], n_tokens: int) -> Tuple[int, int]:
        if torch.is_tensor(x):
            H = int(x.shape[-2])
            W = int(x.shape[-1])
            if self.patch_size > 0 and H % self.patch_size == 0 and W % self.patch_size == 0:
                grid_h = H // self.patch_size
                grid_w = W // self.patch_size
                if grid_h * grid_w == n_tokens:
                    return grid_h, grid_w
        side = int(math.sqrt(n_tokens))
        if side * side == n_tokens:
            return side, side
        return 1, n_tokens

    def _get_patchshuffle_mask_generator(self, grid_hw: Tuple[int, int], ratio_max: float) -> MaskingGenerator:
        if (
            self._patchshuffle_mask_generator is None
            or self._patchshuffle_mask_grid != grid_hw
            or self._patchshuffle_mask_ratio_max != ratio_max
        ):
            n_tokens = grid_hw[0] * grid_hw[1]
            max_num_patches = max(1, int(n_tokens * ratio_max))
            self._patchshuffle_mask_generator = MaskingGenerator(input_size=grid_hw, max_num_patches=max_num_patches)
            self._patchshuffle_mask_grid = grid_hw
            self._patchshuffle_mask_ratio_max = ratio_max
        return self._patchshuffle_mask_generator

    def _sample_patchshuffle_mask(
        self,
        *,
        n_tokens: int,
        batch_size: int,
        device: torch.device,
        grid_source: Optional[Tensor],
    ) -> Optional[Tensor]:
        if batch_size <= 0 or n_tokens <= 0:
            return None
        ratio_min, ratio_max = self.patchshuffle_patch_min_max
        ratio_min = max(0.0, min(1.0, ratio_min))
        ratio_max = max(0.0, min(1.0, ratio_max))
        if ratio_max <= 0.0:
            return torch.zeros((batch_size, n_tokens), dtype=torch.bool, device=device)
        n_samples_masked = int(batch_size * self.patchshuffle_patch_probability)
        if n_samples_masked <= 0:
            return torch.zeros((batch_size, n_tokens), dtype=torch.bool, device=device)
        grid_hw = self._infer_patch_grid_hw(grid_source, n_tokens)
        mask_generator = self._get_patchshuffle_mask_generator(grid_hw, ratio_max)
        probs = torch.linspace(ratio_min, ratio_max, n_samples_masked + 1)
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_max = probs[i + 1]
            mask = torch.BoolTensor(mask_generator(int(n_tokens * prob_max)))
            if self.cfg.ibot.mask_random_circular_shift:
                shift_x, shift_y = (
                    random.randint(0, mask.shape[0] - 1),
                    random.randint(0, mask.shape[1] - 1),
                )
                mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
            masks_list.append(mask)
        for _ in range(n_samples_masked, batch_size):
            masks_list.append(torch.BoolTensor(mask_generator(0)))
        random.shuffle(masks_list)
        return torch.stack(masks_list).flatten(1).to(device=device)

    def _get_patchshuffle_sample_mask(
        self,
        *,
        n_tokens: int,
        batch_size: int,
        device: torch.device,
        grid_source: Optional[Tensor],
    ) -> Optional[Tensor]:
        masks = None
        if self._legacy_outputs is not None:
            masks = self._legacy_outputs.get("patchshuffle_sample_mask")
        if torch.is_tensor(masks):
            if masks.shape == (batch_size, n_tokens):
                if masks.device != device:
                    masks = masks.to(device=device)
                    if self._legacy_outputs is not None:
                        self._legacy_outputs["patchshuffle_sample_mask"] = masks
                return masks
            logger.warning("patchshuffle_sample_mask shape mismatch: %s vs (%s, %s)", masks.shape, batch_size, n_tokens)
        masks = self._sample_patchshuffle_mask(
            n_tokens=n_tokens,
            batch_size=batch_size,
            device=device,
            grid_source=grid_source,
        )
        if masks is not None and self._legacy_outputs is not None:
            self._legacy_outputs["patchshuffle_sample_mask"] = masks
        return masks

    def _apply_patchshuffle_exclusion(self, sample_mask: Tensor, exclude_mask: Optional[Tensor]) -> Tensor:
        if exclude_mask is None:
            return sample_mask
        if not torch.is_tensor(exclude_mask):
            return sample_mask
        if exclude_mask.shape != sample_mask.shape:
            logger.warning("patchshuffle exclude mask shape mismatch: %s vs %s", exclude_mask.shape, sample_mask.shape)
            return sample_mask
        if exclude_mask.device != sample_mask.device:
            exclude_mask = exclude_mask.to(device=sample_mask.device)
        return sample_mask & ~exclude_mask

    def _build_legacy_teacher_outputs(
        self,
        *,
        legacy_r: Mapping[str, Any],
        legacy_o: Mapping[str, Any],
        legacy_r_raw: Optional[Tensor],
        legacy_o_raw: Optional[Tensor],
    ) -> Optional[Dict[str, Tensor]]:
        if legacy_r_raw is None or legacy_o_raw is None:
            return None
        outputs: Dict[str, Tensor] = {}

        teacher_resized_out, teacher_original_out = self.teacher.backbone(
            [legacy_r_raw, legacy_o_raw], masks=[None, None], is_training=True
        )
        teacher_cls_pre = torch.cat(
            [teacher_resized_out["x_norm_clstoken"], teacher_original_out["x_norm_clstoken"]], dim=0
        )  # [2B, D]
        teacher_cls_logits_all = self.teacher.dino_head(teacher_cls_pre)  # [2B, K]
        teacher_cls_logits_resized, teacher_cls_logits_original = teacher_cls_logits_all.chunk(2, dim=0)
        outputs["raw_cls_logits_resized"] = teacher_cls_logits_resized
        outputs["raw_cls_logits_original"] = teacher_cls_logits_original

        if self.patchshuffle_weight > 0.0:
            teacher_patch_tokens_resized = teacher_resized_out["x_norm_patchtokens"]
            teacher_patch_tokens_original = teacher_original_out["x_norm_patchtokens"]
            n_tokens = teacher_patch_tokens_resized.shape[1]
            perm_pair = self._get_shift_perm_idx(legacy_r, legacy_o)
            if perm_pair is not None:
                perm_idx_resized, perm_idx_original = perm_pair
                sample_mask = self._get_patchshuffle_sample_mask(
                    n_tokens=n_tokens,
                    batch_size=teacher_patch_tokens_resized.shape[0],
                    device=teacher_patch_tokens_resized.device,
                    grid_source=legacy_r_raw,
                )
                if torch.is_tensor(sample_mask) and sample_mask.any():
                    shift_mask_resized = self._get_legacy_mask(legacy_r, "shift_mask")
                    shift_mask_original = self._get_legacy_mask(legacy_o, "shift_mask")
                    patch_mask_resized = self._apply_patchshuffle_exclusion(sample_mask, shift_mask_resized)
                    patch_mask_original = self._apply_patchshuffle_exclusion(sample_mask, shift_mask_original)

                    inv_perm_resized = _invert_perm_idx(perm_idx_resized)
                    inv_perm_original = _invert_perm_idx(perm_idx_original)
                    teacher_patch_tokens_resized = _gather_tokens(teacher_patch_tokens_resized, inv_perm_resized)
                    teacher_patch_tokens_original = _gather_tokens(teacher_patch_tokens_original, inv_perm_original)

                    if patch_mask_resized.any():
                        teacher_patch_logits_resized = self.teacher.ibot_head(
                            teacher_patch_tokens_resized[patch_mask_resized]
                        )
                        outputs["raw_patch_logits_resized"] = teacher_patch_logits_resized
                        outputs["patchshuffle_masks_resized"] = patch_mask_resized
                    if patch_mask_original.any():
                        teacher_patch_logits_original = self.teacher.ibot_head(
                            teacher_patch_tokens_original[patch_mask_original]
                        )
                        outputs["raw_patch_logits_original"] = teacher_patch_logits_original
                        outputs["patchshuffle_masks_original"] = patch_mask_original
        return outputs

    def _build_legacy_student_outputs(
        self,
        *,
        legacy_r: Mapping[str, Any],
        legacy_o: Mapping[str, Any],
    ) -> Optional[Dict[str, Tensor]]:
        outputs: Dict[str, Tensor] = {}
        expected_b = self._batch_meta.get("B")

        if self.cropresize_weight > 0.0 and "cropPaste" in legacy_r and "cropPaste" in legacy_o:
            crop_resized = legacy_r["cropPaste"]
            crop_original = legacy_o["cropPaste"]
            if torch.is_tensor(crop_resized) and torch.is_tensor(crop_original):
                if expected_b is None or (crop_resized.shape[0] == expected_b and crop_original.shape[0] == expected_b):
                    crop_mask_resized = self._get_legacy_mask(legacy_r, "cropPaste_mask")
                    crop_mask_original = self._get_legacy_mask(legacy_o, "cropPaste_mask")
                    if self.is_distillation_enabled:
                        crop_mask_resized = None
                        crop_mask_original = None
                    student_crop_resized_out, student_crop_original_out = self.student.backbone(
                        [crop_resized, crop_original],
                        masks=[crop_mask_resized, crop_mask_original],
                        is_training=True,
                    )
                    student_crop_cls_pre = torch.cat(
                        [
                            student_crop_resized_out["x_norm_clstoken"],
                            student_crop_original_out["x_norm_clstoken"],
                        ],
                        dim=0,
                    )
                    student_crop_cls_logits_all = self.student.dino_head(student_crop_cls_pre)
                    student_crop_cls_logits_resized, student_crop_cls_logits_original = (
                        student_crop_cls_logits_all.chunk(2, dim=0)
                    )
                    outputs["cropresize_cls_logits_resized"] = student_crop_cls_logits_resized
                    outputs["cropresize_cls_logits_original"] = student_crop_cls_logits_original
                else:
                    logger.warning(
                        "cropPaste batch mismatch: got %s and %s, expected %s",
                        crop_resized.shape,
                        crop_original.shape,
                        expected_b,
                    )

        if self.patchshuffle_weight > 0.0 and "shift" in legacy_r and "shift" in legacy_o:
            shift_resized = legacy_r["shift"]
            shift_original = legacy_o["shift"]
            if torch.is_tensor(shift_resized) and torch.is_tensor(shift_original):
                if expected_b is None or (shift_resized.shape[0] == expected_b and shift_original.shape[0] == expected_b):
                    perm_pair = self._get_shift_perm_idx(legacy_r, legacy_o)
                    shift_mask_resized = self._get_legacy_mask(legacy_r, "shift_mask")
                    shift_mask_original = self._get_legacy_mask(legacy_o, "shift_mask")
                    if self.is_distillation_enabled:
                        shift_mask_resized = None
                        shift_mask_original = None
                    student_shift_resized_out, student_shift_original_out = self.student.backbone(
                        [shift_resized, shift_original],
                        masks=[shift_mask_resized, shift_mask_original],
                        is_training=True,
                    )
                    student_patch_tokens_resized = student_shift_resized_out["x_norm_patchtokens"]
                    student_patch_tokens_original = student_shift_original_out["x_norm_patchtokens"]

                    if perm_pair is not None:
                        sample_mask = self._get_patchshuffle_sample_mask(
                            n_tokens=student_patch_tokens_resized.shape[1],
                            batch_size=student_patch_tokens_resized.shape[0],
                            device=student_patch_tokens_resized.device,
                            grid_source=shift_resized,
                        )
                        if torch.is_tensor(sample_mask) and sample_mask.any():
                            patch_mask_resized = self._apply_patchshuffle_exclusion(sample_mask, shift_mask_resized)
                            patch_mask_original = self._apply_patchshuffle_exclusion(sample_mask, shift_mask_original)
                            if patch_mask_resized.any():
                                student_patch_logits_resized = self.student.ibot_head(
                                    student_patch_tokens_resized[patch_mask_resized]
                                )
                                outputs["patchshuffle_patch_logits_resized"] = student_patch_logits_resized
                                outputs["patchshuffle_masks_resized"] = patch_mask_resized
                            if patch_mask_original.any():
                                student_patch_logits_original = self.student.ibot_head(
                                    student_patch_tokens_original[patch_mask_original]
                                )
                                outputs["patchshuffle_patch_logits_original"] = student_patch_logits_original
                                outputs["patchshuffle_masks_original"] = patch_mask_original

                    student_shift_cls_pre = torch.cat(
                        [
                            student_shift_resized_out["x_norm_clstoken"],
                            student_shift_original_out["x_norm_clstoken"],
                        ],
                        dim=0,
                    )
                    student_shift_cls_logits_all = self.student.dino_head(student_shift_cls_pre)
                    student_shift_cls_logits_resized, student_shift_cls_logits_original = (
                        student_shift_cls_logits_all.chunk(2, dim=0)
                    )
                    outputs["patchshuffle_cls_logits_resized"] = student_shift_cls_logits_resized
                    outputs["patchshuffle_cls_logits_original"] = student_shift_cls_logits_original
                else:
                    logger.warning(
                        "shift batch mismatch: got %s and %s, expected %s",
                        shift_resized.shape,
                        shift_original.shape,
                        expected_b,
                    )
        return outputs or None

    def get_teacher_output(self, images, *, upperbound, mask_indices_list, teacher_temp, n_masked_patches_tensor, iteration=0, logger_freq=0,):
        teacher_global = super().get_teacher_output(
            images,
            upperbound=upperbound,
            mask_indices_list=mask_indices_list,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            iteration=iteration,
            logger_freq=logger_freq,
        )
        if self.cropresize_weight <= 0.0 and self.patchshuffle_weight <= 0.0:
            return teacher_global
        legacy_inputs = self._get_legacy_aug_inputs()
        if legacy_inputs is None:
            return teacher_global
        legacy_r, legacy_o, legacy_r_raw, legacy_o_raw = legacy_inputs
        if self._legacy_outputs is None:
            return teacher_global
        teacher_legacy = self._build_legacy_teacher_outputs(
            legacy_r=legacy_r,
            legacy_o=legacy_o,
            legacy_r_raw=legacy_r_raw,
            legacy_o_raw=legacy_o_raw,
        )
        if teacher_legacy is not None:
            self._legacy_outputs["teacher"] = teacher_legacy
            teacher_global["legacy_aug"] = teacher_legacy
        return teacher_global

    def get_student_output(self, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
        student_global, student_local = super().get_student_output(
            global_crops=global_crops,
            local_crops=local_crops,
            upperbound=upperbound,
            masks=masks,
            mask_indices_list=mask_indices_list,
        )
        if self.cropresize_weight <= 0.0 and self.patchshuffle_weight <= 0.0:
            return student_global, student_local
        legacy_inputs = self._get_legacy_aug_inputs()
        if legacy_inputs is None:
            return student_global, student_local
        legacy_r, legacy_o, legacy_r_raw, legacy_o_raw = legacy_inputs
        if self._legacy_outputs is None:
            return student_global, student_local
        student_legacy = self._build_legacy_student_outputs(legacy_r=legacy_r, legacy_o=legacy_o)
        if student_legacy is not None:
            self._legacy_outputs["student"] = student_legacy
            student_global["legacy_aug"] = student_legacy
        return student_global, student_local

    def _build_bg_mask_from_uncovered_idx(self, uncovered_list: list, *, H: int, W: int, device) -> Optional[Tensor]:
        """
        uncovered_list: list[tensor] length B, each tensor: [Mi] uncovered tile indices (linear over R*Cc)
        returns bg_mask: [B, N] bool mask over patch tokens order (N = (H/ps)*(W/ps))
        Assumptions:
          - CropPaste.tile == patch_size (16) OR at least same grid as patch tokens.
          - patch tokens order is row-major over (H/ps, W/ps)
        """
        if not isinstance(uncovered_list, list) or len(uncovered_list) == 0:
            return None
        # Use tile grid (your CropPaste uses tile=t, default 16)
        t = self.bg_tile
        R = H // t
        C = W // t
        if R <= 0 or C <= 0:
            return None
        N = R * C
        bg = torch.zeros((len(uncovered_list), N), dtype=torch.bool, device=device)
        for i, idx in enumerate(uncovered_list):
            if not torch.is_tensor(idx):
                continue
            idx = idx.to(device=device)
            if idx.numel() == 0:
                continue
            idx = idx.clamp(min=0, max=N - 1).long()
            bg[i, idx] = True
        return bg

    def _loss_cropresize_ce_from_legacy(
        self,
        *,
        legacy_student: Mapping[str, Tensor],
        legacy_teacher: Mapping[str, Tensor],
        iteration: int = 0,
        logger_freq: int = 0,
    ) -> Optional[Tensor]:
        if legacy_student is None or legacy_teacher is None:
            return None
        student_cls_logits_resized = legacy_student.get("cropresize_cls_logits_resized")
        student_cls_logits_original = legacy_student.get("cropresize_cls_logits_original")
        teacher_cls_logits_resized = legacy_teacher.get("raw_cls_logits_resized")
        teacher_cls_logits_original = legacy_teacher.get("raw_cls_logits_original")
        if any(x is None for x in [
            student_cls_logits_resized,
            student_cls_logits_original,
            teacher_cls_logits_resized,
            teacher_cls_logits_original,
        ]):
            return None

        teacher_r_targets = self.cropresize_loss_resized.sinkhorn_knopp_teacher(
            teacher_cls_logits_resized,
            teacher_temp=self.cropresize_temp,
            iteration=iteration,
            logger_freq=logger_freq,
            logger_loss="cropresize_resized",
        )
        teacher_o_targets = self.cropresize_loss.sinkhorn_knopp_teacher(
            teacher_cls_logits_original,
            teacher_temp=self.cropresize_temp,
            iteration=iteration,
            logger_freq=logger_freq,
            logger_loss="cropresize_original",
        )
        loss_crop_resize = self.cropresize_loss_resized(
            student_cls_logits_resized.unsqueeze(0), teacher_r_targets.unsqueeze(0)
        ) + self.cropresize_loss(
            student_cls_logits_original.unsqueeze(0), teacher_o_targets.unsqueeze(0)
        )
        return loss_crop_resize

    def _loss_patchshuffle_ibot_from_legacy(
        self,
        *,
        legacy_student: Mapping[str, Tensor],
        legacy_teacher: Mapping[str, Tensor],
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if legacy_student is None or legacy_teacher is None:
            return None

        student_patch_logits_resized = legacy_student.get("patchshuffle_patch_logits_resized")
        student_patch_logits_original = legacy_student.get("patchshuffle_patch_logits_original")
        teacher_patch_logits_resized = legacy_teacher.get("raw_patch_logits_resized")
        teacher_patch_logits_original = legacy_teacher.get("raw_patch_logits_original")
        patch_masks_resized = legacy_student.get("patchshuffle_masks_resized")
        patch_masks_original = legacy_student.get("patchshuffle_masks_original")
        if patch_masks_resized is None:
            patch_masks_resized = legacy_teacher.get("patchshuffle_masks_resized")
        if patch_masks_original is None:
            patch_masks_original = legacy_teacher.get("patchshuffle_masks_original")

        patch_losses = []
        if torch.is_tensor(patch_masks_resized) and patch_masks_resized.any():
            if student_patch_logits_resized is None or teacher_patch_logits_resized is None:
                return None
            if patch_masks_resized.device != student_patch_logits_resized.device:
                patch_masks_resized = patch_masks_resized.to(device=student_patch_logits_resized.device)
            if teacher_patch_logits_resized.shape[0] == student_patch_logits_resized.shape[0]:
                teacher_r_count = torch.full(
                    (1,),
                    teacher_patch_logits_resized.shape[0],
                    dtype=torch.long,
                    device=teacher_patch_logits_resized.device,
                )
                teacher_r_patch_targets = self.patchshuffle_patch_loss_resized.sinkhorn_knopp_teacher(
                    teacher_patch_logits_resized,
                    teacher_temp=self.patchshuffle_temp,
                    n_masked_patches_tensor=teacher_r_count,
                )
                loss_resized = self.patchshuffle_patch_loss_resized.forward_masked(
                    student_patch_logits_resized,
                    teacher_r_patch_targets,
                    student_masks_flat=patch_masks_resized,
                )
                patch_losses.append(loss_resized)
            else:
                logger.warning(
                    "patchshuffle resized logits length mismatch: student=%s teacher=%s",
                    student_patch_logits_resized.shape[0],
                    teacher_patch_logits_resized.shape[0],
                )

        if torch.is_tensor(patch_masks_original) and patch_masks_original.any():
            if student_patch_logits_original is None or teacher_patch_logits_original is None:
                return None
            if patch_masks_original.device != student_patch_logits_original.device:
                patch_masks_original = patch_masks_original.to(device=student_patch_logits_original.device)
            if teacher_patch_logits_original.shape[0] == student_patch_logits_original.shape[0]:
                teacher_o_count = torch.full(
                    (1,),
                    teacher_patch_logits_original.shape[0],
                    dtype=torch.long,
                    device=teacher_patch_logits_original.device,
                )
                teacher_o_patch_targets = self.patchshuffle_patch_loss.sinkhorn_knopp_teacher(
                    teacher_patch_logits_original,
                    teacher_temp=self.patchshuffle_temp,
                    n_masked_patches_tensor=teacher_o_count,
                )
                loss_original = self.patchshuffle_patch_loss.forward_masked(
                    student_patch_logits_original,
                    teacher_o_patch_targets,
                    student_masks_flat=patch_masks_original,
                )
                patch_losses.append(loss_original)
            else:
                logger.warning(
                    "patchshuffle original logits length mismatch: student=%s teacher=%s",
                    student_patch_logits_original.shape[0],
                    teacher_patch_logits_original.shape[0],
                )

        patch_loss = None
        if patch_losses:
            patch_loss = sum(patch_losses) / len(patch_losses)

        student_cls_logits_resized = legacy_student.get("patchshuffle_cls_logits_resized")
        student_cls_logits_original = legacy_student.get("patchshuffle_cls_logits_original")
        teacher_cls_logits_resized = legacy_teacher.get("raw_cls_logits_resized")
        teacher_cls_logits_original = legacy_teacher.get("raw_cls_logits_original")
        if any(x is None for x in [
            student_cls_logits_resized,
            student_cls_logits_original,
            teacher_cls_logits_resized,
            teacher_cls_logits_original,
        ]):
            return None

        teacher_r_cls_targets = self.patchshuffle_cls_loss_resized.sinkhorn_knopp_teacher(
            teacher_cls_logits_resized, teacher_temp=self.patchshuffle_temp
        )
        teacher_o_cls_targets = self.patchshuffle_cls_loss.sinkhorn_knopp_teacher(
            teacher_cls_logits_original, teacher_temp=self.patchshuffle_temp
        )
        cls_loss = self.patchshuffle_cls_loss_resized(
            student_cls_logits_resized.unsqueeze(0), teacher_r_cls_targets.unsqueeze(0)
        ) + self.patchshuffle_cls_loss(
            student_cls_logits_original.unsqueeze(0), teacher_o_cls_targets.unsqueeze(0)
        )
        if patch_loss is None:
            patch_loss = cls_loss.new_zeros(())

        return patch_loss, cls_loss

    def _loss_anchor_cls_from_student_global(self, *, student_global: Mapping[str, Any]) -> Tensor:
        # student_global["cls_pre_head"]: [n_global_crops,B,D]
        # anchor space is projected bottleneck
        cls = student_global["cls_pre_head"].flatten(0, 1)            # [(n_global*B), D]
        proj = self.anchor_proj(cls)                                   # [(..), Db]
        return _cosine_anchor_loss(proj, self.anchor.to(device=proj.device, dtype=proj.dtype))

    def _loss_bg_patch_anchor_from_legacy(
        self,
        *,
        legacy_o: Mapping[str, Any],
        student_global: Mapping[str, Any],
        global_crops: Tensor,  # [2B,3,H,W] in forward_backward parent
    ) -> Optional[Tensor]:
        expected_b = self._batch_meta.get("B")
        # Need cropPaste_info["uncovered_idx"] list[tensor] length B
        if "cropPaste_info" not in legacy_o:
            return None
        info_list = legacy_o["cropPaste_info"]  # list[dict]
        if not isinstance(info_list, list) or len(info_list) == 0:
            return None
        if expected_b is not None and len(info_list) != expected_b:
            logger.warning("cropPaste_info length mismatch: %s vs expected %s", len(info_list), expected_b)
            return None

        # extract uncovered_idx list (variable length per sample)
        uncovered = []
        for d in info_list:
            if not isinstance(d, dict) or "uncovered_idx" not in d:
                uncovered.append(torch.zeros((0,), dtype=torch.long, device=global_crops.device))
                continue
            u = d["uncovered_idx"]
            if torch.is_tensor(u):
                uncovered.append(u)
            else:
                uncovered.append(torch.zeros((0,), dtype=torch.long, device=global_crops.device))

        # infer H,W from cropPaste tensor if exists, else from global_crops
        if "cropPaste" in legacy_o and torch.is_tensor(legacy_o["cropPaste"]):
            H = int(legacy_o["cropPaste"].shape[-2])
            W = int(legacy_o["cropPaste"].shape[-1])
        else:
            H = int(global_crops.shape[-2])
            W = int(global_crops.shape[-1])

        bg_mask = self._build_bg_mask_from_uncovered_idx(uncovered, H=H, W=W, device=global_crops.device)
        if bg_mask is None:
            return None

        # student_global patch tokens: [n_global_crops,B,N,D]
        patch = student_global["patch_pre_head"]  # [2,B,N,D]
        # use only first global crop's patches to match legacy_aug batch size B (legacy aug is per-sample, not per-crop)
        patch0 = patch[0]                          # [B,N,D]
        if expected_b is not None and patch0.shape[0] != expected_b:
            logger.warning("patch batch mismatch: %s vs expected %s", patch0.shape, expected_b)
            return None
        proj = self.anchor_proj(patch0)            # [B,N,Db]

        # bg select
        x = proj[bg_mask]                          # [M,Db]
        if x.numel() == 0:
            return proj.new_zeros(())
        return _cosine_anchor_loss(x, self.bg_anchor.to(device=x.device, dtype=x.dtype))

    # ------------------------------------------------------------
    # compute_losses override: add 3 losses with weights
    # ------------------------------------------------------------
    def compute_losses(
        self,
        *,
        teacher_global,
        student_global,
        student_local,
        gram_global,
        masks,
        mask_indices_list,
        masks_weight,
        iteration,
        logger_freq=0,
    ):
        loss_acc, loss_dict = super().compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
            logger_freq=logger_freq,
        )

        if self._cached_data is None:
            return loss_acc, loss_dict

        legacy_r = self._cached_data.get("legacy_aug_resized", None)
        legacy_o = self._cached_data.get("legacy_aug", None)
        if legacy_r is None:
            legacy_r = legacy_o
        legacy_student = None
        legacy_teacher = None
        if isinstance(self._legacy_outputs, dict):
            legacy_student = self._legacy_outputs.get("student")
            legacy_teacher = self._legacy_outputs.get("teacher")

        # 1) CropResize CE (CLS logits)
        if self.cropresize_weight > 0.0:
            loss = self._loss_cropresize_ce_from_legacy(
                legacy_student=legacy_student,
                legacy_teacher=legacy_teacher,
                iteration=iteration,
                logger_freq=logger_freq,
            )
            if loss is not None:
                loss_dict["aug/cropresize_ce"] = loss
                loss_dict["aug/cropresize_weight"] = float(self.cropresize_weight)
                loss_acc = loss_acc + self.cropresize_weight * loss

        # 2) PatchShuffle iBOT-style CE (patch logits + perm_idx)
        if self.patchshuffle_weight > 0.0:
            loss = self._loss_patchshuffle_ibot_from_legacy(
                legacy_student=legacy_student, legacy_teacher=legacy_teacher
            )
            if loss is not None:
                if isinstance(loss, tuple):
                    patch_loss, cls_loss = loss
                    loss_dict["aug/patchshuffle_patch"] = patch_loss
                    loss_dict["aug/patchshuffle_cls"] = cls_loss
                    loss_dict["aug/patchshuffle_patch_weight"] = float(self.patchshuffle_patch_weight)
                    loss_dict["aug/patchshuffle_cls_weight"] = float(self.patchshuffle_cls_weight)
                    loss = (patch_loss * self.patchshuffle_patch_weight) + (cls_loss * self.patchshuffle_cls_weight)
                loss_dict["aug/patchshuffle_ibot"] = loss
                loss_dict["aug/patchshuffle_weight"] = float(self.patchshuffle_weight)
                loss_acc = loss_acc + self.patchshuffle_weight * loss

        # 3) Anchor CLS (proj/bottleneck space)
        if self.anchor_cls_weight > 0.0:
            loss = self._loss_anchor_cls_from_student_global(student_global=student_global)
            loss_dict["aug/anchor_cls"] = loss
            loss_dict["aug/anchor_cls_weight"] = float(self.anchor_cls_weight)
            loss_acc = loss_acc + self.anchor_cls_weight * loss

        # 3b) Weak BG-patch anchor from CropPaste uncovered tiles (optional)
        if self.anchor_bg_patch_weight > 0.0 and isinstance(legacy_o, dict):
            global_crops = self._cached_data.get("collated_global_crops", None)
            if torch.is_tensor(global_crops):
                loss = self._loss_bg_patch_anchor_from_legacy(
                    legacy_o=legacy_o,
                    student_global=student_global,
                    global_crops=global_crops,
                )
                if loss is not None:
                    loss_dict["aug/anchor_bg_patch"] = loss
                    loss_dict["aug/anchor_bg_patch_weight"] = float(self.anchor_bg_patch_weight)
                    loss_acc = loss_acc + self.anchor_bg_patch_weight * loss
                    
        # 3c) K-dim Dirichlet anchor on CLS (head space, optional)
        if self.anchor_k_weight > 0.0:
            loss = self._loss_anchor_cls_kdirichlet(
                student_global=student_global
            )
            loss_dict["aug/anchor_cls_kdir"] = loss
            loss_dict["aug/anchor_cls_kdir_weight"] = float(self.anchor_k_weight)
            loss_acc = loss_acc + self.anchor_k_weight * loss


        return loss_acc, loss_dict

    def _loss_anchor_cls_kdirichlet(
        self,
        *,
        student_global: Mapping[str, Any],
    ) -> Tensor:
        """
        K-dim soft anchor on DINO head logits using Dirichlet-sampled targets.
        Operates purely in head space (logits), student only.
        """
        # student_global["cls_pre_head"]: [n_global_crops, B, D]
        cls = student_global["cls_pre_head"].flatten(0, 1)  # [M, D]

        # head logits: [M, K]
        logits = self.student.dino_head(cls)

        # sample one Dirichlet anchor per batch (broadcast over M)
        with torch.no_grad():
            y = _sample_dirichlet_anchor(
                K=logits.shape[-1],
                alpha=self.anchor_k_alpha,
                device=logits.device,
                dtype=logits.dtype,
            )
            y = y.unsqueeze(0).expand_as(logits)  # [M, K]

        return _soft_ce(logits, y)
    
    



# ------------------------------------------------------------
# Small smoke test using png_dataset from data/augmentations.py
# ------------------------------------------------------------
def _smoke_test_png_dataset():
    """
    Lightweight test:
      - load default config
      - build DataAugmentationDINO + png_dataset
      - run one forward_backward pass on CPU
    Requires resize_2/data/dataset.py with png_dataset available.
    """
    from functools import partial
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    from dinov3.data.augmentations import DataAugmentationDINO
    from dinov3.data import collate_data_and_cast, MaskingGenerator

    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "ssl_default_config.yaml")

    # transforms / dataset
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

    # mask generator + collate
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

    # dataset is defined in resize_2/data/dataset.py
    import sys
    sys.path.append(str(REPO_ROOT.parent / "resize_2"))
    from data.dataset import (
        png_dataset
    )
    from data.dataset import png_dataset  # noqa: WPS433
    batch_size = 4
    loader = DataLoader(
        png_dataset(img_transforms=aug),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return loader, cfg

def _smoke_test_train():
    loader, cfg = _smoke_test_png_dataset()
    batch = next(iter(loader))

    model = SSLAugmentedCropRoll(cfg)
    model.train()
    batch['global_batch_size'] = 4
    loss, logs = model.forward_backward(batch, teacher_temp=0.059, iteration=0)
    print("smoke loss:", loss)
    for k, v in logs.items():
        print(k, v)


if __name__ == "__main__":
    _smoke_test_train()
