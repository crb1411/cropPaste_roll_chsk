import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf

from dinov3.distributed import get_process_subgroup, get_subgroup_size

logger = logging.getLogger("dinov3")


class DTCH_SK(nn.Module):
    """
    Dual-temperature Cumulative History Sinkhorn

    - SK uses softened logits: temp * dt_temp_scale
    - after SK, apply power on assignments and renormalize over K
    """

    def __init__(
        self,
        K: int,
        history_cache_size: int = 3072,
        cfg=None,
        *,
        boost_alpha: float = 0.3,
        boost_w_max: float = 50.0,
        boost_threshold_divisor: float = 500.0,
        boost_eps: float = 1e-6,
        boost_enabled: bool = True,
        logits_temp_max=None,
        dt_temp_scale: float = 5.0,
        dt_exp_power: Optional[float] = None,
        history_update: str = "cache",
    ):
        super().__init__()
        self.K = K
        self.history_cache_size = history_cache_size
        self.register_buffer("history_Q", torch.full((K,), float("nan")))
        self._history_Q_initialized = False
        self.logits_temp_max = 30.0 if logits_temp_max is None else logits_temp_max

        if cfg is not None:
            boost_alpha = OmegaConf.select(cfg, "ch_sk.boost_alpha", default=boost_alpha)
            boost_w_max = OmegaConf.select(cfg, "ch_sk.boost_w_max", default=boost_w_max)
            boost_threshold_divisor = OmegaConf.select(
                cfg, "ch_sk.boost_threshold_divisor", default=boost_threshold_divisor
            )
            boost_eps = OmegaConf.select(cfg, "ch_sk.boost_eps", default=boost_eps)
            boost_enabled = OmegaConf.select(cfg, "ch_sk.boost_enabled", default=boost_enabled)
        if cfg is not None:
            dt_temp_scale = OmegaConf.select(cfg, "ch_sk.dt_temp_scale", default=dt_temp_scale)
            dt_exp_power = OmegaConf.select(cfg, "ch_sk.dt_exp_power", default=dt_exp_power)
            history_update = OmegaConf.select(cfg, "ch_sk.history_update", default=history_update)
            history_update = OmegaConf.select(cfg, "ch_sk.history_update_mode", default=history_update)
        self.boost_alpha = float(boost_alpha)
        self.boost_w_max = float(boost_w_max)
        self.boost_threshold_divisor = float(boost_threshold_divisor)
        self.boost_eps = float(boost_eps)
        # Toggle boost without removing parameters (keeps config compatibility).
        self.boost_enabled = bool(boost_enabled)

        self.dt_temp_scale = float(dt_temp_scale)
        if dt_exp_power is None:
            dt_exp_power = self.dt_temp_scale
        self.dt_exp_power = float(dt_exp_power)
        # history_update: "ema" (single vector EMA) or "cache" (LRU cache -> sum)
        mode = str(history_update).lower()
        if mode not in ("ema", "cache"):
            logger.warning("Unknown history_update %s, fallback to ema", history_update)
            mode = "ema"
        self.history_update = mode
        self._history_cache = None  # [n_cache, K], LRU ring buffer (rank-local)
        self._history_cache_pos = 0
        self._history_cache_batch = None
        self._history_cache_size_eff = None
        self._history_cache_capacity = 0

    def _history_scale(self, b_local: int) -> int:
        # Use a single scale definition for both EMA and cache modes.
        if self._use_cache_history():
            return self._effective_history_size(b_local)
        return int(self.history_cache_size)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._history_cache is None:
            return
        cache = self._history_cache if keep_vars else self._history_cache.detach()
        destination[prefix + "_history_cache"] = cache
        destination[prefix + "_history_cache_pos"] = torch.tensor(self._history_cache_pos)
        destination[prefix + "_history_cache_batch"] = torch.tensor(
            -1 if self._history_cache_batch is None else self._history_cache_batch
        )
        destination[prefix + "_history_cache_size_eff"] = torch.tensor(
            -1 if self._history_cache_size_eff is None else self._history_cache_size_eff
        )
        destination[prefix + "_history_cache_capacity"] = torch.tensor(self._history_cache_capacity)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        cache_key = prefix + "_history_cache"
        cache_pos_key = prefix + "_history_cache_pos"
        cache_batch_key = prefix + "_history_cache_batch"
        cache_size_eff_key = prefix + "_history_cache_size_eff"
        cache_capacity_key = prefix + "_history_cache_capacity"
        cache = state_dict.pop(cache_key, None)
        cache_pos = state_dict.pop(cache_pos_key, None)
        cache_batch = state_dict.pop(cache_batch_key, None)
        cache_size_eff = state_dict.pop(cache_size_eff_key, None)
        cache_capacity = state_dict.pop(cache_capacity_key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        history_key = prefix + "history_Q"
        if history_key in missing_keys:
            missing_keys.remove(history_key)
        if torch.isfinite(self.history_Q).all():
            self._history_Q_initialized = True
        else:
            self.history_Q.fill_(float("nan"))
            self._history_Q_initialized = False
        if cache is None or not torch.is_tensor(cache) or cache.numel() == 0:
            self._history_cache = None
            self._history_cache_pos = 0
            self._history_cache_batch = None
            self._history_cache_size_eff = None
            self._history_cache_capacity = 0
            return
        self._history_cache = cache
        self._history_cache_pos = int(cache_pos.item()) if cache_pos is not None else 0
        cache_batch_val = int(cache_batch.item()) if cache_batch is not None else -1
        self._history_cache_batch = None if cache_batch_val < 0 else cache_batch_val
        cache_size_eff_val = int(cache_size_eff.item()) if cache_size_eff is not None else -1
        if cache_size_eff_val < 0 and self._history_cache_batch is not None:
            cache_size_eff_val = self._history_cache_batch * int(cache.shape[0])
        self._history_cache_size_eff = None if cache_size_eff_val < 0 else cache_size_eff_val
        if cache_capacity is not None:
            self._history_cache_capacity = int(cache_capacity.item())
        else:
            self._history_cache_capacity = int(cache.shape[0]) if cache is not None else 0
        if self._history_cache_capacity > 0:
            self._history_cache_pos = self._history_cache_pos % self._history_cache_capacity
        if not self._history_Q_initialized and cache.shape[1] == self.K:
            self.history_Q = cache.sum(dim=0)
            self._history_Q_initialized = True

    def _use_cache_history(self) -> bool:
        return self.history_update == "cache"

    def _effective_history_size(self, b_local: int) -> int:
        # Floor to a multiple of batch size for cache mode.
        if b_local <= 0:
            return int(self.history_cache_size)
        eff = (int(self.history_cache_size) // b_local) * b_local
        if eff <= 0:
            eff = b_local
        return eff

    def _cache_init_mode(self, b_local: int) -> str:
        if b_local <= 0:
            return "batch"
        eff_size = self._effective_history_size(b_local)
        capacity = max(1, eff_size // b_local)
        cache_ok = (
            self._history_cache is not None
            and self._history_cache_batch == b_local
            and self._history_cache_size_eff == eff_size
            and self._history_cache_capacity == capacity
            and self._history_cache.shape[0] == capacity
            and self._history_cache.shape[1] == self.K
            and torch.isfinite(self.history_Q).all()
        )
        return "checkpoint" if cache_ok else "batch"

    def _init_history_cache(self, Q_local: torch.Tensor, *, init_mode: str) -> None:
        _, b_local = Q_local.shape
        eff_size = self._effective_history_size(b_local)
        if init_mode == "checkpoint":
            return
        # Cache capacity is in units of batches.
        capacity = max(1, eff_size // b_local)
        device = Q_local.device
        dtype = Q_local.dtype
        # Fresh start: fill cache with current batch sum (n * sum_Q_local).
        sum_Q_local = torch.sum(Q_local, dim=1)
        if dist.is_initialized():
            dist.all_reduce(sum_Q_local, group=get_process_subgroup())
        scale = self._history_scale(b_local)
        self.history_Q = (
            sum_Q_local
            * (scale / b_local)
            / (get_subgroup_size() if dist.is_initialized() else 1)
        )
        self._history_Q_initialized = True
        cache = self.history_Q.to(device=device, dtype=dtype).unsqueeze(0).repeat(capacity, 1)
        cache = cache / capacity
        # Cache is rank-local; history_Q is synchronized via all_reduce.
        self._history_cache = cache
        self._history_cache_pos = 0
        self._history_cache_batch = b_local
        self._history_cache_size_eff = eff_size
        self._history_cache_capacity = capacity

    def _ensure_history_Q(self, Q_local):
        if not self._use_cache_history():
            # EMA history: single vector with exponential update.
            if not self._history_Q_initialized:
                _, b_local = Q_local.shape
                sum_Q_local = torch.sum(Q_local, dim=1)
                if dist.is_initialized():
                    dist.all_reduce(sum_Q_local, group=get_process_subgroup())
                scale = self._history_scale(b_local)
                self.history_Q = (
                    sum_Q_local
                    * (scale / b_local)
                    / (get_subgroup_size() if dist.is_initialized() else 1)
                )
                self._history_Q_initialized = True
            return
        _, b_local = Q_local.shape
        init_mode = self._cache_init_mode(b_local)
        self._init_history_cache(Q_local, init_mode=init_mode)

    def _update_history(self, Q_local):
        if not self._use_cache_history():
            # EMA update keeps history_Q scale aligned with history_cache_size.
            self._ensure_history_Q(Q_local)
            _, b_local = Q_local.shape
            scale = self._history_scale(b_local)
            self.history_Q = ((scale - b_local) / scale) * self.history_Q + torch.sum(Q_local, dim=1)
            return
        self._ensure_history_Q(Q_local)
        sum_Q_local = torch.sum(Q_local, dim=1)
        if self._history_cache is None or self._history_cache_capacity <= 0:
            self.history_Q = sum_Q_local
            return
        if (
            self._history_cache.device != self.history_Q.device
            or self._history_cache.dtype != self.history_Q.dtype
        ):
            self._history_cache = self._history_cache.to(device=self.history_Q.device, dtype=self.history_Q.dtype)
        # LRU ring update: subtract old, add new.
        pos = self._history_cache_pos
        old = self._history_cache[pos]
        self.history_Q = self.history_Q - old + sum_Q_local
        self._history_cache[pos] = sum_Q_local
        self._history_cache_pos = (pos + 1) % self._history_cache_capacity

    @torch.no_grad()
    def forward(
        self,
        teacher_output,
        teacher_temp,
        n_masked_patches_tensor=None,
        n_iterations: int = 3,
        iteration: int = 0,
        logger_freq: int = 0,
        logger_loss: str | None = None,
        *,
        boost_alpha: float | None = None,
        boost_w_max: float | None = None,
        boost_threshold_divisor: float | None = None,
        boost_eps: float | None = None,
        dt_temp_scale: float | None = None,
        dt_exp_power: float | None = None,
        boost_enabled: bool | None = None,
    ):
        teacher_output = teacher_output.float()

        world_size = get_subgroup_size() if dist.is_initialized() else 1

        scale = float(self.dt_temp_scale if dt_temp_scale is None else dt_temp_scale)
        scale = max(scale, 1e-6)
        exp_power = float(self.dt_exp_power if dt_exp_power is None else dt_exp_power)
        exp_power = max(exp_power, 1e-6)

        logits_temp = teacher_output / (teacher_temp * scale)
        logits_temp_clamp = logits_temp.clamp(min=-10.0, max=self.logits_temp_max)

        # Q_batch_soft: [K, B_batch], used for history + SK
        Q_batch_soft = torch.exp(logits_temp_clamp).t()

        # ===============================
        # pre-SK prototype boosting (optional)
        # ===============================
        self._ensure_history_Q(Q_batch_soft)

        alpha = float(self.boost_alpha if boost_alpha is None else boost_alpha)
        w_max = float(self.boost_w_max if boost_w_max is None else boost_w_max)
        divisor = float(
            self.boost_threshold_divisor
            if boost_threshold_divisor is None
            else boost_threshold_divisor
        )
        eps = float(self.boost_eps if boost_eps is None else boost_eps)
        eps_t = torch.tensor(eps, device=Q_batch_soft.device, dtype=Q_batch_soft.dtype)
        # Allow runtime override; keep config default if not provided.
        use_boost = self.boost_enabled if boost_enabled is None else bool(boost_enabled)

        with torch.no_grad():
            hist = self.history_Q
            mean_hist = hist.mean()
            threshold = mean_hist / divisor

            if use_boost:
                low_mask = hist < threshold
                w = abs((mean_hist / (hist + eps_t) - divisor)).pow(alpha)
                w = w.clamp(1.0, w_max)
                boost_mask = low_mask
                boost_w = w
            else:
                boost_mask = torch.zeros_like(hist, dtype=torch.bool)
                boost_w = None

        Q_batch = Q_batch_soft

        
        # ===============================
        # logging (pre-boost focus)
        # ===============================
        do_log = logger is not None and logger_freq and iteration % logger_freq == 0
        hist_snapshot = None
        loss_tag = f"[{logger_loss}] " if logger_loss else ""
        if do_log:
            with torch.no_grad():
                hist_snapshot = self.history_Q.detach()
                boost_cnt = int(boost_mask.sum().item())
                logger.info(
                    f"{loss_tag}[CHSK-BOOST][iter={iteration}] "
                    f"alpha={alpha:.3g} w_max={w_max:.3g} divisor={divisor:.3g} | "
                    f"boost_cnt={boost_cnt} | "
                    f"mean=%.3e thr=%.3e"
                    % (mean_hist.item(), threshold.item())
                )
                hist_flat = hist_snapshot.flatten()
                k_hist = min(5, hist_flat.numel())
                hist_top = torch.topk(hist_flat, k_hist).values
                hist_bottom = torch.topk(-hist_flat, k_hist).values.neg()
                logger.info(
                    f"{loss_tag}history: mean=%.3e max=%.3e min=%.3e | "
                    f"top5={['%.3e' % v for v in hist_top.tolist()]} | "
                    f"bottom5={['%.3e' % v for v in hist_bottom.tolist()]}"
                    % (hist_snapshot.mean().item(), hist_snapshot.max().item(), hist_snapshot.min().item())
                )

        if boost_w is not None and boost_mask.any():
            Q_batch[boost_mask, :] *= boost_w[boost_mask, None]

        # ===============================
        # SK (softened temp)
        # ===============================
        B_batch = Q_batch.shape[1]
        # Use the same scale as history_Q for SK normalization.
        history_size = self._history_scale(B_batch)
        B = (B_batch + history_size) * world_size
        K = Q_batch.shape[0]

        Q = torch.cat([self.history_Q.unsqueeze(1), Q_batch], dim=1)

        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        for _ in range(n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            sum_of_columns = torch.sum(Q, dim=0, keepdim=True)
            sum_of_columns[0, 0] = sum_of_columns[0, 0] / history_size
            Q /= sum_of_columns
            Q /= B

        Q *= B

        # history update uses softened Q_batch (pre-boost)
        self._update_history(Q_batch_soft)

        Q_assign = Q[:, -B_batch:].t()
        if exp_power != 1.0:
            Q_assign = torch.pow(Q_assign, exp_power)
            denom = Q_assign.sum(dim=1, keepdim=True)
            Q_assign = Q_assign / denom.clamp_min(1e-25)

        if do_log:
            with torch.no_grad():
                q_soft_row = Q_batch_soft[:, -1]
                k_soft = min(5, q_soft_row.numel())
                q_soft_top = torch.topk(q_soft_row, k_soft).values
                q_soft_bottom = torch.topk(-q_soft_row, k_soft).values.neg()
                logger.info(
                    f"{loss_tag}Q_soft(pre-boost): max=%.3e min=%.3e | "
                    f"top5={['%.3e' % v for v in q_soft_top.tolist()]} | "
                    f"bottom5={['%.3e' % v for v in q_soft_bottom.tolist()]}"
                    % (q_soft_row.max().item(), q_soft_row.min().item())
                )
                q_assign_row = Q_assign[-1]
                k_assign = min(5, q_assign_row.numel())
                q_assign_top = torch.topk(q_assign_row, k_assign).values
                q_assign_bottom = torch.topk(-q_assign_row, k_assign).values.neg()
                logger.info(
                    f"{loss_tag}Q_assign: max=%.3e min=%.3e | "
                    f"top5={['%.3e' % v for v in q_assign_top.tolist()]} | "
                    f"bottom5={['%.3e' % v for v in q_assign_bottom.tolist()]}"
                    % (q_assign_row.max().item(), q_assign_row.min().item())
                )

        return Q_assign
