import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf

from dinov3.distributed import get_process_subgroup, get_subgroup_size
logger = logging.getLogger("dinov3")

_LOADED_STATE_KEYS: set[str] | None = None


def set_loaded_state_keys(keys: set[str] | None) -> None:
    global _LOADED_STATE_KEYS
    _LOADED_STATE_KEYS = keys


def _history_key_loaded(history_key: str) -> bool | None:
    if _LOADED_STATE_KEYS is None:
        return None
    return history_key in _LOADED_STATE_KEYS or f"model.{history_key}" in _LOADED_STATE_KEYS


class CH_SK(nn.Module):
    """
    Cumulative History Sinkhorn
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
        logits_temp_max=None,
    ):
        super().__init__()
        self.K = K
        self.history_cache_size = history_cache_size
        self.register_buffer("history_Q", torch.full((K,), float("nan")))
        self._history_Q_initialized = False
        self.logits_temp_max = 30.0 if logits_temp_max is None else logits_temp_max

        # --- boost params  ---
        if cfg is not None:
            boost_alpha = OmegaConf.select(cfg, "ch_sk.boost_alpha", default=boost_alpha)
            boost_w_max = OmegaConf.select(cfg, "ch_sk.boost_w_max", default=boost_w_max)
            boost_threshold_divisor = OmegaConf.select(
                cfg, "ch_sk.boost_threshold_divisor", default=boost_threshold_divisor
            )
            boost_eps = OmegaConf.select(cfg, "ch_sk.boost_eps", default=boost_eps)
        self.boost_alpha = float(boost_alpha)
        self.boost_w_max = float(boost_w_max)
        self.boost_threshold_divisor = float(boost_threshold_divisor)
        self.boost_eps = float(boost_eps)

    def _ensure_history_Q(self, Q_local):
        if not self._history_Q_initialized:
            K, b_local = Q_local.shape
            sum_Q_local = torch.sum(Q_local, dim=1)
            if dist.is_initialized():
                dist.all_reduce(sum_Q_local, group=get_process_subgroup())
            self.history_Q = (
                sum_Q_local
                * (self.history_cache_size / b_local)
                / (get_subgroup_size() if dist.is_initialized() else 1)
            )
            self._history_Q_initialized = True

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
        loaded_state = _history_key_loaded(history_key)
        if loaded_state is None:
            history_missing = history_key in missing_keys
        else:
            history_missing = not loaded_state
        if history_missing and history_key in missing_keys:
            missing_keys.remove(history_key)
        history_has_nan = torch.isnan(self.history_Q).any()
        if history_missing or history_has_nan:
            if history_missing:
                logger.info(f"load {history_key} missing")
            if history_has_nan:
                logger.info(f"load {history_key} missing (nan)")
            self.history_Q.fill_(float("nan"))
            self._history_Q_initialized = False
        else:
            logger.info(f"load {history_key} success")
            self._history_Q_initialized = True

    def _update_history(self, Q_local):
        self._ensure_history_Q(Q_local)
        _, b_local = Q_local.shape
        self.history_Q = (
            (self.history_cache_size - b_local) / self.history_cache_size * self.history_Q
            + torch.sum(Q_local, dim=1)
        )

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
        # forward 级别可覆盖（不传则用 __init__ 的默认值）
        boost_alpha: float | None = None,
        boost_w_max: float | None = None,
        boost_threshold_divisor: float | None = None,
        boost_eps: float | None = None,
    ):
        # teacher_output: [batch, prototypes]
        teacher_output = teacher_output.float()

        world_size = get_subgroup_size() if dist.is_initialized() else 1

        logits_temp = teacher_output / teacher_temp
        logits_temp_clamp = logits_temp.clamp(min=-10, max=self.logits_temp_max)

        # Q_batch: [K, B_batch]
        Q_batch = torch.exp(logits_temp_clamp).t()

        # ===============================
        # pre-SK prototype boosting (smooth vector)
        # ===============================
        self._ensure_history_Q(Q_batch)

        # 取参数：forward 覆盖 > init 默认
        alpha = float(self.boost_alpha if boost_alpha is None else boost_alpha)
        w_max = float(self.boost_w_max if boost_w_max is None else boost_w_max)
        divisor = float(
            self.boost_threshold_divisor
            if boost_threshold_divisor is None
            else boost_threshold_divisor
        )
        eps = float(self.boost_eps if boost_eps is None else boost_eps)

        # 用 tensor eps，避免 dtype/device mismatch
        eps_t = torch.tensor(eps, device=Q_batch.device, dtype=Q_batch.dtype)

        with torch.no_grad():
            hist = self.history_Q  # [K]
            mean_hist = hist.mean()
            threshold = mean_hist / divisor

            low_mask = hist < threshold  # [K] bool
            # w: [K]
            w = abs((mean_hist / (hist + eps_t) - divisor)).pow(alpha)
            w = w.clamp(1.0, w_max)

            # 为了后面“撤销 boost”，记录哪些被 boost 以及对应系数
            boost_mask = low_mask
            boost_w = w

            if boost_mask.any():
                Q_batch[boost_mask, :] *= boost_w[boost_mask, None]

        # ===============================
        # logging
        # ===============================
        if logger is not None and logger_freq and iteration % logger_freq == 0:
            loss_tag = f"[{logger_loss}] " if logger_loss else ""
            with torch.no_grad():
                if boost_mask.any():
                    boost_idx = torch.nonzero(boost_mask, as_tuple=False).flatten()
                    # 为避免日志爆炸，只展示最小的 30 个
                    show_n = min(30, boost_idx.numel())
                    boost_hist = hist[boost_idx]
                    show_idx = boost_idx[torch.argsort(boost_hist)[:show_n]]
                    # 展示最小的 show_n 个

                    logger.info(
                        f"{loss_tag}[CHSK-BOOST][iter={iteration}] "
                        f"alpha={alpha:.3g} w_max={w_max:.3g} divisor={divisor:.3g} | "
                        f"boost_cnt={boost_idx.numel()} | "
                        f"mean=%.3e thr=%.3e | "
                        f"idx(head)={show_idx.tolist()} | "
                        f"hist(head)={['%.3e' % v for v in hist[show_idx].tolist()]} | "
                        f"w(head)={['%.3e' % v for v in boost_w[show_idx].tolist()]}"
                        % (mean_hist.item(), threshold.item())
                    )
                else:
                    logger.info(
                        f"{loss_tag}[CHSK-BOOST][iter={iteration}] no boost | "
                        f"alpha={alpha:.3g} w_max={w_max:.3g} divisor={divisor:.3g} | "
                        f"min_hist=%.3e mean=%.3e thr=%.3e"
                        % (hist.min().item(), mean_hist.item(), threshold.item())
                    )

                logger.info(
                    f"{loss_tag}iteration {iteration}, logits_temp: max {logits_temp[-1].max().item():.3e}, "
                    f"min {logits_temp[-1].min().item():.3e}, mean {logits_temp[-1].mean().item():.3e}, "
                    f"(-1, :5){['%.3e' % v for v in logits_temp[-1, :5].tolist()]}"
                )
                logger.info(
                    f"{loss_tag}iteration {iteration}, logits_temp_clamp: max {logits_temp_clamp[-1].max().item():.3e}, "
                    f"min {logits_temp_clamp[-1].min().item():.3e}, mean {logits_temp_clamp[-1].mean().item():.3e}, "
                    f"(-1, :5){['%.3e' % v for v in logits_temp_clamp[-1, :5].tolist()]}"
                )
                logger.info(
                    f"{loss_tag}iteration {iteration}, "
                    f"Q_batch_max: {Q_batch.t()[-1].max().item():.3e}, "
                    f"Q_batch_min: {Q_batch.t()[-1].min().item():.3e}, "
                    f"Q_batch_mean: {Q_batch.t()[-1].mean().item():.3e}, "
                    f"Q_batch[-1, :5]: {['%.3e' % v for v in Q_batch.t()[-1, :5].tolist()]}"
                )
                logger.info(
                    f"self.history_Q_max: {self.history_Q.max().item():.3e}, "
                    f"self.history_Q_min: {self.history_Q.min().item():.3e}, "
                    f"self.history_Q_mean: {self.history_Q.mean().item():.3e}, "
                    f"self.history_Q[:5]: {['%.3e' % v for v in self.history_Q[:5].tolist()]}"
                )

        # ===============================
        # SK
        # ===============================
        B_batch = Q_batch.shape[1]
        B = (B_batch + self.history_cache_size) * world_size
        K = Q_batch.shape[0]

        # Q: [K, 1 + B_batch] where col0 is history_Q "pseudo-sample"
        Q = torch.cat([self.history_Q.unsqueeze(1), Q_batch], dim=1)

        # make the matrix sums to 1 (global)
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        for _ in range(n_iterations):
            # row normalize -> each prototype sums to 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            # col normalize -> each sample sums to 1/B
            sum_of_columns = torch.sum(Q, dim=0, keepdim=True)
            sum_of_columns[0, 0] = sum_of_columns[0, 0] / self.history_cache_size
            Q /= sum_of_columns
            Q /= B

        Q *= B  # columns sum to 1

        # ===============================
        # undo boost BEFORE updating history
        # ===============================
        if boost_mask.any():
            Q_batch[boost_mask, :] /= boost_w[boost_mask, None]

        self._update_history(Q_batch)

        # return assignment for current batch only
        return Q[:, -B_batch:].t()
