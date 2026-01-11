import logging
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from dinov3.distributed import get_process_subgroup, get_subgroup_size
from dinov3.loss.ch_sk import CH_SK

logger = logging.getLogger("dinov3")


class DTCH_SK(CH_SK):
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
        logits_temp_max=None,
        dt_temp_scale: float = 5.0,
        dt_exp_power: Optional[float] = None,
    ):
        super().__init__(
            K=K,
            history_cache_size=history_cache_size,
            cfg=cfg,
            boost_alpha=boost_alpha,
            boost_w_max=boost_w_max,
            boost_threshold_divisor=boost_threshold_divisor,
            boost_eps=boost_eps,
            logits_temp_max=logits_temp_max,
        )
        if cfg is not None:
            dt_temp_scale = OmegaConf.select(cfg, "ch_sk.dt_temp_scale", default=dt_temp_scale)
            dt_exp_power = OmegaConf.select(cfg, "ch_sk.dt_exp_power", default=dt_exp_power)
        self.dt_temp_scale = float(dt_temp_scale)
        if dt_exp_power is None:
            dt_exp_power = self.dt_temp_scale
        self.dt_exp_power = float(dt_exp_power)

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
        # pre-SK prototype boosting (smooth vector)
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

        with torch.no_grad():
            hist = self.history_Q
            mean_hist = hist.mean()
            threshold = mean_hist / divisor

            low_mask = hist < threshold
            w = abs((mean_hist / (hist + eps_t) - divisor)).pow(alpha)
            w = w.clamp(1.0, w_max)

            boost_mask = low_mask
            boost_w = w

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

        if boost_mask.any():
            Q_batch[boost_mask, :] *= boost_w[boost_mask, None]

        # ===============================
        # SK (softened temp)
        # ===============================
        B_batch = Q_batch.shape[1]
        B = (B_batch + self.history_cache_size) * world_size
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
            sum_of_columns[0, 0] = sum_of_columns[0, 0] / self.history_cache_size
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
