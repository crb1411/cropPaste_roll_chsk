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
        # logging
        # ===============================
        if logger is not None and logger_freq and iteration % logger_freq == 0:
            loss_tag = f"[{logger_loss}] " if logger_loss else ""
            with torch.no_grad():
                if boost_mask.any():
                    boost_idx = torch.nonzero(boost_mask, as_tuple=False).flatten()
                    show_n = min(30, boost_idx.numel())
                    boost_hist = hist[boost_idx]
                    show_idx = boost_idx[torch.argsort(boost_hist)[:show_n]]

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
                    f"{loss_tag}Q_batch_soft_max: {Q_batch_soft.t()[-1].max().item():.3e}, "
                    f"Q_batch_soft_min: {Q_batch_soft.t()[-1].min().item():.3e}, "
                    f"Q_batch_soft_mean: {Q_batch_soft.t()[-1].mean().item():.3e}"
                )
                logger.info(
                    f"self.history_Q_max: {self.history_Q.max().item():.3e}, "
                    f"self.history_Q_min: {self.history_Q.min().item():.3e}, "
                    f"self.history_Q_mean: {self.history_Q.mean().item():.3e}, "
                    f"self.history_Q[:5]: {['%.3e' % v for v in self.history_Q[:5].tolist()]}"
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
            Q_assign = Q_assign / denom.clamp_min(1e-12)

        return Q_assign
