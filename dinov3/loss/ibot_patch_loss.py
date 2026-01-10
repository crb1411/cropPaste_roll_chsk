# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import logging 
import torch


import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
# from .dino_cls_loss_cache_global import ProtoQueue
from dinov3.distributed import get_process_subgroup, get_subgroup_size
from dinov3.loss.ch_sk import CH_SK
from dinov3.loss.dtch_sk import DTCH_SK
logger = logging.getLogger("dinov3")


def lossfunc(t, s, temp):  # noqa: F811
    return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


class SinkhornKnoppTeacher(nn.Module):
    """
    NOTE: This is a module and not a function in the `iBOTPatchLoss` class
    This is because we want to torch.compile it, and torch.compil-ing a single
    function with the `@torch.compile` decorator is bad.
    It's better to `module.compile()` it, as we can control when we enable or
    disable compilation globally.
    """

    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * world_size # number of samples to assign
        B = n_masked_patches_tensor
        dist.all_reduce(B, group=get_process_subgroup())
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
    
class CH_SK_v1(nn.Module):
    def __init__(self, K, history_cache_size=3072):
        super().__init__()  
        self.K = K
        self.history_cache_size = history_cache_size
        self.register_buffer("history_Q", torch.empty(0))
        self.logits_temp_max = 30.0

    def _ensure_history_Q(self, Q_local):
        if self.history_Q is None or self.history_Q.numel() == 0:
            K, b_local = Q_local.shape
            sum_Q_local = torch.sum(Q_local, dim=1)
            if dist.is_initialized():
                dist.all_reduce(sum_Q_local, group=get_process_subgroup())
            self.history_Q = sum_Q_local * (self.history_cache_size / b_local ) / (get_subgroup_size() if dist.is_initialized() else 1)


            # self.history_Q = torch.sum(Q_local, dim=1) * (self.history_cache_size / b_local)
            # init_value = 5e8 * self.history_cache_size / 8e4  # 1000 iterations for 80k cache_size history_Q is 1e10

            # init_value = 5e8 * torch.exp(torch.tensor(self.logits_temp_max - 25)) \
            #  * self.history_cache_size / 8e4
            # self.history_Q = torch.full(
            #     (K,),
            #     init_value,
            #     dtype=Q_local.dtype,
            #     device=Q_local.device,
            # )

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
        if history_key in missing_keys:
            missing_keys.remove(history_key)
            
            
    def _update_history(self, Q_local):
        self._ensure_history_Q(Q_local)
        _, b_local = Q_local.shape
        self.history_Q = (self.history_cache_size - b_local)/self.history_cache_size * self.history_Q + torch.sum(Q_local, dim=1)
    
    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_masked_patches_tensor=None, n_iterations=3, iteration=0, logger_freq=0):
        
        # teacher_output: [batch, prototypes]
        teacher_output = teacher_output.float()
        
        world_size = get_subgroup_size() if dist.is_initialized() else 1
        logits_temp = teacher_output / teacher_temp
        logits_temp_clamp = logits_temp.clamp(min=-10, max=self.logits_temp_max)

        # Q_batch = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper

        Q_batch = torch.exp(logits_temp_clamp).t()

        # ===============================
        # pre-SK prototype boosting
        # ===============================
        self._ensure_history_Q(Q_batch)
        
        with torch.no_grad():
            hist = self.history_Q  # [K], must already exist
            mean_hist = hist.mean()
            threshold = mean_hist / 500.0

            # 找 history 最小的 20 个 prototype
            _, low_idx = torch.topk(hist, k=20, largest=False)
            boost_idx = low_idx[hist[low_idx] < threshold]

            if boost_idx.numel() > 0:
                Q_batch[boost_idx, :] *= 5.0
    
        if logger_freq and iteration % logger_freq == 0:
            if boost_idx.numel() > 0:
                boost_idx_list = boost_idx.tolist()
                boost_hist_list = hist[boost_idx].tolist()
                logger.info(
                    f"[CHSK-BOOST][iter={iteration}] "
                    f"boost_idx={boost_idx_list} | "
                    f"hist={['%.3e' % v for v in boost_hist_list]} | "
                    f"mean=%.3e thr=%.3e"
                    % (mean_hist.item(), threshold.item())
                )
            else:
                logger.info(
                    f"[CHSK-BOOST][iter={iteration}] no boost | "
                    f"min_hist=%.3e mean=%.3e thr=%.3e"
                    % (hist.min().item(), mean_hist.item(), threshold.item())
                )
            logger.info(f"iteration {iteration}, logits_temp: max {logits_temp[-1].max().item():.3e}, min {logits_temp[-1].min().item():.3e}, mean {logits_temp[-1].mean().item():.3e}, (-1, :5){['%.3e' % v for v in logits_temp[-1, :5].tolist()]}")
            logger.info(f"iteration {iteration}, logits_temp_clamp: max {logits_temp_clamp[-1].max().item():.3e}, min {logits_temp_clamp[-1].min().item():.3e}, mean {logits_temp_clamp[-1].mean().item():.3e}, (-1, :5){['%.3e' % v for v in logits_temp_clamp[-1, :5].tolist()]}")
            logger.info(f"iteration {iteration}, "f"Q_batch_max: {Q_batch.t()[-1].max().item():.3e}, Q_batch_min: {Q_batch.t()[-1].min().item():.3e}, Q_batch_mean: {Q_batch.t()[-1].mean().item():.3e}, Q_batch[-1, :5]: {['%.3e' % v for v in Q_batch.t()[-1, :5].tolist()]}")
            logger.info(f"self.history_Q_max: {self.history_Q.max().item():.3e}, self.history_Q_min: {self.history_Q.min().item():.3e}, self.history_Q_mean: {self.history_Q.mean().item():.3e}, self.history_Q[:5]: {['%.3e' % v for v in self.history_Q[:5].tolist()]}")
        B_batch = Q_batch.shape[1]
        B = (B_batch + self.history_cache_size)  * world_size # number of samples to assign
        K = Q_batch.shape[0]  # how many prototypes
        Q = torch.cat([self.history_Q.unsqueeze(1), Q_batch], dim=1)
        
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            sum_of_columns = torch.sum(Q, dim=0, keepdim=True)
            sum_of_columns[0, 0] = sum_of_columns[0, 0] / self.history_cache_size
            Q /= sum_of_columns
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        Q_batch[boost_idx, :] /= 5.0
        self._update_history(Q_batch)
        return Q[:, -B_batch:].t()


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9, use_sk_cache=False, cahe_size=10000,
                 use_history=False, history_cache_size=20000):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, 1, patch_out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None
        
        # self.sinkhorn_knopp_teacher.compile()
        if use_history:
            logger.info(f"__init__ ibotloss, use history: history_cache_size: {history_cache_size}")
            self.sinkhorn_knopp_teacher = DTCH_SK(K=patch_out_dim, history_cache_size=history_cache_size)
        # if use_sk_cache:
        #     pass
            # self.sinkhorn_knopp_teacher = SinkhornKnoppTeacher_cache(K=patch_out_dim, queue_size=cahe_size)
        else:
            self.sinkhorn_knopp_teacher = SinkhornKnoppTeacher()
            # self.sinkhorn_knopp_teacher.compile()

    def init_weights(self) -> None:
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = lossfunc(t, s, self.student_temp)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
        iteration=0,
        logger_freq=0,
        logger_loss=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        
        # loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        if logger_freq > 0 and iteration % logger_freq == 0:
            loss_tag = f"[{logger_loss}] " if logger_loss else ""
            logger.info(
                f"{loss_tag}iteration {iteration}: "
                f"t_n_masked_patches={t.shape[0]}, s_n_masked_patches={s.shape[0]}, n_masked_patches={n_masked_patches}"
            )
            stuednt_softmax_end = F.softmax(s[-1].float() / self.student_temp, dim=-1)
            logger.info(
                f"{loss_tag}s[-1].mean() ={stuednt_softmax_end.mean()}, "
                f"max = {stuednt_softmax_end.max()}, min={stuednt_softmax_end.min()}"
            )
            logger.info(f"{loss_tag}t[-1].mean() ={t[-1].mean()}, max = {t[-1].max()}, min={t[-1].min()}")
            logger.info(
                f"{loss_tag}\n t[-1][:5] = {['%.3e' % v for v in t[-1][:5].tolist()]}, "
                f"\n stuednt_softmax_end[-1][:5] = {['%.3e' % v for v in stuednt_softmax_end[:5].tolist()]}"
            )

        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True, group=get_process_subgroup())

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = get_subgroup_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
