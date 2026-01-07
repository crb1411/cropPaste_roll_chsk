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

from dinov3.distributed import get_process_subgroup, get_subgroup_size

logger = logging.getLogger("dinov3")


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
        use_history=True,
        history_cache=4096,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        self.history_Q = None
        self.history_cache = history_cache
        self.use_history = use_history

    def init_weights(self) -> None:
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
    
    def _ensure_history_Q(self, Q_local):
        if self.history_Q is None:
            _, b_local = Q_local.shape
            self.history_Q = torch.sum(Q_local, dim=1) * (self.history_cache / b_local)
            self.history_Q = self.history_Q.to(Q_local.device)
            
        
    def _update_history(self, Q_local):
        self._ensure_history_Q(Q_local)
        _, b_local = Q_local.shape
        self.history_Q = (self.history_cache - b_local)/self.history_cache * self.history_Q + torch.sum(Q_local, dim=1)
    
    @torch.no_grad()
    def sinkhorn_knopp_teacher_history(
        self,
        teacher_output,
        teacher_temp,
        n_iterations=3,
        iteration=0,
        logger_freq=0,
        logger_loss=None,
    ):
        
        # teacher_output: [batch, prototypes]
        teacher_output = teacher_output.float()
        
        world_size = get_subgroup_size() if dist.is_initialized() else 1

        if logger_freq > 0 and iteration % logger_freq == 0:
            loss_tag = f"[{logger_loss}] " if logger_loss else ""
            t = teacher_output.float()
            head = t[-1, : min(5, t.shape[1])].tolist()
            logger.info(
                f"{loss_tag}iteration {iteration}: teacher_temp={teacher_temp}, "
                f"teacher_output mean={t.mean().item():.3e}, max={t.max().item():.3e}, "
                f"min={t.min().item():.3e}, head={['%.3e' % v for v in head]}"
            )
        
        Q_batch = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        self._ensure_history_Q(Q_batch)
        B_batch = Q_batch.shape[1]
        B = (B_batch + self.history_cache)  * world_size # number of samples to assign
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
            sum_of_columns[0, 0] = sum_of_columns[0, 0] / self.history_cache
            Q /= sum_of_columns
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        self._update_history(Q_batch)
        targets = Q[:, -B_batch:].t()
        if logger_freq > 0 and iteration % logger_freq == 0:
            loss_tag = f"[{logger_loss}] " if logger_loss else ""
            t = targets[-1]
            head = t[: min(5, t.numel())].tolist()
            logger.info(
                f"{loss_tag}targets mean={t.mean().item():.3e}, max={t.max().item():.3e}, "
                f"min={t.min().item():.3e}, head={['%.3e' % v for v in head]}"
            )
        return targets
    
    
    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self,
        teacher_output,
        teacher_temp,
        n_iterations=3,
        iteration=0,
        logger_freq=0,
        logger_loss=None,
    ):
        if self.use_history:
            return self.sinkhorn_knopp_teacher_history(
                teacher_output,
                teacher_temp,
                n_iterations,
                iteration=iteration,
                logger_freq=logger_freq,
                logger_loss=logger_loss,
            )
        # teacher_output: [batch, prototypes]
        teacher_output = teacher_output.float()
        world_size = get_subgroup_size() if dist.is_initialized() else 1

        if logger_freq > 0 and iteration % logger_freq == 0:
            loss_tag = f"[{logger_loss}] " if logger_loss else ""
            head = teacher_output[-1, : min(5, teacher_output.shape[1])].tolist()
            logger.info(
                f"{loss_tag}iteration {iteration}: teacher_temp={teacher_temp}, "
                f"teacher_output mean={teacher_output.mean().item():.3e}, "
                f"max={teacher_output.max().item():.3e}, min={teacher_output.min().item():.3e}, "
                f"head={['%.3e' % v for v in head]}"
            )
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
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
        targets = Q.t()
        if logger_freq > 0 and iteration % logger_freq == 0:
            loss_tag = f"[{logger_loss}] " if logger_loss else ""
            t = targets[-1]
            head = t[: min(5, t.numel())].tolist()
            logger.info(
                f"{loss_tag}targets mean={t.mean().item():.3e}, max={t.max().item():.3e}, "
                f"min={t.min().item():.3e}, head={['%.3e' % v for v in head]}"
            )
        return targets

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_logits: [student crops, batch, prototypes]
        teacher_probs:  [teacher crops, batch, prototypes] must sum to 1 over the last dim

        loss = 0
        count = 0
        for each sample `b` in the batch:
            for each student crop `s` of this sample:
                for each teacher crop `t` of this sample:
                    if ignore_diagonal and s == t:
                        continue
                    loss += cross_entropy(softmax(student_logits[s, b] / student_temp), teacher_probs[t, b])
                    count += 1
        return loss / count
        """
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        if not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            return loss.sum() / (B * student_crops * teacher_crops - B * min_st)

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True, group=get_process_subgroup())

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = get_subgroup_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


if __name__ == "__main__":
    dinols = DINOLoss(
        out_dim=16384,
    )
    for i in range(100):
        dinols.sinkhorn_knopp_teacher(torch.randn(10, 16384), 0.07)
    
    pass
    
