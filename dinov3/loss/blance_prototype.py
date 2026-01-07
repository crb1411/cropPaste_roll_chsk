import logging
import torch

import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import torch.distributed as dist
from dinov3.distributed import get_process_subgroup, get_subgroup_size

logger = logging.getLogger("dinov3")



class PrototypeBalancer:
    def __init__(self, K, momentum=0.9, alpha=1.0, soft_T=10.0, device=None, update_type="blance_all"):
        """
        K: number of prototypes (e.g., 16384)
        momentum: EMA strength
        alpha: balancing strength
        """
        self.K = K
        self.momentum = momentum
        self.alpha = alpha
        self.device = device
        # global EMA marginal (概率分布，初始均匀)
        self.m_ema = None
        self.offset = None
        self.eps = 1e-12
        self.soft_T = 1.0 / math.sqrt(K) * soft_T
        self.update_type = update_type if update_type is not None else "blance_all"
        logger.info(
            f"sk: K={K}, momentum={momentum}, alpha={alpha}, soft_T={self.soft_T}, update_type={self.update_type}"
        )
    def _ensure_data(self, device=None):
        if device is None:
            device = self.device
        if self.m_ema is None:
            self.m_ema = torch.ones(self.K) / self.K
            self.m_ema = self.m_ema.to(device)
        if self.offset is None:
            self.offset = self.m_ema[0]  # = 1/K
            self.offset = self.offset.to(device)

    def _distributed_marginal(self, p_prob: torch.Tensor) -> torch.Tensor:
        """
        计算跨所有 rank 的全局 m_batch:
          m_batch[k] = 所有样本在第 k 维上的平均概率
        支持不等 batch_size（最后一个 batch 的情况）
        """
        B, K = p_prob.shape
        device = p_prob.device
        dtype = p_prob.dtype

        # 本 rank 的和、样本数
        local_sum = p_prob.sum(dim=0)                  # [K]
        local_B = torch.tensor([B], device=device, dtype=dtype)  # [1]

        if dist.is_available() and dist.is_initialized():
            # all_reduce 求所有 rank 上的和
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=get_process_subgroup())
            dist.all_reduce(local_B, op=dist.ReduceOp.SUM, group=get_process_subgroup())

        global_sum = local_sum                         # [K]
        global_B = local_B.item()                      # 标量

        m_batch = global_sum / (global_B + self.eps)   # [K]
        return m_batch

    def _sync_m_ema(self):
        """
        在所有 rank 之间同步 m_ema（从 rank 0 广播）
        """
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.m_ema, src=0, group=get_process_subgroup())

    def update(self, p: torch.Tensor, update_type: str = None):
        """
        p: [B, K], already normalized L2 (这里当作 logits)
        return:
            p_new:        [B, K]  均衡后的概率
            m_batch:      [K]     全局 batch marginal
            m_ema:        [K]     EMA usage（所有 rank 一致）
            p_prob:       [B, K]  原始 softmax(prob)
            p_balanced:   [B, K]  均衡前的“中间量”
            balance_strength: [K] m_ema - 1/K
        """
        device = p.device
        B, K = p.shape
        if update_type is None:
            update_type = self.update_type
        assert K == self.K, f"Expect K={self.K}, got {K}"
        assert update_type in ["blance_index", "blance_all"]
        p = p.float()
        # 0. 特征当 logits，先转成概率，用于统计 usage
        p_norm = nn.functional.normalize(p, dim=-1, p=2, eps=1e-12)
        p_prob = F.softmax(p_norm / self.soft_T, dim=-1)  # [B, K]

        # 保证 m_ema / offset 在正确 device/dtype
        self._ensure_data(device)
        self.m_ema = self.m_ema.to(device=device, dtype=p.dtype)
        self.offset = self.offset.to(device=device, dtype=p.dtype)

        # 1. 全局 batch marginal（跨所有 rank）
        #    如果没初始化 dist，就退化为本地 mean
        if dist.is_available() and dist.is_initialized():
            m_batch = self._distributed_marginal(p_prob.detach())  # [K]
        else:
            m_batch = p_prob.detach().mean(dim=0)                  # [K]

        # 2. EMA marginal（保持 sum=1）
        new_ema = self.momentum * self.m_ema + (1.0 - self.momentum) * m_batch
        self.m_ema = new_ema / (new_ema.sum() + self.eps)          # [K]

        # 2.5. 同步 m_ema 到所有 rank（保证每个 rank 用到的 m_ema 一致）
        self._sync_m_ema()
        # 再 cast 一下，防止 broadcast 改了 dtype/device
        self.m_ema = self.m_ema.to(device=device, dtype=p.dtype)

        # 3. balancing strength
        balance_strength = self.m_ema - self.offset  # [K]

        # 4. 做均衡
        if update_type == "blance_index":
            p_balanced_logits = p_prob - self.alpha * balance_strength.unsqueeze(0)  # [B, K]

            p_new = F.softmax(p_balanced_logits / self.soft_T * 10.0, dim=-1)

        else:
            assert update_type == "blance_all"
            if -balance_strength.min().item() * 10 > balance_strength.max().item():
                delta = self.alpha * balance_strength.min().item()
                p_balanced_logits = p_prob - delta
            else:
                delta = self.alpha * balance_strength.max().item()
                p_balanced_logits = p_prob + delta

            p_new = p_balanced_logits / (
                p_balanced_logits.sum(dim=-1, keepdim=True) + self.eps
            )

        return p_new, m_batch, self.m_ema, p_prob, p_balanced_logits, balance_strength
