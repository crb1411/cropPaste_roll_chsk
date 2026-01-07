import math

import torch


import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

# import sys
# sys.path.append('/data/work/git_proj/dinov3')

from dinov3.distributed import get_process_subgroup, get_subgroup_size



class ProtoQueue:
    def __init__(self, K, max_len=4096, device=None, dtype=torch.float16):
        """
        这里 queue 只保存“传进来的 logits”，
        不做 all_gather —— all_gather 放在 sinkhorn 里统一做。
        """
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.K = K

        self.queue = None

    def _resolve_device(self, ref_device):
        if self.device is None:
            return ref_device
        return torch.device(self.device)

    def _ensure_queue(self, device):
        if self.queue is None or self.queue.device != device:
            self.queue = torch.empty((0, self.K), device=device, dtype=self.dtype)

    @torch.no_grad()
    def add(self, logits_global):
        """
        logits_global: [B_global, K]，已经是“全局”的（由外面 all_gather 得到）。
        """
        queue_device = self._resolve_device(logits_global.device)
        self._ensure_queue(queue_device)
        logits_global = logits_global.detach().to(queue_device, dtype=self.dtype)

        self.queue = torch.cat([self.queue, logits_global], dim=0)

        if self.queue.shape[0] > self.max_len:
            self.queue = self.queue[-self.max_len:]

    def get(self):
        if self.queue is None or self.queue.shape[0] == 0:
            return None
        return self.queue          # [Q, K] on GPU


class DINOLoss_skcache(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
        use_sinkhorn_queue=False,
        sk_cache_size=4096,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        if use_sinkhorn_queue:
            self.queue = ProtoQueue(K=out_dim, max_len=sk_cache_size, device=None, dtype=torch.float32)
        else:
            self.queue = None


    def init_weights(self) -> None:
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    # ---- 关键：Sinkhorn + Queue（全局版本，无 dist 运算）----
    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        """
        teacher_output: 当前 rank 的 logits，shape [B_local_raw, K]

        流程：
          1. all_gather → teacher_output_global [B_global, K]
          2. queue.add(teacher_output_global) 仅缓存原始 logits
          3. teacher_output_all_global = 当前 global + queue_cache
          4. 在所有 rank 上对 teacher_output_all_global 做同一份 Sinkhorn
          5. 从全局结果中切出本 rank 对应的 B_local_raw 段
        """

        # 当前 rank 的 batch 大小
        B_local_raw = teacher_output.shape[0]

        # ---- 1) all_gather 当前 batch 到全局 ----
        if dist.is_initialized():
            world_size = get_subgroup_size()
            rank = dist.get_rank()

            gather_list = [torch.zeros_like(teacher_output) for _ in range(world_size)]
            dist.all_gather(gather_list, teacher_output)
            teacher_output_global = torch.cat(gather_list, dim=0)   # [B_global, K]
        else:
            world_size = 1
            rank = 0
            teacher_output_global = teacher_output

        B_global = teacher_output_global.shape[0]

        # ---- 2) queue: 缓存全局 logits ----
        if self.queue is not None:
            self.queue.add(teacher_output_global)   # 只存原始 logits
            teacher_output_all_global = self.queue.get()         # [Q, K] or None
        else:
            teacher_output_all_global = teacher_output_global

        # ---- 3) 在所有 rank 上使用同一份 teacher_output_all_global 做“普通 Sinkhorn” ----

        teacher_output_all_global = teacher_output_all_global.float()
        # Q: [K, B_eff_global]
        Q = torch.exp(teacher_output_all_global / teacher_temp).t()
        K, B_eff_global = Q.shape  # 这里 B_eff_global 已经是“全局有效样本数”，不再区分 local / global

        # make matrix sum to 1
        Q /= torch.sum(Q)

        for _ in range(n_iterations):
            # row-normalize: 每个 prototype 总量 = 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)   # [K,1]
            Q /= sum_of_rows
            Q /= K

            # column-normalize: 每个样本总量 = 1/B_eff_global
            Q /= torch.sum(Q, dim=0, keepdim=True)            # [1,B_eff]
            Q /= B_eff_global

        Q *= B_eff_global   # 让每列和 = 1
        Q_all = Q.t()       # [B_eff_global, K]

        # ---- 4) 从全局结果中取回“当前这一 global batch 的部分”，再切出本 rank 对应的段 ----
        Q_cur_global = Q_all[-B_global:]   # [B_global, K]

        # 再切本 rank 对应的 B_local_raw 那一段
        start = rank * B_local_raw
        end = start + B_local_raw
        Q_cur_local = Q_cur_global[start:end]   # [B_local_raw, K]

        return Q_cur_local  # 返回：与输入 teacher_output 对应的那部分 assignment
    
    
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

if __name__ == '__main__':
    
    # B = 128
    # K = 65536

    # teacher_output = torch.randn(B, K, device="cuda") * 0.1  # small noise
    # teacher_output[:, :1] += 3                             # prototype 0 占绝对优势
    B = 4096
    K = 65536

    teacher_output = torch.distributions.pareto.Pareto(1.0, 1.5).sample((B, K)).cuda()
    teacher_output = torch.log(teacher_output + 1e-9) * 50  # 极端拉大差距
    
    loss = DINOLoss(K, 0.1, 0.9, use_sinkhorn_queue=True)
    Q = loss.sinkhorn_knopp_teacher(teacher_output, 10, 3)
    print('\nraw_Q:')
    print(teacher_output)
    print(Q.shape)
    print("\nSinkhorn-Knopp balanced assignment Q:")
    print(Q)

    print("\nColumn sums (should be 1):")
    print(Q.sum(dim=1))

    print("\nRow sums of Q^T (prototype usage, should be ~uniform):")
    print(Q.t().sum(dim=1))
    pass
