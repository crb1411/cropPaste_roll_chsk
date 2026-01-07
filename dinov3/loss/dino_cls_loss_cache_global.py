import math
import logging 
import torch


import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import time

# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[2]))

from dinov3.distributed import get_process_subgroup, get_subgroup_size
from dinov3.loss.blance_prototype import PrototypeBalancer
from dinov3.loss.ch_sk import CH_SK
logger = logging.getLogger("dinov3")

class ProtoQueue:
    def __init__(self, K, max_len=3072, device=None, dtype=torch.float16):
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
    def add(self, logits_local):
        """ 
        logits_local: [B_local, K]
        """
        queue_device = self._resolve_device(logits_local.device)
        self._ensure_queue(queue_device)
        logits_local = logits_local.detach().to(queue_device, dtype=self.dtype)

        self.queue = torch.cat([self.queue, logits_local], dim=0)

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
        sk_cache=4096,
        device=None,
        use_blance_p=False,
        blance_momentum=0.9,
        blance_alpha=1.0,
        use_history=False,
        history_cache_size=20000,
        cfg=None
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        self.sk_cache = sk_cache
        self.use_sinkhorn_queue = use_sinkhorn_queue
        self.use_blance_p = use_blance_p
        self.use_history = use_history
        self.history_cache_size = history_cache_size
        if use_history:
            logger.info(f'INIT_INFOdino loss use CH_SK, cache_size: {history_cache_size}')
            self.sinkhorn_knopp_teacher_ch_sk = CH_SK(K=out_dim, history_cache_size=history_cache_size, cfg=cfg)

        elif use_sinkhorn_queue:
            self.queue = ProtoQueue(K=out_dim, max_len=sk_cache, device=device, dtype=torch.float32)
            self.print_init = True
            self.time_op = 1
            logger.info(f"Using Sinkhorn queue with {sk_cache} elements")
        elif use_blance_p:
            self.balancer = PrototypeBalancer(K=out_dim, momentum=blance_momentum, alpha=blance_alpha)
            self.print_init = True
            self.time_op = 1
        else:
            self.queue = None
            self.print_init = False
        


    def init_weights(self) -> None:
        self.center.zero_()

    def blance_prototype(self, teacher_output):
        if self.balancer is not None:
            (
                p_new, 
                m_batch, 
                m_ema, 
                p_prob, 
                p_balanced_logits, 
                balance_strength
             ) = self.balancer.update(teacher_output)
            if self.print_init:
                if self.time_op % 40 == 0:
                    logger.info(
                        "===== prototype balance debug =====\n"
                        f"p_raw              first={teacher_output[0, 0].item():.4e}, last={teacher_output[0, -1].item():.4e}\n"
                        f"p_prob             first={p_prob[0, 0].item():.4e}, last={p_prob[0, -1].item():.4e}\n"
                        f"p_new              first={p_new[0, 0].item():.4e}, last={p_new[0, -1].item():.4e}\n"
                        f"balance_strength   first={balance_strength[0].item():.4e}, last={balance_strength[-1].item():.4e}\n"
                        f"p_bal_logits       first={p_balanced_logits[0, 0].item():.4e}, last={p_balanced_logits[0, -1].item():.4e}\n"
                        f"m_ema              first={m_ema[:5].tolist()}, last={m_ema[-5:].tolist()}\n"
                        f"m_batch            first={m_batch[0].item():.4e}, last={m_batch[-1].item():.4e}"
                    )

                self.time_op += 1
            return p_new
        else:
            return teacher_output

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    # ---- 关键：Sinkhorn + Queue（全局版本，无 dist 运算）----
    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3, iteration=0, logger_freq=20):
        """
        teacher_output: shape [B_local, K]

        """
        if self.use_history: 
            return self.sinkhorn_knopp_teacher_ch_sk(teacher_output, teacher_temp, n_iterations, iteration=iteration, logger_freq=logger_freq)
        # 当前 rank 的 batch 大小
        B_local = teacher_output.shape[0]


        teacher_output_local = teacher_output
        if self.queue is not None:
            self.queue.add(teacher_output_local)   
            teacher_output_all_local = self.queue.get()         # [Q, K] or None
        else:
            teacher_output_all_local = teacher_output_local

        teacher_output_all_local = teacher_output_all_local.float()
        
        # Q: [K, B_eff]
        Q = torch.exp(teacher_output_all_local / teacher_temp).t()
        K, B_eff_local = Q.shape
        B_local_tensor = torch.tensor([B_eff_local], device=Q.device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(B_local_tensor, op=dist.ReduceOp.SUM, group=get_process_subgroup())
        B = int(B_local_tensor.item())   
        # make matrix sum to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        if self.print_init and self.time_op<=20:
            if self.time_op>=20:
                logger.info(f"B_local: {B_local}")
                logger.info(f"B_global: {B}")
                logger.info(f"queue_size: {self.sk_cache}")
                self.print_init = False
            self.time_op += 1

        for _ in range(n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)   # [K,1]
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, op=dist.ReduceOp.SUM, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)           
            Q /= B

        Q *= B   
        Q_all = Q.t()       # [B_eff_global, K]

        Q_cur_local = Q_all[-B_local:]   # [B_local, K]


        return Q_cur_local  
    
    
    def forward(
        self,
        student_logits,
        teacher_probs,
        ignore_diagonal=False,
        iteration=0,
        logger_freq=0,
        logger_loss=None,
    ):
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
        if logger_freq > 0 and iteration % logger_freq == 0:
            loss_tag = f"[{logger_loss}] " if logger_loss else ""
            logger.info(f"{loss_tag}iteration {iteration}: student_crops={student_crops}, teacher_crops={teacher_crops}")
            stuednt_softmax_end = F.softmax(student_logits[-1, -1].float() / self.student_temp, dim=-1)
            logger.info(
                f"{loss_tag}stuednt_softmax[-1][-1].mean() ={stuednt_softmax_end.mean()}, "
                f"max = {stuednt_softmax_end.max()}, min={stuednt_softmax_end.min()}"
            )
            logger.info(
                f"{loss_tag}teacher_probs[-1][-1].mean()   ={teacher_probs[-1][-1].mean()}, "
                f"max = {teacher_probs[-1][-1].max()}, min={teacher_probs[-1][-1].min()}"
            )
            logger.info(
                f"{loss_tag}\nteacher_probs[-1][-1][:5] = "
                f"{['%.3e' % v for v in teacher_probs[-1][-1][:5].tolist()]}, "
                f"\nstudent_logits[-1][-1][:5] = {['%.3e' % v for v in stuednt_softmax_end[:5].tolist()]}"
            )

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
    # B = 4096
    # K = 65536

    # teacher_output = torch.distributions.pareto.Pareto(1.0, 1.5).sample((B, K)).cuda()
    # teacher_output = torch.log(teacher_output + 1e-9) * 50  # 极端拉大差距
    
    # loss = DINOLoss(K, 0.1, 0.9, use_sinkhorn_queue=True)
    # Q = loss.sinkhorn_knopp_teacher(teacher_output, 10, 3)
    torch.cuda.set_device(0)

    B = 768
    K = 65536
    N = 50        # 重复 Sinkhorn 次数
    iters = 3     # 内部迭代次数
    
    # ————把 Q 的 CPU buffer 提前准备好（避免计时受 CPU 随机采样干扰）————
    print("Pre-generating CPU buffers...")
    Q_cpu_list = []
    for _ in range(N):
        q = torch.distributions.pareto.Pareto(1.0, 1.5).sample((B, K))
        q = torch.log(q + 1e-9) * 50
        Q_cpu_list.append(q)
    print("Done.")

    # CUDA warmup
    x = torch.randn(8).cuda()
    torch.cuda.synchronize()

    total_time = 0.0
    print(f"Running Sinkhorn {N} times, B={B}, K={K}, iters={iters}")
    loss = DINOLoss_skcache(K, 0.1, 0.9, use_sinkhorn_queue=True, sk_cache=768*6, device="cuda")
    for i in range(N):
        # —— 每轮都复制一个新的 Q 到 GPU —— #
        torch.cuda.synchronize()
        Q = Q_cpu_list[i].cuda()  # NEW Q for each iteration

        t0 = time.time()
        
        Q2 = loss.sinkhorn_knopp_teacher(Q, 10)
        torch.cuda.synchronize()

        elapsed = time.time() - t0
        total_time += elapsed

        print(f"[{i+1}/{N}] time = {elapsed:.4f}s")

    print("----------------------------------------------------")
    print(f"Total time: {total_time:.4f} s")
    print(f"Average per run: {total_time / N:.4f} s")
