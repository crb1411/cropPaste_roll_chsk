from __future__ import annotations
from pathlib import Path
import os
import shutil
import tempfile
from typing import Any
import logging


import torch

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

Stateful = Any  # 你原来的别名
logger = logging.getLogger('dinov3')


def _get_rank(pg: dist.ProcessGroup | None) -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank(pg)


def _get_src_rank(pg: dist.ProcessGroup | None) -> int:
    if pg is None:
        return 0
    return dist.get_global_rank(pg, 0)


def _to_cpu_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    """普通模型/DDP 的 state_dict 搬到 CPU（FSDP FULL 已经在 CPU 上就不会多搬）"""
    cpu_sd = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            cpu_sd[k] = v.detach().cpu()
        else:
            cpu_sd[k] = v
    return cpu_sd

from torch.distributed._tensor.api import DTensor

def unwrap_dtensor(v):
    """
    将一个可能是 DTensor 的对象转换成普通 torch.Tensor。
    """
    if isinstance(v, DTensor):
        # 取本地 shard（local_tensor)
        return v.to_local()

    return v


def convert_state_dict(d):
    """
    对整个 state_dict 遍历，将所有 DTensor 转成普通 Tensor。
    """
    new_d = {}
    for k, v in d.items():
        if isinstance(v, DTensor):
            new_d[k] = unwrap_dtensor(v)
        elif isinstance(v, torch.Tensor):
            new_d[k] = v
        else:
            # 如果出现 list / tuple / nested dict，也递归处理
            if isinstance(v, dict):
                new_d[k] = convert_state_dict(v)
            else:
                new_d[k] = v
    return new_d
def save_checkpoint(
    ckpt_dir: str | Path,  # output_dir/ckpt/199
    *,
    iteration: int | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    overwrite: bool = True,
    process_group: dist.ProcessGroup | None = None,
    **others: Stateful,
):
    """
    不依赖 DCP 的保存方式：
    - plain/DDP: 直接 state_dict -> CPU -> torch.save
    - FSDP/FSDP2: 使用 FULL_STATE_DICT 聚合到 CPU，再保存
    """
    ckpt_dir = Path(ckpt_dir)
    rank = _get_rank(process_group)
    src_rank = _get_src_rank(process_group)

    # ====== 1) 检查 & 处理已存在目录（保留你原来的语义） ======
    ckpt_dir_exists = [ckpt_dir.exists() if rank == 0 else None]
    dist.broadcast_object_list(ckpt_dir_exists, src=src_rank, group=process_group)
    ckpt_dir_exists = ckpt_dir_exists[0]

    if ckpt_dir_exists:
        if overwrite:
            if rank == 0:
                if ckpt_dir.is_dir():
                    shutil.rmtree(ckpt_dir)
                else:
                    ckpt_dir.unlink()
                logger.info(f"Deleted: {ckpt_dir}")
            dist.barrier(group=process_group)
        else:
            raise RuntimeError(f"Checkpoint already exists: {ckpt_dir}")

    # ====== 2) rank0 创建临时目录，并广播给其它 rank ======
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_list = [tempfile.mkdtemp(dir=ckpt_dir.parent, prefix=ckpt_dir.name) if rank == 0 else None]
    dist.broadcast_object_list(tmp_list, src=src_rank, group=process_group)
    ckpt_dir_tmp = Path(tmp_list[0])

    # ====== 3) 准备 model_state / optimizer_state / others_state ======
    # 3.1 model_state
    if isinstance(model, FSDP):
        # FSDP/FSDP2: FULL_STATE_DICT + CPU
        full_cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            model_state = model.state_dict()
    else:
        # plain / DDP
        # print(type(model))
        sd = model.state_dict()
        model_state = _to_cpu_state_dict(sd)
    
    # 3.2 optimizer_state
    optim_state = None
    if optimizer is not None:
        optim_state = optimizer.state_dict()  # 里面大部分张量在 CPU 上

    # 3.3 others_state（例如 scheduler, ema 等）
    others_state: dict[str, Any] = {}
    for name, obj in others.items():
        if obj is None:
            continue
        if hasattr(obj, "state_dict"):
            others_state[name] = obj.state_dict()
        else:
            # 不可 state_dict 的，就直接存原始对象（一般不建议，但兼容）
            others_state[name] = obj

    # ====== 4) rank0 写 checkpoint.pt，其他 rank 同步等待 ======
    ckpt_file = ckpt_dir_tmp / "checkpoint.pt"
    if rank == 0:
        payload = {
            "iteration": iteration,
            "model": model_state,
            "optimizer": optim_state,
            "others": others_state,
        }
        payload = convert_state_dict(payload)
        torch.save(payload, ckpt_file)
        logger.info(f"[rank0] Saved checkpoint to tmp {ckpt_file}")

    if dist.is_initialized():
        dist.barrier(group=process_group)

    # ====== 5) 原子重命名临时目录 ======
    if rank == 0:
        ckpt_dir_tmp.rename(ckpt_dir)
    if dist.is_initialized():
        dist.barrier(group=process_group)

    logger.info(f"Saved: {ckpt_dir}")



def load_checkpoint(
    ckpt_dir: str | Path,  # output_dir/ckpt/199
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict_loading: bool = True,
    process_group: dist.ProcessGroup | None = None,
    **others: Stateful,
) -> int | None:
    """
    FSDP-friendly 的加载方式（不使用 DCP）：
    - 支持 plain/DDP/FSDP/FSDP2
    - 支持 N 卡保存、M 卡加载（通过 FULL_STATE_DICT）
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_file = ckpt_dir / "checkpoint.pt"

    if not ckpt_file.exists():
        logger.warning(f"[load_checkpoint] no checkpoint found at {ckpt_file}")
        return None

    # ====== 1) 所有 rank 从同一路径 load 到 CPU ======
    payload = torch.load(ckpt_file, map_location="cpu")

    iteration = payload.get("iteration", 0)
    model_state = payload["model"]
    optim_state = payload.get("optimizer", None)
    others_state = payload.get("others", {})

    # ====== 2) 恢复模型参数 ======
    if isinstance(model, FSDP):
        full_cfg = FullStateDictConfig(
            rank0_only=False,
            offload_to_cpu=False,      # ?? 加载时必须为 False
        )
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            with torch.no_grad():
                missing, unexpected = model.load_state_dict(
                    model_state, strict=strict_loading
                )
    else:
        missing, unexpected = model.load_state_dict(
            model_state, strict=strict_loading
        )

    if missing or unexpected:
        logger.warning(
            f"[load_checkpoint] missing_keys={missing}, unexpected_keys={unexpected}"
        )

    # ====== 3) 恢复 optimizer ======
    if optimizer is not None and optim_state is not None:
        optimizer.load_state_dict(optim_state)

    # ====== 4) 恢复 others ======
    for name, obj in others.items():
        if obj is None:
            continue
        state = others_state.get(name, None)
        if state is None:
            continue
        if hasattr(obj, "load_state_dict"):
            obj.load_state_dict(state)
        else:
            # 没有 load_state_dict 的，就忽略或自定义
            pass

    if dist.is_initialized():
        dist.barrier(group=process_group)

    logger.info(f"Loaded: {ckpt_dir} (iteration={iteration})")
    return int(iteration)

if __name__ == "__main__":
    from dinov3.new_train.infer.load_model_from_fsdp import model_init
    model = model_init()
    info_path = '/mnt/local09/train/crb/npu_adp/npu_code_test/ckpt_info.txt'
    with open(info_path, 'w') as f:
        f.write(str(model))
    print('************')
    load_checkpoint(
        ckpt_dir='/mnt/local09/train/crb/npu_adp/ckpt/1260000/',
        model=model,
        optimizer=None,
        strict_loading=False,
    )
    with open(info_path, 'a') as f:
        f.write(str(model))
    # save_checkpoint(
    #     ckpt_dir='/mnt/local09/train/crb/npu_adp/ckpt',
    #     iteration=start_iter,
    #     model=model,
    #     optimizer=None,
    #     overwrite=True,
    # )
