# utils/device.py
from __future__ import annotations

import torch
from typing import Literal


DeviceType = Literal["cpu", "cuda", "npu", "mps", "xpu"]


def get_device(
    *,
    cpu: bool = False,
) -> torch.device:
    """
    Device auto-selection (Transformers / LLaMA-Factory style)

    Priority:
      1. cpu=True
      2. auto backend order
      3. cpu fallback
    """

    if cpu:
        return torch.device("cpu")

    # --- auto order (Transformers-like) ---
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")


def set_device_index(local_rank: int | None = None) -> None:
    """
    Set device index for multi-GPU training
    """
    device = get_device()
    if device.type == "cuda" and local_rank is not None:
        torch.cuda.set_device(local_rank)
    if device.type == "npu" and local_rank is not None:
        torch.npu.set_device(local_rank)
    if device.type == "xpu" and local_rank is not None:
        torch.xpu.set_device(local_rank)
    
    
def synchronize(device_type=torch.device('cuda')):
    if device_type.type == "cuda":
        torch.cuda.synchronize()
    elif device_type.type == "npu":
        torch.npu.synchronize()
    elif device_type.type == "xpu":
        torch.xpu.synchronize()
        