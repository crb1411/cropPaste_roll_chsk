
from typing import Any, Optional

import torch


def _normalize_device_type(device_type: Optional[str]) -> Optional[str]:
    if device_type is None:
        return None
    device_type = str(device_type).strip().lower()
    if device_type.startswith("cuda") or device_type in ("gpu", "cuda"):
        return "cuda"
    if device_type.startswith("npu") or device_type in ("npu",):
        return "npu"
    if device_type.startswith("cpu") or device_type in ("cpu",):
        return "cpu"
    if device_type in ("auto", ""):
        return None
    return device_type


def has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def has_cuda() -> bool:
    return torch.cuda.is_available()


def get_config_device_type(cfg: Any | None) -> Optional[str]:
    if cfg is None:
        return None
    model_cfg = getattr(cfg, "MODEL", None)
    if model_cfg is not None and "DEVICE" in model_cfg:
        return _normalize_device_type(model_cfg.DEVICE)
    if "DEVICE" in cfg:
        return _normalize_device_type(cfg.DEVICE)
    return None


def get_available_device_type(prefer: Optional[str] = None) -> str:
    prefer = _normalize_device_type(prefer)
    if prefer == "npu" and has_npu():
        return "npu"
    if prefer == "cuda" and has_cuda():
        return "cuda"
    if prefer == "cpu":
        return "cpu"
    if has_npu():
        return "npu"
    if has_cuda():
        return "cuda"
    return "cpu"


def resolve_device_type(cfg: Any | None = None, override: Optional[str] = None) -> str:
    requested = _normalize_device_type(override) or get_config_device_type(cfg)
    if requested is None:
        return get_available_device_type()
    if requested == "npu":
        if not has_npu():
            raise RuntimeError("Requested device type 'npu' but torch.npu is not available.")
        return "npu"
    if requested == "cuda":
        if not has_cuda():
            raise RuntimeError("Requested device type 'cuda' but torch.cuda is not available.")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device type: {requested}")


def get_device(device_type: Optional[str] = None) -> torch.device:
    if device_type is None:
        device_type = get_available_device_type()
    return torch.device(device_type)


def get_device_module(device_type: Optional[str]):
    if device_type == "cuda":
        return torch.cuda
    if device_type == "npu":
        return torch.npu if hasattr(torch, "npu") else None
    return None


def set_device(device_type: Optional[str], index: int) -> None:
    module = get_device_module(device_type)
    if module is None:
        return
    module.set_device(index)


def synchronize(device_type: Optional[str] = None) -> None:
    if device_type is None:
        device_type = get_available_device_type()
    module = get_device_module(device_type)
    if module is None:
        return
    if device_type == "cuda" and not module.is_available():
        return
    if device_type == "npu" and not module.is_available():
        return
    module.synchronize()


def get_memory_stats(device_type: Optional[str] = None):
    if device_type is None:
        device_type = get_available_device_type()
    module = get_device_module(device_type)
    if module is None:
        return None
    if device_type == "cuda" and not module.is_available():
        return None
    if device_type == "npu" and not module.is_available():
        return None
    return {
        "current": module.memory_allocated(),
        "max": module.max_memory_allocated(),
    }
