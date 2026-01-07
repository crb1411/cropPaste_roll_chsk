import os
import torch

 
from torchvision.utils import save_image  # 也可用 to_pil_image 再 .save

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


WSI_MEAN = (0.8116132616996765, 0.6650876998901367, 0.8012862205505371)
WSI_STD = (0.09951633960008621, 0.1301795095205307, 0.07808226346969604)

def _denorm(x: torch.Tensor, mean, std) -> torch.Tensor:
    """
    x: CHW float tensor (normalized)
    return: CHW float tensor in [0,1]
    """
    if x.dim() != 3 or x.size(0) not in (1, 3):
        raise ValueError(f"expect CHW tensor, got {tuple(x.shape)}")
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std,  dtype=x.dtype, device=x.device).view(-1, 1, 1)
    y = (x * s + m).clamp(0, 1)
    return y

def save_index(
    img: dict,
    out_dir: str = "aug_vis",
    prefix: str = "idx1",
    norm: str | tuple = "wsi",   # "imagenet" | "wsi" | (mean, std)
    max_locals: int = 8,
    save_teacher: bool = True
):
    """
    img: 就是 dataset[1] 的返回 dict
    """
    os.makedirs(out_dir, exist_ok=True)

    if norm == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif norm == "wsi":
        mean, std = WSI_MEAN, WSI_STD
    elif isinstance(norm, (list, tuple)) and len(norm) == 2:
        mean, std = norm
    else:
        raise ValueError(f"unknown norm spec: {norm}")

    def _save_one(t: torch.Tensor, path: str):
        # t: CHW normalized tensor
        t = t.detach().cpu()
        t_denorm = _denorm(t, mean, std)             # CHW, [0,1]
        save_image(t_denorm, path)                   # 接受 BCHW/CHW，范围[0,1]

    # 1) global crops (student)
    for i, t in enumerate(img.get("global_crops", [])):
        _save_one(t, os.path.join(out_dir, f"{prefix}_global_{i+1}.png"))

    # 2) local crops（只保存前 max_locals 张，避免太多）
    for i, t in enumerate(img.get("local_crops", [])[:max_locals]):
        _save_one(t, os.path.join(out_dir, f"{prefix}_local_{i+1}.png"))

    # 3) teacher crops（通常与 global_crops 相同，这里可选保存）
    if save_teacher:
        for i, t in enumerate(img.get("global_crops_teacher", [])):
            _save_one(t, os.path.join(out_dir, f"{prefix}_teacher_{i+1}.png"))

    if 'raw_image' in img:
        _save_one(img['raw_image'], os.path.join(out_dir, f"{prefix}_raw_img.png"))

    return out_dir
