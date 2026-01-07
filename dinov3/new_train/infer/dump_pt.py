import argparse
import copy
import gc
import math
import os


import torch

 
import torch.distributed
from torch.distributed._tensor import DTensor

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    load_checkpoint,
)
from dinov3.configs import setup_config, setup_job


from dinov3.train.ssl_meta_arch import SSLMetaArch

assert torch.__version__ >= (2, 1)
torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
torch.backends.cudnn.benchmark = False  # True


def load_fsdp(model, checkpoint_dir: str= ""):
    assert os.path.isdir(checkpoint_dir)
    process_subgroup = distributed.get_process_subgroup()
    start_iter = (
            load_checkpoint(
                checkpoint_dir,
                model=model,
                optimizer=None,
                strict_loading=False,
                process_group=process_subgroup,
            )
            + 1
        )
    return model, start_iter


def model_init(cfg):
    with torch.device("meta"):
        model = SSLMetaArch(cfg)
    model.prepare_for_distributed_training()
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="npu",
        ),
        recurse=True,
    )
    return model

def dump_pt(model, output_dir: str):
    new_state_dict = model.model_ema.state_dict()
    student_state_dict = model.student.state_dict()
    for k, tensor in list(new_state_dict.items()):
        if isinstance(tensor, DTensor):
            new_state_dict[k] = tensor.full_tensor()
    if not distributed.is_subgroup_main_process():
        return
    # save teacher checkpoint
    ckpt_path_teacher = output_dir + "teacher_checkpoint.pth"
    torch.save({"teacher": new_state_dict}, ckpt_path_teacher)
    ckpt_path_student = output_dir + "student_checkpoint.pth"
    torch.save({"student": student_state_dict}, ckpt_path_student)
    print(f"saved  checkpoint to {output_dir}")

if __name__ == "__main__":
    
    checkpoint_dir = "/mnt/data/train/crb/train_out/train_svs/log_20251013_1641/ckpt/93749"
    output_dir = "/mnt/data/train/crb/train_out/train_svs/log_20251013_1641/ckpt/93749"
    assert checkpoint_dir is not None and output_dir is not None
    
    try:
        from dinov3.train.train_svs import get_args_parser
        args = get_args_parser().parse_args()
    except:
        raise ImportError
    setup_job(output_dir=None, seed=0)
    cfg = setup_config(args, strict_cfg=False)
    print(cfg)

    
    model = model_init(cfg)
    model, start_iter = load_fsdp(model, checkpoint_dir=checkpoint_dir)
    print(f"loaded model from {checkpoint_dir}, start_iter={start_iter}")
    
    dump_pt(model, output_dir=output_dir)