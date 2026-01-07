import os
import argparse
import torch
# 


from collections import OrderedDict
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))


    
def get_cfg():
    try:
        from dinov3.configs import get_default_config
    except ImportError:
        print("Please install DINOv3 to use this function.")
        exit(1)
    default_cfg = get_default_config()
    return default_cfg

def get_model(cfg=None):
    if cfg is None:
        cfg = get_cfg()
    try:
        from dinov3.models.vision_transformer import DinoVisionTransformer
        from dinov3.models import build_model_from_cfg
    except ImportError:
        print("Please install DINOv3 to use this function.")
        exit(1)
    
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    return model


def get_all_model_state_dict(state_dict: OrderedDict) -> OrderedDict:
    state_dict = state_dict['teacher']
    backbone, ibot_head, dino_head = OrderedDict(), OrderedDict(), OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone'):
            backbone[k.replace('backbone.', '')] = v
        elif k.startswith('ibot_head'):
            ibot_head[k.replace('ibot_head.', '')] = v
        elif k.startswith('dino_head'): 
            dino_head[k.replace('dino_head.', '')] = v
        else:
            print(f'Ignore key: {k}')
    
    return backbone, ibot_head, dino_head
    

def get_model_from_pt(pt_path: str, device: str='cpu') -> torch.nn.Module:
    cfg = get_cfg()
    model = get_model(cfg)
    model = model.to_empty(device=device)
    state_dict = torch.load(pt_path, map_location='cpu')
    model.load_state_dict(state_dict=state_dict, strict=False)
    return model
    

def generate_pt(source, output_pt_dir):

    state_dict = torch.load(source, map_location='cpu')
    backbone, ibot_head, dino_head = get_all_model_state_dict(state_dict)
    os.makedirs(output_pt_dir, exist_ok=True)
    torch.save(backbone, os.path.join(output_pt_dir, 'backbone.pt'))
    torch.save(ibot_head, os.path.join(output_pt_dir, 'ibot_head.pt'))
    torch.save(dino_head, os.path.join(output_pt_dir, 'dino_head.pt'))
    print(f'backbone.pt, ibot_head.pt, dino_head.pt saved successfully {output_pt_dir}')
    
if __name__ == '__main__':
    source = '/mnt/local09/train/crb/train_out/train_1226/logs_out/log_20251226_2334/eval/training_845999/teacher_checkpoint.pth'
    output_pt_dir = '/mnt/crb/work/ckpts/ckpts_mul/dinov3_storage/845999_v2_4img'
    generate_pt(source, output_pt_dir)
    # args = parse_args()
    # model = get_model_from_pt(os.path.join(args.output_pt_dir, 'backbone.pt'))
    # print(model)
    
