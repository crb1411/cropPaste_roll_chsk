import torch

from functools import partial
import logging
import sys
sys.path.append('/mnt/local09/train/crb/npu_adp/dinov3')

from dinov3.configs import setup_config, setup_job
from dinov3.data import (
    DataAugmentationDINO_Wsi,
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    _make_sampler,
    make_dataset,
    CombinedDataLoader,
)
from dinov3.new_train.data.svs_h5.new_h5_svs_dataset import WsiPatchDataset, make_data_loader_wsi
from dinov3.new_train.data.svs_h5.svs_samplers import SamplerType_WSI
from dinov3.new_train.data.h5_file.h5_dataset import H5Dataset, H5Dataset_NoCache
from dinov3.new_train.data.multi_dataset import CombinedDataset, CombinedSampler, DatasetType

from dinov3.new_train.utils.log_create import creat_subdir
logger = logging.getLogger("dinov3")
def get_augmention(cfg):
    return DataAugmentationDINO_Wsi(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
        gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
        local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
        share_color_jitter=cfg.crops.share_color_jitter,
        horizontal_flips=cfg.crops.horizontal_flips,
        mean=cfg.crops.rgb_mean,
        std=cfg.crops.rgb_std,
    )




def build_dataset_from_cfg_wsi(
    cfg,
    dataset_type=DatasetType.SVS_DATE, # 0 svs 1: h5
):
    # Collate function
    if dataset_type == DatasetType.SVS_DATE:
        dataset = WsiPatchDataset(
            tensor_transform=get_augmention(cfg),
            patch_npy='/mnt/local09/train/crb/data/svs_data_v1/index.npy',
            path_txt='/mnt/local09/train/crb/data/svs_data_v1/path.txt',
            fix_size=224,
            return_dic=False
        )
    elif dataset_type==DatasetType.H5_DATE:
        dataset = H5Dataset_NoCache(
            # index_path ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index.npy',          # npy 索引 (structured npy: h5_id, patch_id)
            # files_txt ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index_files.txt',          # 保存 h5 文件路径
            index_path='/mnt/local09/train/crb/data/h5_data_local10/h5_patch_index.npy',
            files_txt='/mnt/local09/train/crb/data/h5_data_local10/h5_patch_index_files.txt',
            dataset_key = "patches", 
            transform=get_augmention(cfg), 
            max_open = 32,
            subdata_advance = None,
        )
    elif dataset_type==DatasetType.IMGNET_DATE:
        dataset_path = 'ImageNet:split=TRAIN:root=/mnt/local09/train/crb/env_init/data/dataset_imagenet/tiny-imagenet:extra=/mnt/local09/train/crb/env_init/data/dataset_imagenet/tiny-imagenet_extra'
        dataset = make_dataset(
            dataset_str=dataset_path,
            transform=get_augmention(cfg),
            target_transform=None,
        )
    return dataset

def get_collate(cfg):
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    local_batch_size = None  # will default to the standard local batch size matching the data batch size
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=local_batch_size,
    )
    return collate_fn
def build_CombinedDataset_loader(cfg, start_iter=0):
    
    dataset_svs = build_dataset_from_cfg_wsi(cfg, DatasetType.SVS_DATE)
    dataset_h5 = build_dataset_from_cfg_wsi(cfg, DatasetType.H5_DATE)
    dataset_img = build_dataset_from_cfg_wsi(cfg, DatasetType.IMGNET_DATE)
    datasets = [dataset_svs, dataset_h5, dataset_img]
    samplers = []
    ratios=[0.2, 0.7, 0.1]
    for ds, ratio in zip(datasets, ratios):
        sampler_type = SamplerType.SHARDED_INFINITE_NEW
        sampler = _make_sampler(
            dataset=ds,
            type=sampler_type,
            shuffle=True,
            seed=87,
            advance=int(start_iter*cfg.train.batch_size_per_gpu*ratio),
        )
        samplers.append(sampler)
    dataset_combined = CombinedDataset(dataset_list=datasets)
    sampler_combined = CombinedSampler(
        dataset_samplers = samplers,
        ratios=ratios
    )
    collate_fn = get_collate(cfg)
    loader_combined = torch.utils.data.DataLoader(
        dataset_combined,
        sampler=sampler_combined,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        timeout=0,
        multiprocessing_context="spawn",
        
    )
    return loader_combined



      
    
def test_data_loader(loader, feq=10, max=1000):
    from tqdm import tqdm
    i = 0
    for data in loader:
        ty = type(data)
        if i%feq == 0:
            logger.info(f'index: {i}, batch_size: {loader.batch_size}, data_size:{data["collated_global_crops"].shape}')
        i+=1
        if i>= max:
            break
import torch.multiprocessing as mp
import os  
os.environ.setdefault("OMP_NUM_THREADS", "1")
    
if mp.get_start_method(allow_none=True) not in ("spawn", "forkserver"):
    mp.set_start_method("spawn", force=True)
if __name__ == '__main__':
    log_dir_base = '/mnt/local09/train/crb/train_out/train_svs_mul_npu/logs_test'
    log_dir = creat_subdir(
            base_dir=log_dir_base,
            prefix='test_loader',
            time=True
        )
    from dinov3.new_train.train.train_mul import get_args_parser
    args = get_args_parser().parse_args()
    args.output_dir = log_dir
    setup_job(output_dir=args.output_dir, seed=args.seed)
    cfg = setup_config(args, strict_cfg=False)
    logger.info(cfg)
    loader_mul = build_CombinedDataset_loader(cfg, start_iter=0)
    test_data_loader(loader=loader_mul)