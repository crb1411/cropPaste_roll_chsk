import torch


from functools import partial
import logging
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))
from dinov3.configs import setup_config, setup_job
from dinov3.data import (
    DataAugmentationDINO_Wsi,
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
    CombinedDataLoader,
)
from dinov3.new_train.data.svs_h5.new_h5_svs_dataset import WsiPatchDataset, make_data_loader_wsi
from dinov3.new_train.data.svs_h5.svs_samplers import SamplerType_WSI
from dinov3.new_train.data.h5_file.h5_dataset import H5Dataset

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
        # mean=cfg.crops.rgb_mean,
        # std=cfg.crops.rgb_std,
    )


def get_cfg():
    try:
        from dinov3.configs import get_default_config
    except ImportError:
        print("Please install DINOv3 to use this function.")
        exit(1)
    default_cfg = get_default_config()
    return default_cfg


def collate_ignore_none(batch):
    # batch 是 [(patch, None), (patch, None), ...]
    patches = [item[0] for item in batch]    # 只取 patch
    patches = torch.stack(patches, dim=0)    # -> [B, C, H, W]
    return patches

def build_data_loader_from_cfg(
    cfg,
    start_iter,
):
    # Collate function
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    local_batch_size = None  # will default to the standard local batch size matching the data batch size
    dataloader_batch_size_per_gpu = cfg.train.batch_size_per_gpu

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
    batch_size = dataloader_batch_size_per_gpu
    num_workers = cfg.train.num_workers
    dataset_path = 'ImageNet:split=TRAIN:root=/mnt/data/train/crb/env_init/data/dataset_imagenet/tiny-imagenet:extra=/mnt/data/train/crb/env_init/data/dataset_imagenet/tiny-imagenet_extra'
    dataset = make_dataset(
        dataset_str=dataset_path,
        transform=get_augmention(cfg),
        target_transform=None,
    )

    if isinstance(dataset, torch.utils.data.IterableDataset):
        sampler_type = SamplerType.INFINITE
    else:
        sampler_type = SamplerType.SHARDED_INFINITE if cfg.train.cache_dataset else SamplerType.INFINITE

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=cfg.train.seed + start_iter + 1,
        sampler_type=sampler_type,
        sampler_advance=start_iter * dataloader_batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return data_loader

def build_data_loader_from_cfg_wsi(
    cfg,
    start_iter=0,
    dataset_type=0, # 0 svs 1: h5
    use_wsi_sampler=False,
):
    # Collate function
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    local_batch_size = None  # will default to the standard local batch size matching the data batch size
    dataloader_batch_size_per_gpu = cfg.train.batch_size_per_gpu

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
    

    dataset = WsiPatchDataset(
        tensor_transform=None,
        patch_npy='/mnt/data/svs_train_data/h52npy/index.npy',
        path_txt='/mnt/data/svs_train_data/h52npy/path.txt',
        fix_size=224,
        # advance=82_000_000,
        return_dic=False
    ) if dataset_type==0 else H5Dataset(
        index_path ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index.npy',          # npy 索引 (structured npy: h5_id, patch_id)
        files_txt ='/mnt/local09/train/crb/data/h5_data_v1/h5_patch_index_files.txt',          # 保存 h5 文件路径
        dataset_key = "patches", 
        transform=None,
        max_open = 32,
        # subdata_advance = 619_000_000,
    )
    sampler_type = SamplerType.SHARDED_INFINITE_NEW
    data_loader = (
                    make_data_loader_wsi(
                dataset=dataset,
                batch_size=cfg.train.batch_size_per_gpu,
                num_workers=cfg.train.num_workers,
                shuffle=True,
                seed=0,  # TODO: Fix this -- cfg.train.seed
                sampler_type=sampler_type,
                sampler_advance=start_iter * cfg.train.batch_size_per_gpu,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
                drop_last=True,
                chunked=True,
                collate_fn=collate_ignore_none,
                chunk_size=32,
            ) if use_wsi_sampler else 
                    make_data_loader(
                dataset=dataset,
                batch_size=cfg.train.batch_size_per_gpu,
                num_workers=cfg.train.num_workers,
                shuffle=True,
                seed=cfg.train.seed + start_iter + 1,
                sampler_type=sampler_type,
                sampler_advance=start_iter * dataloader_batch_size_per_gpu,
                drop_last=True,
                collate_fn=collate_ignore_none,
            )
    )
    return data_loader
       
    
def test_data_loader(loader, feq=10, max=32):
    from tqdm import tqdm
    i = 0
    for data, idx in loader:
        ty = type(data)
        if i%feq == 0:
            logger.info(f'data_idx: {idx}, index: {i}, batch_size: {loader.batch_size}, data.shape: {data.shape}')
        i+=1
        if i>= max:
            break
        
def get_mean_std_from_dataloader(loader, max=200000, freq=50):
    """
    专门处理已经是 0~1 float32 的 image tensor: [B, C, H, W]
    在线统计 RGB mean/std
    """
    sum_total = torch.zeros(3, dtype=torch.float64)
    sum_sq_total = torch.zeros(3, dtype=torch.float64)
    count_total = 0

    for i, batch in enumerate(loader):

        # ------------------------
        # 自动从 batch 中提取 image tensor
        # ------------------------
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        elif isinstance(batch, dict):
            # 自动找出 CHW 图像数据
            data = None
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1] == 3:
                    data = v
                    break
            if data is None:
                continue
        else:
            data = batch

        # 要求 data 是 float32 0~1
        if not isinstance(data, torch.Tensor):
            continue
        if data.ndim != 4 or data.shape[1] != 3:
            continue

        # 移到 CPU（统计更快且不占显存）
        data = data.to(torch.float32).cpu()

        B, C, H, W = data.shape
        N = B * H * W  # 当前 batch 的像素数

        # reshape 成 (N, 3) 做 sum / sum of squares
        x = data.permute(0, 2, 3, 1).reshape(-1, 3)

        sum_total += x.sum(dim=0)
        sum_sq_total += (x * x).sum(dim=0)
        count_total += N

        if i % freq == 0:
            logger.info(f"[Iter {i}] pixels={count_total:,}")
            mean = sum_total / count_total
            var = (sum_sq_total / count_total) - mean * mean
            std = torch.sqrt(var)
            logger.info(f"mean={mean.tolist()}, std={std.tolist()}")

        if i >= max:
            break

    # ========== 计算 mean/std ==========
    mean = sum_total / count_total
    var = (sum_sq_total / count_total) - mean * mean
    std = torch.sqrt(var)

    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    log_dir_base = '/mnt/data/train/crb/train_out/train_test'
    log_dir = creat_subdir(
            base_dir=log_dir_base,
            prefix='test_loader',
            time=True
        )
    setup_job(output_dir=log_dir, distributed_enabled=False)
    cfg = get_cfg()
    cfg.log_dir = log_dir
    cfg.train.batch_size_per_gpu = 128
    cfg.train.num_workers = 20
    loader0 = build_data_loader_from_cfg_wsi(
        cfg=cfg,
        start_iter=0,
        dataset_type=0
    )
    logger.info(f'begin test loader 0 len:{len(loader0.dataset)}')
    # test_data_loader(loader0)
    loader1 = build_data_loader_from_cfg_wsi(
        cfg, 0, 1
    )
    logger.info(f'begin test loader 1 len:{len(loader1.dataset)}')
    # test_data_loader(loader1)
    loader2 = build_data_loader_from_cfg(
        cfg, 0
    )
    logger.info(f'begin test loader 2 len:{len(loader2.dataset)}')
    # test_data_loader(loader2)
    loaders=[loader0, loader1, loader2]
    data_loader = CombinedDataLoader(
            loaders_with_ratios=zip(loaders, [0.2, 0.7, 0.1]),
            batch_size=cfg.train.batch_size_per_gpu,
            combining_mode=0,
            seed=0,
            name="MultiResDL",
        )
    get_mean_std_from_dataloader(data_loader, max=10000)
    
    

    
