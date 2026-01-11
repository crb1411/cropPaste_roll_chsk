
import argparse
import copy
import time
import gc
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch

import torch.distributed
from torch.distributed._tensor import DTensor
from torch.distributed._shard.sharded_tensor import ShardedTensor
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from dinov3.new_train.data.svs_h5.new_h5_svs_dataset import WsiPatchDataset, make_data_loader_wsi
from dinov3.new_train.data.svs_h5.svs_samplers import SamplerType_WSI
from dinov3.new_train.data.h5_file.h5_dataset import H5Dataset, H5Dataset_NoCache, worker_init_fn_h5
from dinov3.new_train.data.patches_dataset import PaddedPatchDataset
from dinov3.new_train.data.npy_dataset import ShardedPatchDataset
from dinov3.new_train.data.imgnet import ImageNetResizeDataset
import torch.multiprocessing as mp
import dinov3.distributed as distributed
from dinov3.checkpointer import (
    find_latest_checkpoint,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    keep_last_n_eval,
    load_checkpoint,
    register_dont_save_hooks,
    save_checkpoint,
)
from dinov3.new_train.data.multi_dataset import CombinedDataset, CombinedSampler, DatasetType
from dinov3.configs import get_cfg_from_args, setup_config, setup_job, setup_multidistillation
from dinov3.data import (
    DataAugmentationDINO,
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
    _make_sampler,
    CombinedDataLoader,
)
from dinov3.logging import MetricLogger, setup_logging
from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay
from dinov3.train.multidist_meta_arch import MultiDistillationMetaArch
from dinov3.new_train.train.ssl_meta_arch import SSLMetaArch
from dinov3.new_train.train.ssl_crop_roll import SSLAugmentedCropRoll
from dinov3.new_train.utils import get_device, synchronize
import traceback

logger = logging.getLogger("dinov3")

try:
    # Avoid /dev/shm limitations when multiprocessing DataLoader workers share tensors.
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

assert torch.__version__ >= (2, 1)
try:
    torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
    torch.backends.cudnn.benchmark = False  # True
except Exception as e:
    traceback.print_exc(f"Error setting torch.backends.cuda.matmul.allow_tf32: {e}")

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv3 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "--eval_pretrained_weights",
        type=str,
        default="",
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument("--seed", default=87, type=int, help="RNG seed")
    parser.add_argument(
        "--benchmark-codebase",
        action="store_true",
        help="test the codebase for a few iters",
    )
    parser.add_argument("--test-ibot", action="store_true", help="test ibot")
    parser.add_argument("--profiling", action="store_true", help="do profiling")
    parser.add_argument("--dump-fsdp-weights", action="store_true", help="dump fsdp weights")
    parser.add_argument("--record_ref_losses", action="store_true", help="record reference losses")
    parser.add_argument("--ref_losses_path", default="", type=str)
    parser.add_argument("--multi-distillation", action="store_true", help="run multi-distillation")

    # new
    parser.add_argument("--checkpoint_dir", default="")
    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))

def get_augmention(cfg):
    if hasattr(cfg, 'legacy_augmentor'):
        use_legacy_augmentor = True
    else:
        use_legacy_augmentor = False
    return DataAugmentationDINO(
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
        use_legacy_augmentor=use_legacy_augmentor,
    )

def build_schedulers(cfg):
    if "schedules" in cfg:
        logger.info("Using schedules v2")
        return build_schedulers_v2(cfg)

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = (
        0  # mimicking the original schedules
    )
    logger.info("Schedulers ready.")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def build_schedulers_v2(cfg):
    iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
    total_iterations = cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs
    logger.info(f"Total training iterations {total_iterations}")

    # LR scaling rules
    lr_peak = cfg.schedules.lr.peak
    lr_end = cfg.schedules.lr.end
    if cfg.optim.scaling_rule == "linear_wrt_256":
        lr_peak *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        lr_end *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        lr_peak *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        lr_end *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    else:
        logger.info(f"No scaling rule for {cfg.optim.scaling_rule=}")

    lr = linear_warmup_cosine_decay(
        start=cfg.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=iter_per_epoch * cfg.schedules.lr.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.lr.cosine_epochs if "cosine_epochs" in cfg.schedules.lr else None
        ),
    )
    last_layer_lr = lr.copy()
    last_layer_lr[: iter_per_epoch * cfg.schedules.lr.freeze_last_layer_epochs] = 0
    weight_decay = linear_warmup_cosine_decay(
        start=cfg.schedules.weight_decay.start,
        peak=cfg.schedules.weight_decay.peak,
        end=cfg.schedules.weight_decay.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.weight_decay.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.weight_decay.cosine_epochs
            if "cosine_epochs" in cfg.schedules.weight_decay
            else None
        ),
    )
    momentum = linear_warmup_cosine_decay(
        start=cfg.schedules.momentum.start,
        peak=cfg.schedules.momentum.peak,
        end=cfg.schedules.momentum.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.momentum.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.momentum.cosine_epochs if "cosine_epochs" in cfg.schedules.momentum else None
        ),
    )
    teacher_temp = linear_warmup_cosine_decay(
        start=cfg.schedules.teacher_temp.start,
        peak=cfg.schedules.teacher_temp.peak,
        end=cfg.schedules.teacher_temp.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.teacher_temp.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in cfg.schedules.teacher_temp
            else None
        ),
    )
    return lr, weight_decay, momentum, teacher_temp, last_layer_lr

def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        if is_last_layer:
            param_group["lr"] = last_layer_lr * lr_multiplier
        else:
            param_group["lr"] = lr * lr_multiplier


def do_test(cfg, model, iteration, process_group, do_low_freq=False):
    # dump a sharded checkpoint
    eval_dir = Path(cfg.train.output_dir) / "eval" / str(iteration)
    if distributed.is_subgroup_main_process():
        eval_dir.mkdir(parents=True, exist_ok=True)
    if cfg.train.sharded_eval_checkpoint:
        ckpt_path = eval_dir / "sharded_teacher_checkpoint"
        if distributed.is_subgroup_main_process():
            ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        teacher_backbone = model.model_ema
        save_checkpoint(
            ckpt_dir=ckpt_path, iteration=iteration, model=teacher_backbone, overwrite=True, process_group=process_group
        )
        if not distributed.is_subgroup_main_process():
            return
    else:
        new_state_dict = model.model_ema.state_dict()
        for k, tensor in list(new_state_dict.items()):
            if isinstance(tensor, DTensor):
                new_state_dict[k] = tensor.full_tensor()
        if not distributed.is_subgroup_main_process():
            return
        # save teacher checkpoint
        ckpt_path = eval_dir / "teacher_checkpoint.pth"
        torch.save({"teacher": new_state_dict}, ckpt_path)
        logger.info("Saved eval checkpoint: %s", ckpt_path)

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
    elif dataset_type==DatasetType.IMGNET_DATA:
        dataset_path = 'ImageNet:split=TRAIN:root=/mnt/local09/train/crb/env_init/data/dataset_imagenet/tiny-imagenet:extra=/mnt/local09/train/crb/env_init/data/dataset_imagenet/tiny-imagenet_extra'
        dataset = make_dataset(
            dataset_str=dataset_path,
            transform=get_augmention(cfg),
            target_transform=None,
        )
    elif dataset_type==DatasetType.IMGNET_DATA_2:
        dataset = ImageNetResizeDataset(
            index_npy='/mnt/data/imagenet_1k/data_new/imagenet_1k.npy',
            size = 224,
            transform=get_augmention(cfg),
        )
    elif dataset_type==DatasetType.PATCHES_DATA:
        dataset = PaddedPatchDataset(
            index_npy="/mnt/local09/train/crb/data/img_data_v1212/patch_index.npy",
            image_list_txt="/mnt/local09/train/crb/data/img_data_v1212/patch_index.txt",
            patch_size=224,
            pad_value=0,
            transform=get_augmention(cfg),
        )
    elif dataset_type==DatasetType.NPY_DATA:
        shard_dir = ['/mnt/local10/data_new/shard', '/mnt/local10/data_new/shard_10', '/mnt/local10/data_new/shard_task34']
        index_path = ["/mnt/local10/data_new/shard/big_index.npy", "/mnt/local10/data_new/shard_10/big_index.npy", '/mnt/local10/data_new/shard_task34/big_index.npy']
        dataset = ShardedPatchDataset(
            shard_dir=shard_dir,
            index_path=index_path,
            transform=get_augmention(cfg),
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

def _wi(worker_id):
    import os, multiprocessing as mp_
    print(f"[WORKER] pid={os.getpid()} worker={worker_id} start_method={mp_.get_start_method()}")


def build_CombinedDataset_loader(cfg, start_iter=0):
    
    # dataset_svs = build_dataset_from_cfg_wsi(cfg, DatasetType.SVS_DATE)
    # dataset_h5 = build_dataset_from_cfg_wsi(cfg, DatasetType.H5_DATE)
    # dataset_img = build_dataset_from_cfg_wsi(cfg, DatasetType.IMGNET_DATA)
    # dataset_patches = build_dataset_from_cfg_wsi(cfg, DatasetType.PATCHES_DATA)
    # dataset_npy = build_dataset_from_cfg_wsi(cfg, DatasetType.NPY_DATA)
    # shard_dir = ['/mnt/local08/data_new/shard']
    # index_path = ["/mnt/local08/data_new/big_index.npy"]
    # dataset_local08 = ShardedPatchDataset(
    #     shard_dir=shard_dir,
    #     index_path=index_path,
    #     transform=get_augmention(cfg),
    # )
    # shard_dir = ['/mnt/local10/data_new/shard_tcga']
    # index_path = ["/mnt/local10/data_new/shard_tcga/big_index.npy"]
    # dataset_tcga = ShardedPatchDataset(
    #     shard_dir=shard_dir,
    #     index_path=index_path,
    #     transform=get_augmention(cfg),
    # )
    # dataset_multiscale = WsiPatchDataset(
    #         tensor_transform=get_augmention(cfg),
    #         patch_npy='/mnt/local09/train/crb/data/svs_data_v1/index.npy',
    #         path_txt='/mnt/local09/train/crb/data/svs_data_v1/path.txt',
    #         fix_size=None,
    #         return_dic=False
    #     )

    dataset_imagenet2 = build_dataset_from_cfg_wsi(cfg, DatasetType.IMGNET_DATA_2)
    datasets = [dataset_imagenet2]
    samplers = []
    ratios=[1.0]
    info_str = " \n".join(
            f"{ds.__class__.__name__}(len={len(ds)}, ratio={r})"
            for ds, r in zip(datasets, ratios)
        )
    logger.info(f"Use datasets: \n{info_str}")
    seed_num = 0
    for ds, ratio in zip(datasets, ratios):
        sampler_type = SamplerType.SHARDED_INFINITE_NEW
        sampler = _make_sampler(
            dataset=ds,
            type=sampler_type,
            shuffle=True,
            seed=seed_num,
            advance=int(start_iter*cfg.train.batch_size_per_gpu*ratio),
        )
        seed_num += 1
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
        multiprocessing_context="spawn",
        worker_init_fn=_wi,
        persistent_workers=True,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        timeout=0,
    )
    return loader_combined




def do_train(cfg, model, resume=False):
    process_subgroup = distributed.get_process_subgroup()
    device_type = get_device()
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    # Optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    if cfg.multidistillation.enabled:
        register_dont_save_hooks(
            model,
            dont_save=[k for k, _ in model.state_dict().items() if k.startswith("teacher")],
        )
    model.init_weights()
    start_iter = 0
    if resume:
        logger.info(f"Checkpoint found {cfg.checkpoint_dir}")
        model.to_empty(device=device_type.type)
        start_iter = (
            load_checkpoint(
                cfg.checkpoint_dir,
                model=model,
                optimizer=None,
                strict_loading=False,
                process_group=process_subgroup,
            )
            + 1
        )
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    if cfg.multidistillation.enabled:
        global_batch_size = cfg.multidistillation.global_batch_size
    else:
        global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_world_size()

    # Build data loader
    data_loader = build_CombinedDataset_loader(
        cfg,
        start_iter=start_iter
    )

    # Metric logging
    logger.info("Starting training from iteration %d", start_iter)
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    # Manual garbage collection
    gc.disable()
    gc.collect()

    # Training loop
    student = model.student
    iteration = start_iter
    num_gram_updates = 0
    if (
        cfg.gram.use_loss
        and model.has_gram_teacher
        and cfg.gram.rep_update
        and start_iter > 0
        and start_iter >= cfg.gram.it_first_update
    ):
        # If `start_iter == it_first_update`, we have performed one gram teacher update after
        # iteration `start_iter - 1`, except if we are starting training from scratch and `start_iter == 0`.
        num_gram_updates = math.ceil((start_iter + 1 - cfg.gram.it_first_update) / cfg.gram.update_frequency)
        logger.info(f"Gram was updated {num_gram_updates} times before iteration {start_iter}")
    consecutive_nan_count = 0
    start_train_time = time.time()
    end_train_time = time.time()
    all_iteration_time = -1
    logger_freq = 2
    for data in metric_logger.log_every(
        data_loader,
        print_freq=logger_freq,
        header="Training",
        n_iterations=max_iter,
        start_iteration=start_iter,
    ):
        it = iteration
        all_iteration_time = time.time() - start_train_time
        start_train_time = time.time()
        
        data["global_batch_size"] = global_batch_size
        if iteration > max_iter:
            return

        # Garbage collection (trigger manually so it happens on all ranks at the same time)
        if (iteration + 1) % 150 == 0:
            logger.info("Garbage collection")
            gc.collect()

        if cfg.gram.use_loss and model.gram_it_load_ema_teacher == it:
            logger.info(f"Loading EMA teacher into Gram teacher before iteration {it}")
            model.gram_load_ema_teacher()

        # Learning rates and other schedules
        lr = lr_schedule[it]
        wd = wd_schedule[it]
        mom = momentum_schedule[it]
        teacher_temp = teacher_temp_schedule[it]
        last_layer_lr = last_layer_lr_schedule[it]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # Forward backward
        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it, logger_freq=logger_freq)
        forward_backward_time = time.time() - start_train_time

        # Gradient clipping
        clip_start_time = time.time()
        if cfg.optim.clip_grad:
            for k, v in student.items():
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    v.parameters(),
                    max_norm=cfg.optim.clip_grad,
                )
                metrics_dict[f"{k}_grad_norm"] = (
                    grad_norm.full_tensor().item()
                    if isinstance(grad_norm, torch.distributed.tensor.DTensor)
                    else grad_norm.item()
                )
        clip_grad_time = time.time() - clip_start_time
        # Reduce total_loss to check for NaNs, reduce metrics for logging
        reduce_start_time = time.time()
        total_loss_all_ranks = total_loss.new_empty(distributed.get_subgroup_size())
        torch.distributed.all_gather_into_tensor(
            total_loss_all_ranks,
            total_loss.detach(),
            group=distributed.get_process_subgroup(),
        )
        total_loss = total_loss_all_ranks.mean()
        metrics_values = torch.stack(
            [torch.as_tensor(v, dtype=torch.float32, device=total_loss.device).detach() for v in metrics_dict.values()]
        )
        torch.distributed.all_reduce(
            metrics_values,
            op=torch.distributed.ReduceOp.AVG,
            group=distributed.get_process_subgroup(),
        )
        reduce_time = time.time() - reduce_start_time
        metrics_dict = dict(zip(metrics_dict.keys(), metrics_values))
        if total_loss_all_ranks.isnan().any():
            consecutive_nan_count += 1
            which_ranks = total_loss_all_ranks.isnan().nonzero().flatten().tolist()
            logger.warning("NaN loss detected on ranks: %s", which_ranks)
            logger.warning("Consecutive NaNs: %d", consecutive_nan_count)
            metrics_dict_str = "\n".join([f"{k}: {v}" for k, v in metrics_dict.items()])
            logger.warning("All-reduced metrics:\n%s", metrics_dict_str)
            if consecutive_nan_count > 2 and not cfg.multidistillation.enabled:
                msg = "Too many consecutive nans detected in loss, aborting..."
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            consecutive_nan_count = 0
        step_start_time = time.time()
        # Step optimizer
        optimizer.step()
        model.update_ema(mom)
        step_time = time.time() - step_start_time
        # [GRAM] Update gram teacher when using gram teacher and frequent updates
        if (
            cfg.gram.use_loss
            and model.gram_rep_update
            and (it + 1) >= model.gram_it_first_update
            and (it + 1) % model.gram_update_frequency == 0
            and (cfg.gram.max_updates is None or num_gram_updates < cfg.gram.max_updates)
        ):
            logger.info(f"Updating Gram teacher from EMA teacher after iteration {it}")
            model.update_gram()
            num_gram_updates += 1

        # Log metrics
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(teacher_temp=teacher_temp)
        metric_logger.update(total_loss=total_loss, **metrics_dict)
        all_time_dic = {
            'all_iteration_time': all_iteration_time,
            'step_time': step_time,
            'forward_backward_time': forward_backward_time,
            'clip_grad_time': clip_grad_time,
            'reduce_time': reduce_time,
            'data_time': start_train_time - end_train_time
        }
        metric_logger.update(**all_time_dic)
        

        
        # Checkpointing
        if (iteration + 1) % cfg.checkpointing.period == 0:
            do_test(cfg, model, f"training_{iteration}", process_group=process_subgroup)
            synchronize(device_type)
            save_checkpoint(
                ckpt_dir / str(iteration),
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                overwrite=True,
                process_group=process_subgroup,
            )
            if distributed.is_subgroup_main_process():
                keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)
                eval_dir = Path(cfg.train.output_dir) / "eval" 
                keep_last_n_eval(eval_dir, cfg.checkpointing.max_to_keep)
                if "keep_every" in cfg.checkpointing and (iteration + 1) % cfg.checkpointing.keep_every == 0:
                    keep_checkpoint_copy(ckpt_dir / str(iteration))
                
        iteration = iteration + 1
        end_train_time = time.time()
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(argv=None):
    if argv is None:
        args = get_args_parser().parse_args()
    else:
        args = get_args_parser().parse_args(argv[1:])
        args.output_dir = sys.argv[1]
    args.config_file = args.config_file if os.path.isfile(args.config_file) else '/mnt/work/git_proj/dinov3/dinov3/configs/crop_rollv1.yaml'
    base_dir = args.output_dir if args.output_dir is not None else '/mnt/data/train/crb/train_out/train_legacy_debug'
    try:
        from dinov3.new_train.utils.log_create import creat_subdir
        args.output_dir = creat_subdir(base_dir=base_dir, create=True, time=True)
    except:
        raise ImportError
    device_type = None
    if args.multi_distillation:
        print("performing multidistillation run")
        cfg = setup_multidistillation(args)
        torch.distributed.barrier()
        logger.info("setup_multidistillation done")
        assert cfg.MODEL.META_ARCHITECTURE == "MultiDistillationMetaArch"
        device_type = get_device()
    else:
        device_type = get_device()
        setup_job(output_dir=args.output_dir, seed=args.seed)
        cfg = setup_config(args, strict_cfg=False)
        logger.info(cfg)
        setup_logging(
            output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
            name="nan_logger",
        )
    logger.info(f'Use device: {device_type.type}')
    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "SSLCropRoll": SSLAugmentedCropRoll,
        "MultiDistillationMetaArch": MultiDistillationMetaArch,
    }.get(cfg.MODEL.META_ARCHITECTURE, None)
    if meta_arch is None:
        raise ValueError(f"Unknown MODEL.META_ARCHITECTURE {cfg.MODEL.META_ARCHITECTURE}")
    logger.info(f"Making meta arch {meta_arch.__name__}")
    with torch.device("meta"):
        model = meta_arch(cfg)
    model.prepare_for_distributed_training()
    # Fill all values with `nans` so that we identify
    # non-initialized values
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device=device_type,
        ),
        recurse=True,
    )
    logger.info(f"Model after distributed:\n{model}")
    if args.eval_only:
        model.init_weights()
        iteration = (
            model.get_checkpointer_class()(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    if os.path.isdir(args.checkpoint_dir):
        resume = True
        cfg.checkpoint_dir = args.checkpoint_dir
        logger.info(f'load model from {args.checkpoint_dir}/{cfg.checkpoint_dir}')
    else:
        resume = False

    do_train(cfg, model, resume=resume)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if mp.get_start_method(allow_none=True) not in ("spawn", "forkserver"):
        mp.set_start_method("spawn", force=True)
    main()
