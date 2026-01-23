import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

import torch
import torch.distributed as torch_dist
import torch.multiprocessing as mp

import dinov3.distributed as distributed
from dinov3.configs import setup_config, setup_job, setup_multidistillation
from dinov3.logging import setup_logging
from dinov3.new_train.train.train_mul_no_opt_imgnet import (
    build_CombinedDataset_loader,
    get_args_parser,
)
from dinov3.new_train.utils import get_device
from dinov3.new_train.utils.log_create import creat_subdir

logger = logging.getLogger("dinov3")


def _infer_batch_size(batch, fallback: int) -> int:
    if isinstance(batch, dict):
        for key in ("collated_global_crops", "global_crops", "images"):
            value = batch.get(key)
            if torch.is_tensor(value) and value.ndim >= 1:
                return int(value.shape[0])
    if torch.is_tensor(batch) and batch.ndim >= 1:
        return int(batch.shape[0])
    if isinstance(batch, (list, tuple)) and batch:
        value = batch[0]
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    return int(fallback)


def _dist_enabled() -> bool:
    return torch_dist.is_available() and torch_dist.is_initialized()


def _barrier() -> None:
    if _dist_enabled():
        torch_dist.barrier()


def _reduce_stats(total_batches: int, total_samples: int, elapsed: float, data_time_sum: float):
    if not _dist_enabled():
        return total_batches, total_samples, elapsed, data_time_sum
    device = get_device()
    counts = torch.tensor(
        [float(total_batches), float(total_samples), float(data_time_sum)],
        device=device,
        dtype=torch.float64,
    )
    torch_dist.all_reduce(counts, op=torch_dist.ReduceOp.SUM)
    elapsed_t = torch.tensor(float(elapsed), device=device, dtype=torch.float64)
    torch_dist.all_reduce(elapsed_t, op=torch_dist.ReduceOp.MAX)
    return int(counts[0].item()), int(counts[1].item()), float(elapsed_t.item()), float(counts[2].item())


def benchmark_loader(loader, *, iters: int, warmup_iters: int, log_freq: int) -> None:
    total_batches = 0
    total_samples = 0
    batch_size = loader.batch_size or 0
    start_time = None
    data_time_sum = 0.0
    data_time_max = 0.0
    last_time = time.perf_counter()

    for i, batch in enumerate(loader):
        now = time.perf_counter()
        data_time = now - last_time
        last_time = now
        if batch_size <= 0:
            batch_size = _infer_batch_size(batch, fallback=0)
        if i < warmup_iters:
            continue
        if start_time is None:
            _barrier()
            start_time = time.perf_counter()
            last_time = start_time
            data_time_sum = 0.0
            data_time_max = 0.0
            total_batches = 0
            total_samples = 0
            continue
        total_batches += 1
        total_samples += batch_size
        data_time_sum += data_time
        data_time_max = max(data_time_max, data_time)

        if log_freq and total_batches % log_freq == 0 and distributed.is_main_process():
            elapsed = time.perf_counter() - start_time
            avg_data_ms = (data_time_sum / total_batches) * 1000.0 if total_batches else 0.0
            logger.info(
                "iter=%d batches=%d samples=%d | %.2f batches/s %.2f samples/s | data=%.2f ms avg=%.2f ms",
                i,
                total_batches,
                total_samples,
                total_batches / elapsed if elapsed > 0 else 0.0,
                total_samples / elapsed if elapsed > 0 else 0.0,
                data_time * 1000.0,
                avg_data_ms,
            )

        if iters and total_batches >= iters:
            break

    if start_time is None:
        if distributed.is_main_process():
            logger.info("No batches consumed. Check dataset/sampler settings.")
        return

    elapsed = time.perf_counter() - start_time
    local_bps = total_batches / elapsed if elapsed > 0 else 0.0
    local_sps = total_samples / elapsed if elapsed > 0 else 0.0
    local_avg_data_ms = (data_time_sum / total_batches) * 1000.0 if total_batches else 0.0
    local_max_data_ms = data_time_max * 1000.0

    _barrier()
    global_batches, global_samples, global_elapsed, global_data_time_sum = _reduce_stats(
        total_batches, total_samples, elapsed, data_time_sum
    )
    global_bps = global_batches / global_elapsed if global_elapsed > 0 else 0.0
    global_sps = global_samples / global_elapsed if global_elapsed > 0 else 0.0
    global_avg_data_ms = (
        (global_data_time_sum / global_batches) * 1000.0 if global_batches else 0.0
    )

    if distributed.is_main_process():
        logger.info(
            "Local throughput: %.2f batches/s %.2f samples/s | data_avg=%.2f ms data_max=%.2f ms "
            "(batches=%d samples=%d elapsed=%.2fs)",
            local_bps,
            local_sps,
            local_avg_data_ms,
            local_max_data_ms,
            total_batches,
            total_samples,
            elapsed,
        )
        if _dist_enabled():
            logger.info(
                "Global throughput: %.2f batches/s %.2f samples/s | data_avg=%.2f ms "
                "(batches=%d samples=%d elapsed=%.2fs)",
                global_bps,
                global_sps,
                global_avg_data_ms,
                global_batches,
                global_samples,
                global_elapsed,
            )


def main() -> None:
    args = get_args_parser().parse_args()
    if not os.path.isfile(args.config_file):
        args.config_file = "/mnt/work/git_proj/dinov3/dinov3/configs/crop_rollv1.yaml"
    base_dir = (
        args.output_dir
        if args.output_dir is not None
        else "/mnt/data/train/crb/train_out/train_legacy_debug"
    )
    args.output_dir = creat_subdir(base_dir=base_dir, create=True, time=True)

    if args.multi_distillation:
        cfg = setup_multidistillation(args)
        torch_dist.barrier()
    else:
        setup_job(output_dir=args.output_dir, seed=args.seed)
        cfg = setup_config(args, strict_cfg=False)
        logger.info(cfg)
        setup_logging(
            output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
            name="nan_logger",
        )

    loader = build_CombinedDataset_loader(cfg, start_iter=0)
    total_iters = int(getattr(cfg.train, "OFFICIAL_EPOCH_LENGTH", 0))
    if total_iters <= 0:
        total_iters = 200
    warmup_iters = min(20, max(0, total_iters // 10))
    log_freq = 20

    if distributed.is_main_process():
        logger.info(
            "Start dataloader throughput test: batch_size=%s num_workers=%s",
            loader.batch_size,
            cfg.train.num_workers,
        )
        logger.info(
            "Measure iters=%d warmup_iters=%d log_freq=%d",
            total_iters,
            warmup_iters,
            log_freq,
        )

    benchmark_loader(
        loader,
        iters=total_iters,
        warmup_iters=warmup_iters,
        log_freq=log_freq,
    )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if mp.get_start_method(allow_none=True) not in ("spawn", "forkserver"):
        mp.set_start_method("spawn", force=True)
    main()
