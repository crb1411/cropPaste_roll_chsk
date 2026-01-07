import h5py
import numpy as np
from tqdm import tqdm
import os
import traceback
import multiprocessing as mp


def convert_h5_inplace(
    src_path: str,
    dataset_key: str = "patches",
    coords_key: str = "coords",
    compression: str = "lzf",
):
    """
    原地将 H5 转换为 chunk=(1,H,W,C)，同时保留 coords。
    """

    tmp_path = src_path + ".tmp"

    try:
        # ======== 1. 读取源文件 ========
        with h5py.File(src_path, "r") as src:

            if dataset_key not in src:
                print(f"[WARN] '{dataset_key}' missing in {src_path}, skip.")
                return False
            else:
                d = src[dataset_key]
                if d.chunks is not None and d.chunks[0] == 1:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    return  src_path, False
                    
            d_patches = src[dataset_key]
            patches_shape = d_patches.shape  # (N, H, W, C)
            dtype = d_patches.dtype
            N, H, W, C = patches_shape

            # coords 可选，有的文件可能没有
            has_coords = coords_key in src
            if has_coords:
                d_coords = src[coords_key]
                coords_shape = d_coords.shape   # (N, 2)
                coords_dtype = d_coords.dtype

        chunk_shape = (1, H, W, C)

        # ======== 2. 创建临时文件 ========
        with h5py.File(tmp_path, "w") as dst:

            # ---- 新的 chunk 格式 patches ----
            new_patches = dst.create_dataset(
                dataset_key,
                shape=(N, H, W, C),
                dtype=dtype,
                chunks=chunk_shape,
                compression=compression,
            )

            # ---- 复制 coords（如果存在）----
            if has_coords:
                dst.create_dataset(
                    coords_key,
                    data=d_coords,          # 直接复制完整数组
                    dtype=coords_dtype,
                )

            # ===== 3. 再次打开源文件并复制 patches 数据 =====
            with h5py.File(src_path, "r") as src:
                src_patches = src[dataset_key]

                # 先尝试 bulk copy（最快）
                try:
                    new_patches[:] = src_patches[:]
                except Exception:
                    # fallback
                    for i in range(N):
                        try:
                            new_patches[i] = src_patches[i]
                        except Exception:
                            print(f"[ERROR] patch {i} corrupted in {src_path}")
                            dst.close()
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                            return src_path, False

        # ======== 4. 原子覆盖 ========
        os.replace(tmp_path, src_path)

        return src_path, True

    except Exception as e:
        traceback.print_exc()

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return src_path, False

def worker_convert(path, dataset_key, compression):
    try:
        ok = convert_h5_inplace(path, dataset_key, compression)
        return (path, ok)
    except Exception:
        traceback.print_exc()
        return (path, False)
# ==============================================================
# 多进程并行转换器
# ==============================================================

def parallel_convert_h5(
    txt_path,
    dataset_key="patches",
    compression="lzf",
    workers=8,
    done_txt="converted.txt",
):
    mp.set_start_method("spawn", force=True)

    # ---- 加载目标文件列表 ----
    with open(txt_path, "r") as f:
        files = [x.strip() for x in f if x.strip()]

    # ---- 加载已转换文件（支持断点续跑） ----
    done = set()
    if os.path.exists(done_txt):
        with open(done_txt, "r") as f:
            done = set(x.strip() for x in f if x.strip())

    # 过滤掉已完成的
    files = [p for p in files if p not in done]
    total = len(files)
    print(f"[INFO] Remaining = {total}")

    if total == 0:
        print("[INFO] No file to process.")
        return

    lock = mp.Lock()   # 防止多个进程同时写文件

    # ---- 回调函数（每个任务完成时调用） ----
    def on_result(ret):
        path, ok = ret
        if ok:
            with lock:
                with open(done_txt, "a") as f:
                    f.write(path + "\n")
        else:
            print(f"[FAIL] {path}")

    pool = mp.Pool(workers)

    # ---- tqdm 实时进度条 + apply_async ----
    results = []
    for path in files:
        res = pool.apply_async(
            worker_convert,
            args=(path, dataset_key, compression),
            callback=on_result,
        )
        results.append(res)

    # 用 tqdm 监控完成数量
    for _ in tqdm(results, desc="Converting", total=len(results)):
        _.wait()   # 等待某个任务完成（不会阻塞整体）

    pool.close()
    pool.join()

    print("[INFO] Done.")

# ==============================================================
# CLI 入口
# ==============================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", default="/mnt/local09/train/crb/data/h5_data_unicom_nas/h5_all.txt", help="包含 h5 路径的 txt 文件")
    parser.add_argument("--key", default="patches", help="dataset key")
    parser.add_argument("--workers", type=int, default=80)


    args = parser.parse_args()

    parallel_convert_h5(
        txt_path=args.txt,
        dataset_key=args.key,
        workers=args.workers,
        done_txt="/mnt/local09/train/crb/data/h5_data_unicom_nas/finish_h5.txt",
    )
