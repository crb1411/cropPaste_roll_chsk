import os
import h5py
import time
import multiprocessing as mp
import traceback
from tqdm import tqdm


DONE_LIST_PATH = "/mnt/local09/train/crb/data/h5_data_unicom_nas/finish_h5.txt"   # 你指定一个路径

def append_success_log(path):
    """多进程/多机安全地将成功转换的 H5 写入 txt"""
    lock_path = DONE_LIST_PATH + ".lock"
    fd = None

    # 拿锁（防止多个 worker 同时写 log）
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            time.sleep(0.05)

    try:
        with open(DONE_LIST_PATH, "a") as f:
            f.write(path + "\n")
        # print(f"append {path} to {DONE_LIST_PATH}")
    finally:
        os.close(fd)
        os.remove(lock_path)


def ensure_chunk1(file_path, dataset_key="patches"):
    """
    如果文件 chunk!=1，则原地转换为 chunk=(1,H,W,C)
    并将 coords 保持原样复制。
    转换完成后将文件路径写入 converted_success.txt。
    """
    lock_path = file_path + ".chunklock"

    # Step 1：检查是否已经是 chunk1
    try:
        with h5py.File(file_path, "r") as f:
            d = f[dataset_key]
            if d.chunks is not None and d.chunks[0] == 1:
                return  # 已经无需转换
    except Exception:
        error_info = traceback.format_exc()
        print(error_info)
        print(f'skip ========= {file_path}')
        return

    # Step 2：获取文件级转换锁
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            # if time.time() - start > 600:
            #     raise RuntimeError(f"Timeout waiting lock for {file_path}")
            print(f"skip {file_path}, 其他进程正在执行")
            return

    # Step 3：开始转换（只有拿锁者执行）
    try:
        # 再检查一次（可能已经被其他 worker 转换完）
        with h5py.File(file_path, "r") as f:
            d = f[dataset_key]
            if d.chunks is not None and d.chunks[0] == 1:
                return

        tmp_path = file_path + ".tmp"

        with h5py.File(file_path, "r") as src:
            src_patches = src[dataset_key]
            N, H, W, C = src_patches.shape
            dtype = src_patches.dtype

            # ★ 新建临时文件
            with h5py.File(tmp_path, "w") as dst:
                # 1. patches 转换为 chunk=1
                dst_p = dst.create_dataset(
                    dataset_key,
                    shape=(N, H, W, C),
                    dtype=dtype,
                    chunks=(1, H, W, C),
                    compression="lzf",
                )

                for i in range(N):
                    dst_p[i] = src_patches[i]

                # 2. ★ coords 完整保持原样复制
                if "coords" in src:
                    src_c = src["coords"]

                    # 根据源属性创建新的 coords dataset
                    dst_c = dst.create_dataset(
                        "coords",
                        data=src_c[...],          # 原样复制全部内容
                        dtype=src_c.dtype,       # 原始 dtype
                        compression=src_c.compression,  # 原始压缩
                        chunks=src_c.chunks,            # 原始 chunk
                        shuffle=src_c.shuffle,
                        fletcher32=src_c.fletcher32,
                    )

                    # 如果 coords 有 attrs，也保持一致
                    # for k, v in src_c.attrs.items():
                    #     dst_c.attrs[k] = v

        os.replace(tmp_path, file_path)

        # 记录成功日志
        append_success_log(file_path)

        # print(f"[OK] Converted to chunk1 (coords preserved): {file_path}")

    finally:
        os.close(fd)
        os.remove(lock_path)
        return None, None



def parallel_convert_h5(
    txt_path,
    dataset_key="patches",
    compression="lzf",
    workers=80,
    done_txt="converted.txt",
):


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

    with mp.Pool(workers) as p_:
        p_.map(ensure_chunk1, files)



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