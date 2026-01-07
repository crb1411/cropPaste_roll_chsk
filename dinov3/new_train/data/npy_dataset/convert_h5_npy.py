import os
import h5py
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import traceback


def load_h5_to_rgb(h5_path):
    try:
        with h5py.File(h5_path, "r") as f:
            arr = f["patches"][:]
        C = arr.shape[-1]

        if C == 4:
            arr = arr[..., :3]
        elif C == 3:
            pass
        else:
            print(f"[ERROR] channel={C} in {h5_path}")
            return None

        return arr.astype(np.uint8)

    except Exception:
        print(f"[ERROR] Failed loading {h5_path}")
        traceback.print_exc()
        return None


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def worker(args):
    h5_path, src_root, dst_root = args

    # 构造相对路径（保持原目录结构）
    rel = os.path.relpath(h5_path, src_root)
    out_path = os.path.join(dst_root, rel).replace(".h5", ".npy")
    tmp_path = out_path + ".tmp"

    # 1. 已完成 → 跳过
    if os.path.exists(out_path):
        return None

    # 2. 别的机器正在写 → 跳过
    if os.path.exists(tmp_path):
        return None

    ensure_dir(os.path.dirname(out_path))

    arr = load_h5_to_rgb(h5_path)
    if arr is None:
        return None

    try:
        # 3. 创建 tmp 写入锁
        with open(tmp_path, "wb") as f:
            np.save(f, arr)

        # 4. 原子替换
        os.replace(tmp_path, out_path)

    except Exception:
        print(f"[ERROR] Save failed: {h5_path}")
        traceback.print_exc()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None

    return out_path


def main(txt, src_root, dst_root, workers=16):
    with open(txt) as f:
        h5_list = [x.strip() for x in f if x.strip()]

    print(f"Loaded {len(h5_list)} h5 paths.")

    tasks = [(p, src_root, dst_root) for p in h5_list]

    with Pool(workers) as pool:
        for _ in tqdm(pool.imap(worker, tasks), total=len(tasks)):
            pass

    print("All conversions finished.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    txt = '/mnt/local09/train/crb/data/mv_h5/h5_task_2.txt'
    src = '/mnt/unicom_nas/data/inhouse/ssl/bundles'
    dst = '/mnt/local08/data_new/h5_task_2_100k'
    main(txt, src, dst, args.workers)
