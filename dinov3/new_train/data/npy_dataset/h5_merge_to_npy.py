import os
import zlib
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from itertools import accumulate

# ============================================================
# 常量
# ============================================================
PATCH_H = 224
PATCH_W = 224
PATCH_C = 3            # 最终输出 RGB
PATCH_BYTES = PATCH_H * PATCH_W * PATCH_C

SHARD_SIZE_BYTES = 2 * 1024**4    # 2TB
SHARD_CAP = int((SHARD_SIZE_BYTES // PATCH_BYTES) * 0.98)

MAX_PER_FILE = 500                # 每个 h5 最多保留 500 patch

INDEX_DTYPE = np.dtype([
    ("h5_id", np.int32),      # 第几个 h5
    ("patch_id", np.int32),   # 原始 h5 里的 patch 下标
    ("global_idx", np.int64)  # 在大库里的全局 index
])


# ============================================================
# small H5 loader → 返回 (N,224,224,3)
# ============================================================
def load_h5(path):
    try:
        with h5py.File(path, "r") as f:
            arr = f["patches"][:]     # (N,H,W,C)
    except Exception as e:
        print(f"[ERROR] load H5: {path} : {e}")
        return None

    if arr.ndim != 4:
        print(f"[ERROR] invalid shape in {path} : {arr.shape}")
        return None

    C = arr.shape[-1]
    if C == 4:
        arr = arr[..., :3]          # RGBA → RGB
    elif C != 3:
        print(f"[ERROR] unexpected channels={C} in {path}")
        return None

    return arr.astype(np.uint8)


# ============================================================
# 打开 / 创建 shard
# ============================================================
def open_shard(shard_path):
    shape = (SHARD_CAP, PATCH_H, PATCH_W, PATCH_C)

    # 如果之前有残留但没有 .done，当作未完成，重新创建
    if os.path.exists(shard_path) and not os.path.exists(shard_path + ".done"):
        os.remove(shard_path)

    if not os.path.exists(shard_path):
        mem = np.memmap(shard_path, dtype=np.uint8, mode="w+", shape=shape)
        del mem

    mem = np.memmap(shard_path, dtype=np.uint8, mode="r+", shape=shape)
    return mem


# ============================================================
# 写一个 shard（单机单 shard，不需要锁）
# ============================================================
def process_shard(shard_id,
                  shard_tasks,
                  h5_files,
                  global_offsets,
                  n_keep_list,
                  shard_dir,
                  index_path):
    shard_path = os.path.join(shard_dir, f"shard_{shard_id:05d}.npy")
    done_flag = shard_path + ".done"

    # 断点：已经完成的 shard 直接跳过
    if os.path.exists(done_flag):
        print(f"[INFO] Shard {shard_id} already done, skip.")
        return True

    print(f"[INFO] Start shard {shard_id}, files={len(shard_tasks)}")

    shard_mem = open_shard(shard_path)
    index_mmap = np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")

    for h5_id in tqdm(shard_tasks, desc=f"Shard {shard_id}", ncols=100):
        h5_path = h5_files[h5_id]
        n_keep = n_keep_list[h5_id]
        if n_keep == 0:
            continue

        # 从 big_index 里拿到这个 h5 对应的原始 patch 下标
        g0 = global_offsets[h5_id]
        idx_slice = slice(g0, g0 + n_keep)
        patch_ids = index_mmap[idx_slice]["patch_id"]

        arr = load_h5(h5_path)
        if arr is None:
            print(f"[WARN] skip bad H5: {h5_path}")
            continue

        n_orig = arr.shape[0]
        if n_orig == 0:
            continue

        try:
            # 只取选中的 patch
            arr_sel = arr[patch_ids]
        except Exception as e:
            print(f"[ERROR] select patches failed for {h5_path}: {e}")
            continue

        N = arr_sel.shape[0]
        if N == 0:
            continue

        local0 = g0 % SHARD_CAP

        try:
            shard_mem[local0: local0 + N] = arr_sel[:]
        except Exception as e:
            print(f"[ERROR] write failed for {h5_path}: {e}")
            continue

        # 自动删除原 H5（可按需注释掉）
        try:
            os.remove(h5_path)
        except:
            pass

        del arr
        del arr_sel

    shard_mem.flush()
    del shard_mem
    del index_mmap

    # 标记 shard 完成
    with open(done_flag, "w") as f:
        f.write("done\n")

    print(f"[INFO] Shard {shard_id} done.")
    return True


# ============================================================
# big_index.npy：同时完成随机采样（最多 500）
# ============================================================
def build_big_index(h5_files, n_orig_list, index_path):
    n_keep_list = []
    paths = list(h5_files)

    # 先算每个文件最终保留多少 patch
    for n in n_orig_list:
        if n is None or n <= 0:
            n_keep_list.append(0)
        else:
            n_keep_list.append(min(n, MAX_PER_FILE))

    total = sum(n_keep_list)
    print(f"[INFO] big_index total patches = {total:,}")

    if total == 0:
        raise RuntimeError("No valid patches found.")

    mmap = np.memmap(index_path, dtype=INDEX_DTYPE, mode="w+", shape=(total,))

    offset = 0
    for h5_id, (p, n_orig, n_keep) in enumerate(zip(paths, n_orig_list, n_keep_list)):
        if n_keep == 0 or n_orig is None or n_orig <= 0:
            continue

        # 路径确定的随机种子 → 采样稳定，可支持断点
        seed = zlib.crc32(p.encode("utf-8")) ^ 0x1234ABCD
        rng = np.random.RandomState(seed)

        if n_orig <= n_keep:
            sel = np.arange(n_orig, dtype=np.int32)
        else:
            sel = rng.choice(n_orig, size=n_keep, replace=False).astype(np.int32)

        arr = np.zeros(n_keep, dtype=INDEX_DTYPE)
        arr["h5_id"] = h5_id
        arr["patch_id"] = sel
        arr["global_idx"] = np.arange(offset, offset + n_keep, dtype=np.int64)
        mmap[offset:offset + n_keep] = arr
        offset += n_keep

    mmap.flush()
    del mmap
    print(f"[INFO] big_index.npy written to {index_path}")

    return n_keep_list


# ============================================================
# 主入口：单机多进程 + 断点
# ============================================================
def merge_all_h5(
    h5_list_file,
    shard_dir,
    index_out="big_index.npy",
    map_out="h5id_to_h5path.txt",
    workers=32,
):
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # =========== Step 1: 读取 H5 列表 ===========
    with open(h5_list_file) as f:
        h5_files = [x.strip() for x in f if x.strip()]

    print(f"[INFO] Total H5 files: {len(h5_files)}")

    # =========== Step 2: 扫描原始 patch 数 ===========
    n_orig_list = []
    for p in tqdm(h5_files, desc="scan H5 patches"):
        try:
            with h5py.File(p, "r") as f:
                n = f["patches"].shape[0]
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
            n = 0
        n_orig_list.append(n)

    # =========== Step 3: 构建 big_index（顺便完成随机采样） ===========
    n_keep_list = build_big_index(h5_files, n_orig_list, index_out)

    # 用保留后的 n_keep 计算 offsets
    global_offsets = list(accumulate([0] + n_keep_list))[:-1]
    total_patches = sum(n_keep_list)
    print(f"[INFO] Total kept patches = {total_patches:,}")

    if total_patches == 0:
        print("[WARN] No patches kept, exit.")
        return

    # =========== Step 4: 存储 H5 映射 ===========
    with open(map_out, "w") as f:
        for i, p in enumerate(h5_files):
            f.write(f"{i}\t{p}\n")
    print(f"[INFO] mapping written to {map_out}")

    # =========== Step 5: 分配每个 H5 到哪个 shard ===========
    total_shards = (total_patches + SHARD_CAP - 1) // SHARD_CAP
    print(f"[INFO] Total shards = {total_shards}")

    shard_tasks = [[] for _ in range(total_shards)]
    for i in range(len(h5_files)):
        if n_keep_list[i] == 0:
            continue
        gid = global_offsets[i]
        shard_id = gid // SHARD_CAP
        shard_tasks[shard_id].append(i)

    # =========== Step 6: 多进程处理每个 shard（单机） ===========
    tasks = []
    for sid in range(total_shards):
        if not shard_tasks[sid]:
            continue
        tasks.append((sid, shard_tasks[sid], h5_files,
                      global_offsets, n_keep_list,
                      str(shard_dir), index_out))

    print(f"[INFO] Start parallel merge, workers={workers}")
    with Pool(workers) as pool:
        list(pool.starmap(process_shard, tasks))

    print("[INFO] All shards done.")


# ============================================================
# 示例
# ============================================================
if __name__ == "__main__":
    base_dir = '/mnt/local10/data_new'
    shard_dir = os.path.join(base_dir, 'shard_10')
    index_out= os.path.join(shard_dir, 'big_index.npy')
    map_out=os.path.join(shard_dir, 'h5id_to_npypath.txt')
    tmp_dir=os.path.join(base_dir, 'state')
    h5_list_file='/mnt/local09/train/crb/data/npydata_local10/h5_all.txt'
    merge_all_h5(
        h5_list_file=h5_list_file,
        shard_dir=shard_dir,
        index_out=index_out,
        map_out=map_out,
        workers=88,
    )
