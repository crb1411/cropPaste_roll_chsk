import os
import numpy as np
import fcntl
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from itertools import accumulate

# ============================================================
# 常量
# ============================================================
PATCH_H = 224
PATCH_W = 224
PATCH_C = 3
PATCH_BYTES = PATCH_H * PATCH_W * PATCH_C

SHARD_SIZE_BYTES = 2 * 1024**4  # 2TB
SHARD_CAP = int((SHARD_SIZE_BYTES // PATCH_BYTES) * 0.98)

INDEX_DTYPE = np.dtype([
    ("h5_id", np.int32),
    ("patch_id", np.int32),
    ("global_idx", np.int64)
])


# ============================================================
# small npy loader
# ============================================================
def load_npy(path):
    try:
        arr = np.load(path)
        assert arr.ndim == 4, f"shape error: {arr.shape}"
        return arr
    except Exception as e:
        print("[ERROR] load:", path, e)
        return None


# ============================================================
# 创建 / 打开 shard
# ============================================================
def open_shard(shard_path):
    shape = (SHARD_CAP, PATCH_H, PATCH_W, PATCH_C)

    if not os.path.exists(shard_path):
        mem = np.memmap(shard_path, dtype=np.uint8, mode="w+", shape=shape)
        del mem

    mem = np.memmap(shard_path, dtype=np.uint8, mode="r+", shape=shape)
    return mem


# ============================================================
# 写一个 shard（由单台机器单进程负责）
# ============================================================



def process_shard(shard_id, shard_tasks, npy_files, global_offsets, shard_dir):
    shard_path = os.path.join(shard_dir, f"shard_{shard_id:05d}.npy")

    print(f"[INFO] Start shard {shard_id}, files={len(shard_tasks)}")

    shard_mem = open_shard(shard_path)

    for file_id in tqdm(shard_tasks, desc=f"Shard {shard_id}", ncols=100):
        npy_path = npy_files[file_id]

        arr = load_npy(npy_path)
        if arr is None:
            print(f"[WARN] skip bad file: {npy_path}")
            continue

        N = arr.shape[0]
        if N == 0:
            continue

        g0 = global_offsets[file_id]
        local0 = g0 % SHARD_CAP

        try:
            shard_mem[local0: local0 + N] = arr[:]
        except Exception as e:
            print(f"[ERROR] write shard failed for {npy_path}: {e}")
            continue

        # 写入成功 → 删除原 small npy
        try:
            os.remove(npy_path)
        except:
            pass

        del arr

    shard_mem.flush()
    del shard_mem

    print(f"[INFO] Shard {shard_id} done.")
    return True


# ============================================================
# 大索引 big_index.npy
# ============================================================
def build_big_index(n_patches, index_path):
    total = sum(n_patches)
    mmap = np.memmap(index_path, dtype=INDEX_DTYPE, mode="w+", shape=(total,))

    offset = 0
    for h5_id, n in enumerate(n_patches):
        arr = np.zeros(n, dtype=INDEX_DTYPE)
        arr["h5_id"] = h5_id
        arr["patch_id"] = np.arange(n, dtype=np.int32)
        arr["global_idx"] = np.arange(offset, offset + n, dtype=np.int64)
        mmap[offset:offset+n] = arr
        offset += n

    mmap.flush()
    del mmap
    print(f"[INFO] big_index.npy done: {index_path}")


# ============================================================
# 主入口
# ============================================================
def merge_all(
    npy_list_file,
    shard_dir,
    tmp_dir,
    workers=32,
    total_machines=1,
    machine_rank=0,
    index_out="big_index.npy",
    map_out="h5id_to_npypath.txt"
):
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # =========== Step 1: 读取 npy 列表 ===========
    with open(npy_list_file) as f:
        npy_files = [x.strip() for x in f if x.strip()]

    print(f"[INFO] Total npy files: {len(npy_files)}")

    # =========== Step 2: 计算 patch 数 ===========
    n_patches = []
    for p in tqdm(npy_files, desc="scan counts"):
        try:
            arr = np.load(p, mmap_mode="r")
            n_patches.append(arr.shape[0])
        except:
            n_patches.append(0)

    # =========== Step 3: global offsets ===========
    global_offsets = list(accumulate([0] + n_patches))[:-1]
    total_patches = sum(n_patches)
    print(f"[INFO] Total patches = {total_patches:,}")

    # =========== Step 4: 保存 h5id→path 映射 ===========
    with open(map_out, "w") as f:
        for i, p in enumerate(npy_files):
            f.write(f"{i}\t{p}\n")

    # =========== Step 5: 自动构建 big_index.npy ===========
    build_big_index(n_patches, index_out)

    # =========== Step 6: 分配每个 small npy 属于哪个 shard ===========
    total_shards = (total_patches + SHARD_CAP - 1) // SHARD_CAP
    print(f"[INFO] Total shards = {total_shards}")

    shard_tasks = [[] for _ in range(total_shards)]

    for i in range(len(npy_files)):
        gid = global_offsets[i]
        shard_id = gid // SHARD_CAP
        shard_tasks[shard_id].append(i)

    # =========== Step 7: 自动平均分配 shard ===========
    my_shards = [s for s in range(total_shards) if s % total_machines == machine_rank]

    print(f"[INFO] Machine {machine_rank}/{total_machines} will process shards: {my_shards}")

    # =========== Step 8: 多进程处理自己负责的 shards ===========
    args_list = []
    for sid in my_shards:
        args_list.append((sid, shard_tasks[sid], npy_files, global_offsets, str(shard_dir)))

    with Pool(workers) as pool:
        list(pool.starmap(process_shard, args_list))

    print("[INFO] All shards done.")


def list_npy_files(root_dir, out_txt):
    """遍历目录下所有 .npy 文件并保存到 txt 中"""
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f, tqdm(desc="Scanning .npy files") as pbar:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(".npy"):
                    abs_path = os.path.abspath(os.path.join(dirpath, fname))
                    f.write(abs_path + "\n")
                    pbar.update(1)
    print(f"Done! List saved to {out_txt}")

# ===================================================================
# 可直接运行（你可以修改为自己的路径）
# ===================================================================
if __name__ == "__main__":
    list_npy_files(root_dir='/mnt/local08/data_new', 
                   out_txt="/mnt/local09/train/crb/data/h5_data_local08/npy_all2.txt")
    npy_list_file = '/mnt/local09/train/crb/data/h5_data_local08/npy_all2.txt'
    base_dir = '/mnt/local08/data_new'
    shard_dir = os.path.join(base_dir, 'shard')
    index_out= os.path.join(base_dir, 'big_index.npy')
    map_out=os.path.join(base_dir, 'h5id_to_npypath.txt')
    tmp_dir=os.path.join(base_dir, 'state')
    merge_all(
        npy_list_file=npy_list_file,
        shard_dir=shard_dir,
        index_out=index_out,
        map_out=map_out,
        tmp_dir=tmp_dir,
        workers=64
    )
