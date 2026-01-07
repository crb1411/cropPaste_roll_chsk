#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from multiprocessing import Pool, get_context

# ---- 单个目录的扫描函数（子进程执行） ----
def _scan_one_dir(args):
    dirpath, exts, followlinks = args
    hits = []
    try:
        # 使用 scandir 比 listdir/stat 快很多
        with os.scandir(dirpath) as it:
            for entry in it:
                try:
                    if not entry.is_file(follow_symlinks=followlinks):
                        continue
                    name = entry.name
                    # 后缀判断（不区分大小写）
                    for ext in exts:
                        if name.endswith(ext) or name.endswith(ext.upper()):
                            # entry.path 已是绝对/相对，统一转绝对
                            hits.append(os.path.abspath(entry.path))
                            break
                except OSError:
                    # 某些条目可能坏链路/权限问题，跳过
                    continue
    except (PermissionError, FileNotFoundError, NotADirectoryError, OSError):
        # 目录不可读/刚被删除等，跳过
        pass
    return hits

def walk_dirs(root, followlinks=False):
    # 生成器，逐个产出目录路径；onerror 仅打印，不中断
    def _onerror(e):
        print(f"[walk warning] {e}", file=sys.stderr)
    for dirpath, dirnames, _ in os.walk(root, followlinks=followlinks, onerror=_onerror):
        yield dirpath

def main():
    parser = argparse.ArgumentParser(
        description="Multiprocess scan for .sdpc and .svs files and write absolute paths to a txt."
    )
    parser.add_argument("--root", help="根目录（要遍历的超大目录）", default='/mnt/local10/bundles')
    parser.add_argument("--out_txt", help="输出的 txt 文件路径", default='/mnt/local09/train/crb/data/npydata_local10/h5_all.txt')
    parser.add_argument("--workers", type=int, default=156,
                        help="进程数，默认=CPU核数")
    parser.add_argument("--follow-links", action="store_true",
                        help="是否跟随符号链接（谨慎，可能导致循环）")
    parser.add_argument("--batch-write", type=int, default=5000,
                        help="每累计多少条路径落盘一次（默认2000）")
    parser.add_argument("--start-method", choices=["spawn","fork","forkserver"],
                        default="spawn", help="多进程 start method（默认 spawn 更稳）")
    parser.add_argument("--exts", nargs="*", default=[".h5"],
                        help="要匹配的后缀列表（默认 .sdpc .svs）")
    parser.add_argument("--show-progress", action="store_true",
                        help="在stderr打印进度统计（不依赖第三方库）")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_txt = os.path.abspath(args.out_txt)
    exts = tuple(args.exts)
    followlinks = args.follow_links
    batch_write = max(1, int(args.batch_write))

    if not os.path.isdir(root):
        print(f"[error] root 不存在或不是目录: {root}", file=sys.stderr)
        sys.exit(1)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)

    t0 = time.time()
    total_dirs = 0
    total_files = 0
    buffered = []

    # 以只写方式创建/覆盖；如想追加可改为 'a'
    out_f = open(out_txt, "w", encoding="utf-8", buffering=1024*1024)

    try:
        # 使用指定的启动方式创建 Pool
        with get_context(args.start_method).Pool(processes=args.workers) as pool:
            # imap_unordered 支持生成器作为输入（逐步投喂目录），内存占用低
            dir_iter = ((d, exts, followlinks) for d in walk_dirs(root, followlinks=followlinks))
            for hits in pool.imap_unordered(_scan_one_dir, dir_iter, chunksize=32):
                total_dirs += 1
                if hits:
                    buffered.extend(hits)
                    total_files += len(hits)

                # 批量落盘
                if len(buffered) >= batch_write:
                    out_f.write("\n".join(buffered) + "\n")
                    out_f.flush()
                    buffered.clear()

                # 简单进度
                if args.show_progress and (total_dirs % 1000 == 0):
                    dt = time.time() - t0
                    rate = total_dirs / dt if dt > 0 else 0.0
                    print(f"[progress] scanned_dirs={total_dirs}, found_files={total_files}, "
                          f"dirs_per_sec={rate:.1f}", file=sys.stderr)

            # 可能还有尾巴
            if buffered:
                out_f.write("\n".join(buffered) + "\n")
                out_f.flush()
                buffered.clear()

    finally:
        out_f.close()

    dt = time.time() - t0
    print(f"[done] root={root}\n[out ] {out_txt}\n"
          f"dirs_scanned={total_dirs}, files_found={total_files}, "
          f"elapsed={dt:.2f}s", file=sys.stderr)

if __name__ == "__main__":
    main()
