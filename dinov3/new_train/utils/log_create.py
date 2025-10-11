import re
from pathlib import Path
import os
from datetime import datetime

def next_log_dir(base_dir, prefix="log", create=False):
    """
    在 base_dir 下找到下一个自增的日志目录，例如 log1、log2、log3...
    返回新目录的绝对路径。
    
    - base_dir: 日志根目录
    - prefix:   目录前缀（默认 "log"）
    - create:   是否实际创建目录（默认 True）
    """
    base = Path(base_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_n = 0
    for p in base.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))

    n = max_n + 1
    new_dir = base / f"{prefix}{n}"
    if create:
        new_dir.mkdir(parents=True, exist_ok=False)
    return str(new_dir)




def create_timestamped_subfolder(base_dir: str, prefix: str, create=False) -> str:
    """
    在 base_dir 下创建一个以 prefix + 当前日期时间 为名字的子文件夹

    Args:
        base_dir (str): 基础文件夹路径
        prefix (str): 子文件夹前缀

    Returns:
        str: 创建好的子文件夹路径
    """
    # 格式化当前日期时间
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{prefix}_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)

    # 创建文件夹
    if create:
        os.makedirs(folder_path, exist_ok=True)
    return folder_path

def creat_subdir(base_dir, prefix="log", create=False, time=False):
    if time:
        return create_timestamped_subfolder(base_dir=base_dir, prefix=prefix, create=create)
    else:
        return next_log_dir(base_dir=base_dir, prefix=prefix, create=create)