from enum import Enum
import logging
import numpy as np

logger = logging.getLogger("dinov3")
class DatasetType(Enum):

    SVS_DATE = 0
    H5_DATE = 1
    PATCHES_DATA=2
    IMGNET_DATA= 3
    NPY_DATA = 4
    IMGNET_DATA_2 = 5

class CombinedDataset:
    """
    dataset_list: [dataset_A, dataset_B, dataset_C, ...]
    """
    def __init__(self, dataset_list):
        self.datasets = dataset_list
        self.lengths = [len(d) for d in dataset_list]
        self.print_ = 1

    def __getitem__(self, index_tuple):
        if self.print_:
            logger.info(f'准备第 {self.print_} 次加载数据')
        ds_idx, sample_idx = index_tuple
        data_idx = self.datasets[ds_idx][sample_idx]
        if self.print_:
            logger.info(f'第 {self.print_} 次加载数据， dataset_{ds_idx}: {sample_idx}成功')
            self.print_ += 1
            if self.print_ >=100:
                self.print_ = 0
        return data_idx

    def __len__(self):
        # 真实使用 infinite sampler，不需要这个值，随便返回就行
        return sum(self.lengths)

class CombinedSampler:
    """
    dataset_samplers: [sampler_A, sampler_B, sampler_C]
    ratios:           [0.4, 0.3, 0.3]
    """
    def __init__(self, dataset_samplers, ratios):
        self.dataset_samplers = dataset_samplers
        self.ratios = ratios

    def __iter__(self):
        
        rng = np.random.default_rng()

        # 变成可重启的迭代器
        logger.info(f'Begin create combined sampler: {len(self.dataset_samplers)} Samplers')
        iters = [iter(s) for s in self.dataset_samplers]
        logger.info(f'Create combined samplern done')
        while True:
            # ---- 1) 按比例选 dataset ----
            ds_idx = rng.choice(len(self.dataset_samplers), p=self.ratios)

            # ---- 2) 对应 sampler 给出实际 index ----
            try:
                sample_idx = next(iters[ds_idx])
            except StopIteration:
                # 如果某个 sampler 用尽 → 重新启动
                iters[ds_idx] = iter(self.dataset_samplers[ds_idx])
                sample_idx = next(iters[ds_idx])

            # ---- 3) 返回 (dataset_idx, sample_idx)
            yield (ds_idx, sample_idx)

