# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2

import sys
from PIL import Image
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from functools import partial

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform
from dinov3.data.transforms import WSI_MEAN, WSI_STD
from dinov3.data.legacy_augment import Augmentor, AugmentSwitch, save_index, create_dir_time


logger = logging.getLogger("dinov3")


def _build_legacy_augmentor(
    use_legacy: bool,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    switch: Optional[AugmentSwitch]=None,
) -> Optional[Augmentor]:
    if not use_legacy:
        return None
    return Augmentor(
        switches=switch or AugmentSwitch(),
        mean=mean,
        std=std,
        use_normalize=True,
    )


def _image_to_tensor01(image) -> torch.Tensor:
    """PIL RGB -> float32 [C,H,W], in [0,1]."""
    if isinstance(image, torch.Tensor):
        x = image
        if x.dtype != torch.float32:
            x = v2.functional.to_dtype(x, torch.float32, scale=True)
        return x
    x = v2.functional.pil_to_tensor(image)                      # uint8
    x = v2.functional.to_dtype(x, torch.float32, scale=True)  # 0..1
    return x


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        use_legacy_augmentor: bool = False,
        legacy_augmentor_switch: Optional[AugmentSwitch] = None,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std
        self._legacy_augmentor = _build_legacy_augmentor(
            use_legacy_augmentor,
            mean,
            std,
            switch=legacy_augmentor_switch,
        )

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)
        self.global_crop_max_size = global_crop_max_size
        self.resize_legacy_raw = v2.Resize(
            global_crop_max_size,
            interpolation=v2.InterpolationMode.BILINEAR,
        )

        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v2.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])

        self.to_img = v2.ToImage()
    def __call__(self, image):
        image = self.to_img(image) if not isinstance(image, Image.Image) else image
        
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        if self._legacy_augmentor is not None:
            try:
                # 随机选取 img1_base or img2_base 0.7, 0.3 
                img_ = im1_base if np.random.rand() < 0.7 else im2_base
                tensor_input_resized = _image_to_tensor01(img_)
                h, w = v2.functional.get_size(image)
                if h != self.global_crop_max_size or w != self.global_crop_max_size:
                    image = self.resize_legacy_raw(image)
                tensor_input_raw = _image_to_tensor01(image=image)
                output["legacy_aug"] = self._legacy_augmentor(tensor_input_raw)
                output["legacy_aug_resized"] = self._legacy_augmentor(tensor_input_resized)
            except Exception as exc:
                logger.warning("legacy augmentation failed: %s", exc)

        return output

if __name__ == "__main__":
    img = torch.rand(3, 224, 224)
    from omegaconf import DictConfig, OmegaConf
    cfg = OmegaConf.load("/data/work/git_proj/dinov3/dinov3/configs/ssl_default_config.yaml")
    # sys.path.append(str(REPO_ROOT.parent / "resize_2"))
    from dinov3.new_train.data.imgnet import ImageNetResizeDataset
    
    from dinov3.data import (
        collate_data_and_cast,
        MaskingGenerator,
    )
    mask_generator = MaskingGenerator(
        input_size=(14, 14),
        max_num_patches=0.5 * 196,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.compute_precision.param_dtype],
        n_tokens=196,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=None,
    )

    a = DataAugmentationDINO(
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
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
            use_legacy_augmentor=True,
            legacy_augmentor_switch=None,
        )
    
    
    data = ImageNetResizeDataset(
            index_npy='/mnt/data/imagenet_1k/data_new/imagenet_1k.npy',
            size = 224,
            transform=a,
        )
    # for data_idx in data:
    #     out_dir = create_dir_time(
    #             base_dir='/data/work/output_dir/aug_out',
    #             prefix='aug'                   
    #         )
    #     save_index(data_idx, out_dir, prefix='augVis')
        
    # pass
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=8,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    from tqdm import tqdm
    for data_ in tqdm(dataloader):
        # print(data_)
        pass
    
