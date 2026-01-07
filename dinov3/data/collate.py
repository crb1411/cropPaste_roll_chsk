# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import random

import torch

from dinov3.data.masking import MaskingGenerator


def _make_mask_generator_for_grid(mask_generator: MaskingGenerator, grid_hw: tuple[int, int]) -> MaskingGenerator:
    base_h, base_w = mask_generator.get_shape()
    if (base_h, base_w) == grid_hw:
        return mask_generator
    base_n = base_h * base_w
    target_n = grid_hw[0] * grid_hw[1]
    if base_n <= 0 or target_n <= 0:
        return mask_generator
    max_num_patches = None
    if mask_generator.max_num_patches is not None:
        ratio = mask_generator.max_num_patches / base_n
        max_num_patches = ratio * target_n
    min_num_patches = mask_generator.min_num_patches
    if max_num_patches is not None:
        min_num_patches = min(min_num_patches, max_num_patches)
    min_aspect = math.exp(mask_generator.log_aspect_ratio[0])
    max_aspect = math.exp(mask_generator.log_aspect_ratio[1])
    return MaskingGenerator(
        input_size=grid_hw,
        max_num_patches=max_num_patches,
        min_num_patches=min_num_patches,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
    )


def _build_collated_masks(
    *,
    batch_size: int,
    n_tokens: int,
    mask_ratio_tuple,
    mask_probability,
    mask_generator,
    random_circular_shift,
):
    if batch_size <= 0 or n_tokens is None or n_tokens <= 0 or mask_generator is None:
        return None
    n_samples_masked = int(batch_size * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(n_tokens * prob_max)))
        if random_circular_shift:
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
    for _ in range(n_samples_masked, batch_size):
        masks_list.append(torch.BoolTensor(mask_generator(0)))
    random.shuffle(masks_list)
    return torch.stack(masks_list).flatten(1)


def _attach_legacy_masks(
    legacy_aug_batch: dict,
    *,
    patch_hw: tuple[int, int] | None,
    mask_ratio_tuple,
    mask_probability,
    mask_generator,
    random_circular_shift,
):
    if not isinstance(legacy_aug_batch, dict) or patch_hw is None or mask_generator is None:
        return
    patch_h, patch_w = patch_hw
    if patch_h <= 0 or patch_w <= 0:
        return
    for key in ("cropPaste", "shift"):
        value = legacy_aug_batch.get(key)
        if not torch.is_tensor(value):
            continue
        H = int(value.shape[-2])
        W = int(value.shape[-1])
        if H % patch_h != 0 or W % patch_w != 0:
            continue
        grid_hw = (H // patch_h, W // patch_w)
        gen = _make_mask_generator_for_grid(mask_generator, grid_hw)
        masks = _build_collated_masks(
            batch_size=value.shape[0],
            n_tokens=grid_hw[0] * grid_hw[1],
            mask_ratio_tuple=mask_ratio_tuple,
            mask_probability=mask_probability,
            mask_generator=gen,
            random_circular_shift=random_circular_shift,
        )
        if masks is not None:
            legacy_aug_batch[f"{key}_mask"] = masks


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # [n_global_crops, B, ...]
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )  # [n_global_crops, B, ...]
    else:
        collated_gram_teacher_crops = None

    if local_batch_size is not None:
        # multi-distillation case, number of masks is different because the number of samples masked
        # is different of the number of samples passed into the teacher initially
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:  # apply le random circular shift to
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if "legacy_aug" in samples_list[0][0]:
        collate_legacy_aug(samples_list=samples_list, out=out, dtype=dtype, key_='legacy_aug_resized')
        collate_legacy_aug(samples_list=samples_list, out=out, dtype=dtype, key_='legacy_aug')
        patch_hw = None
        if mask_generator is not None:
            grid_h, grid_w = mask_generator.get_shape()
            if grid_h > 0 and grid_w > 0:
                g_h = collated_global_crops.shape[-2]
                g_w = collated_global_crops.shape[-1]
                if g_h % grid_h == 0 and g_w % grid_w == 0:
                    patch_hw = (g_h // grid_h, g_w // grid_w)
        _attach_legacy_masks(
            out.get("legacy_aug_resized", {}),
            patch_hw=patch_hw,
            mask_ratio_tuple=mask_ratio_tuple,
            mask_probability=mask_probability,
            mask_generator=mask_generator,
            random_circular_shift=random_circular_shift,
        )
        _attach_legacy_masks(
            out.get("legacy_aug", {}),
            patch_hw=patch_hw,
            mask_ratio_tuple=mask_ratio_tuple,
            mask_probability=mask_probability,
            mask_generator=mask_generator,
            random_circular_shift=random_circular_shift,
        )
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    return out

def collate_legacy_aug(samples_list, out, dtype=torch.float32, key_='legacy_aug_resized'):
    """
    samples_list: List[List[sample_dict]]
    out: final batch dict to write into
    """

    # 取第一个样本看看有没有 legacy_aug
    first_aug = samples_list[0][0].get(key_, None)
    if first_aug is None:
        return

    # 收集所有样本的 legacy_aug dict
    aug_dicts = [s[0].get(key_, None) for s in samples_list]

    # 收集所有 key（包含 xxx 与 xxx_info）
    all_keys = set()
    for d in aug_dicts:
        if isinstance(d, dict):
            all_keys.update(d.keys())

    legacy_aug_batch = {}

    for key in all_keys:
        # 每个样本的该 key
        values = []
        valid = True

        for d in aug_dicts:
            if not isinstance(d, dict) or key not in d:
                valid = False
                break
            values.append(d[key])

        if not valid:
            continue

        v0 = values[0]

        # ---------- tensor augmentation ----------
        if isinstance(v0, torch.Tensor):
            # expect all tensors same shape
            legacy_aug_batch[key] = torch.stack(
                [v.to(dtype) for v in values], dim=0
            )
            continue

        # ---------- info: dict / list / numeric / tensor ----------
        # info 不能直接 stack，需要逐元素处理
        # 简化方案：保持 list 结构，后续用户自己解析
        if isinstance(v0, (dict, list, tuple, int, float, str)):
            legacy_aug_batch[key] = values
            continue

        # ---------- info 是 tensor 的情况 ----------
        if isinstance(v0, torch.Tensor):
            legacy_aug_batch[key] = torch.stack(values, dim=0)
            continue

        # ---------- fallback ----------
        # 其他类型（如 None），直接存 list
        legacy_aug_batch[key] = values

    out[key_] = legacy_aug_batch



# def get_batch_subset(collated_data_batch, target_bs):
def get_batch_subset(collated_data_batch, divide_by):
    old_bs = collated_data_batch["collated_global_crops"].shape[0] // 2
    target_bs = (old_bs + divide_by - 1) // divide_by
    collated_global_crops = (
        collated_data_batch["collated_global_crops"].unflatten(0, (2, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )
    collated_local_crops = (
        collated_data_batch["collated_local_crops"].unflatten(0, (-1, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )

    masks_old_bs = collated_data_batch["collated_masks"].shape[0] // 2
    masks_target_bs = masks_old_bs // divide_by
    collated_masks = (
        collated_data_batch["collated_masks"]
        .unflatten(0, (2, masks_old_bs))
        .narrow(1, 0, masks_target_bs)
        .flatten(0, 1)
    )
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    while mask_indices_list.shape[0] == 0:
        _unbind = list(collated_data_batch["collated_masks"].unbind(0))
        random.shuffle(_unbind)
        _bind = torch.stack(_unbind, dim=0)
        collated_masks = _bind.unflatten(0, (2, masks_old_bs)).narrow(1, 0, masks_target_bs).flatten(0, 1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    upperbound = collated_data_batch["upperbound"]

    new_batch = {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

    if "global_batch_size" in collated_data_batch.keys():
        new_batch["global_batch_size"] = collated_data_batch["global_batch_size"] // divide_by

    return new_batch
