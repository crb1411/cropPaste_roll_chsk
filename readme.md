# CropPaste Roll + CH_SK Notes

This document summarizes the custom changes around legacy cropPaste/shift (roll)
and CH_SK configuration.

## Legacy cropPaste + shift (roll) pipeline

- Student uses legacy-augmented inputs; teacher uses raw inputs (no mask/roll).
- Collate attaches per-view masks for legacy inputs: `cropPaste_mask` and
  `shift_mask` with shape `[B, N]` boolean.
- Student backbone receives those masks (disabled when distillation is on).
- Patch-roll loss samples a subset of samples and patches per sample
  (iBOT-style):
  - `patch_shuffle_patch_probability` controls how many samples per batch
    participate.
  - `patch_shuffle_patch_min_max` defines the per-sample mask ratio range.
  - Selected patches exclude masked tokens from collate (`shift_mask`).
- Teacher patch tokens are aligned to the shifted order via inverse permutation
  from `shift_info` before computing roll patch loss.
- CLS loss is unchanged and computed on the two shift views.

Related code:
- `dinov3/data/collate.py`
- `dinov3/new_train/train/ssl_crop_roll.py`

## Config knobs

`dinov3/configs/crop_rollv1.yaml`:

```yaml
legacy_augmentor:
  patch_shuffle_patch_probability: 0.3
  patch_shuffle_patch_min_max: [0.1, 0.5]
```

Notes:
- `patch_shuffle_patch_ratio` is treated as a deprecated fallback when
  `patch_shuffle_patch_probability` is missing.

## Legacy outputs for losses

- `cropresize_cls_logits_resized`, `cropresize_cls_logits_original`
- `patchshuffle_cls_logits_resized`, `patchshuffle_cls_logits_original`
- `patchshuffle_patch_logits_resized`, `patchshuffle_patch_logits_original`
- `patchshuffle_masks_resized`, `patchshuffle_masks_original`

## CH_SK (Cumulative History Sinkhorn)

- Implementation: `dinov3/loss/ch_sk.py` (used by iBOT-style losses).
- Config keys expected in code: `ch_sk.*` (for example `ch_sk.boost_alpha`).
- Current configs define `chsk.*` (for example in
  `dinov3/configs/crop_rollv1.yaml`), so align the key name if you want CH_SK to
  read it.
