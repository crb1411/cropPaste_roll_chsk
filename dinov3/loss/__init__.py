# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .dino_clstoken_loss import DINOLoss
# from .dino_cls_cache_loss import DINOLoss_skcache
from .dino_cls_loss_cache_global import DINOLoss_skcache
from .gram_loss import GramLoss
from .ibot_patch_loss import iBOTPatchLoss
from .koleo_loss import KoLeoLoss, KoLeoLossDistributed