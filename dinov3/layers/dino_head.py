# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, no_last_layer=False, only_last_layer=False):
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        if not no_last_layer:
            x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
    
from contextlib import contextmanager

@contextmanager
def freeze_module(module: torch.nn.Module):
    """
    临时冻结一个 module 的所有参数：
    - 保持它们是 DTensor / FSDP 参数
    - 但不产生梯度
    """
    old_flags = []
    for p in module.parameters():
        old_flags.append(p.requires_grad)
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(module.parameters(), old_flags):
            p.requires_grad_(flag)

def frozen_dino_head_forward(head: DINOHead, x):
    """
    完全等价于 head(x)，但：
    - head 参数不更新
    - x 仍然反传
    - 100% 兼容 FSDP / DTensor / torch.compile
    """
    with freeze_module(head):
        return head(x)
