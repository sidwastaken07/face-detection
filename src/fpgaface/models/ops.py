"""Layer factories. Student models build every conv/activation through an `Ops` object so the same
architecture code yields the FP32 model (here) or the Brevitas quantised model (Phase 3), including
per-layer bit-widths for mixed precision (layers are addressed by their module name)."""
from __future__ import annotations

import torch.nn as nn


class FloatOps:
    def conv(self, name: str, cin: int, cout: int, k: int, stride: int = 1, groups: int = 1) -> nn.Module:
        return tag(nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), name)

    def act(self, name: str, channels: int) -> nn.Module:
        return nn.ReLU(inplace=True)

    def input_quant(self) -> nn.Module:
        return nn.Identity()


def tag(module: nn.Module, name: str) -> nn.Module:
    """Attach the architecture-level layer name used for per-layer bit-widths and budgets."""
    module.fpga_name = name
    return module


class ConvBN(nn.Sequential):
    """conv -> BN [-> act]. BN folds into FINN thresholds; `act=False` gives a linear (projection) layer."""

    def __init__(self, ops, name: str, cin: int, cout: int, k: int, stride: int = 1, groups: int = 1,
                 act: bool = True):
        layers = [ops.conv(name, cin, cout, k, stride, groups), nn.BatchNorm2d(cout)]
        if act:
            layers.append(ops.act(name + ".act", cout))
        super().__init__(*layers)


def make_divisible(v: float, divisor: int = 8) -> int:
    """Round channel counts to multiples of 8 (keeps FINN PE/SIMD folding factors flexible)."""
    new = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new + divisor if new < 0.9 * v else new
