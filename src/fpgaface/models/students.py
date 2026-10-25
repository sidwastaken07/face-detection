"""FPGA-friendly face recognition students.

* MobileFaceNet-FPGA — MobileFaceNet (Chen et al., 2018) with PReLU -> ReLU (maps to FINN thresholds),
  width multiplier, and the global depthwise conv (GDC) head kept. ~1 M params at width 1.0.
* VGG-style baseline — plain conv3x3/BN/ReLU/max-pool stacks; the easiest network to stream through FINN.

Both output an embedding [N, emb_dim] (un-normalised) and accept 112×112 or 64×64 input.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ops import ConvBN, FloatOps, make_divisible

# (expansion t, out channels c, repeats n, first stride s) — MobileFaceNet paper, Table 1.
MFN_SETTINGS = [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]


class Bottleneck(nn.Module):
    def __init__(self, ops, name: str, cin: int, cout: int, stride: int, t: int):
        super().__init__()
        hidden = cin * t
        self.residual = stride == 1 and cin == cout
        self.block = nn.Sequential(
            ConvBN(ops, f"{name}.expand", cin, hidden, 1),
            ConvBN(ops, f"{name}.dw", hidden, hidden, 3, stride, groups=hidden),
            ConvBN(ops, f"{name}.project", hidden, cout, 1, act=False),
        )

    def forward(self, x):
        return x + self.block(x) if self.residual else self.block(x)


class GDCHead(nn.Module):
    """Global depthwise conv over the final k×k map (learned spatial weighting — better than average
    pooling for aligned faces), then a linear 1×1 projection to the embedding and BN."""

    def __init__(self, ops, cin: int, hidden: int, emb_dim: int, spatial: int):
        super().__init__()
        self.expand = ConvBN(ops, "head.expand", cin, hidden, 1)
        self.gdc = ConvBN(ops, "head.gdc", hidden, hidden, spatial, groups=hidden, act=False)
        self.gdc[0].padding = (0, 0)
        self.proj = ConvBN(ops, "head.proj", hidden, emb_dim, 1, act=False)

    def forward(self, x):
        return torch.flatten(self.proj(self.gdc(self.expand(x))), 1)


class MobileFaceNetFPGA(nn.Module):
    def __init__(self, width: float = 1.0, emb_dim: int = 128, input_size: int = 112, head_dim: int = 512,
                 ops=None):
        super().__init__()
        ops = ops or FloatOps()
        c = lambda v: make_divisible(v * width)
        self.input_quant = ops.input_quant()
        stem = c(64)
        layers = [ConvBN(ops, "stem.conv", 3, stem, 3, 2), ConvBN(ops, "stem.dw", stem, stem, 3, 1, groups=stem)]
        cin = stem
        for si, (t, ch, n, s) in enumerate(MFN_SETTINGS):
            for i in range(n):
                layers.append(Bottleneck(ops, f"stage{si}.{i}", cin, c(ch), s if i == 0 else 1, t))
                cin = c(ch)
        self.features = nn.Sequential(*layers)
        self.head = GDCHead(ops, cin, c(head_dim), emb_dim, input_size // 16)

    def forward(self, x):
        return self.head(self.features(self.input_quant(x)))


class VGGFace(nn.Module):
    def __init__(self, width: float = 1.0, emb_dim: int = 128, input_size: int = 112,
                 channels=(32, 64, 128, 256), convs_per_stage: int = 2, head_dim: int = 512, ops=None):
        super().__init__()
        ops = ops or FloatOps()
        self.input_quant = ops.input_quant()
        layers, cin = [], 3
        for si, ch in enumerate(channels):
            ch = make_divisible(ch * width)
            for i in range(convs_per_stage):
                layers.append(ConvBN(ops, f"stage{si}.{i}", cin, ch, 3))
                cin = ch
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = GDCHead(ops, cin, make_divisible(head_dim * width), emb_dim, input_size // 2 ** len(channels))

    def forward(self, x):
        return self.head(self.features(self.input_quant(x)))


def build_student(arch: str, **kwargs) -> nn.Module:
    if arch == "mobilefacenet":
        return MobileFaceNetFPGA(**kwargs)
    if arch == "vgg":
        return VGGFace(**kwargs)
    raise ValueError(f"unknown student architecture {arch!r}")
