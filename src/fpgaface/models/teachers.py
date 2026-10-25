"""Frozen FP32 teacher networks.

Convention for every model in this project: input is an RGB float tensor [N, 3, H, W] normalised as
(x - 127.5) / 127.5, i.e. in [-1, 1]; output is an (un-normalised) embedding [N, D].
"""
from __future__ import annotations

import os
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
IRESNET50_ONNX = "w600k_r50.onnx"


class FaceNetTeacher(nn.Module):
    """FaceNet InceptionResnetV1 pretrained on VGGFace2 (facenet-pytorch), 512-d.
    Applies FaceNet's own standardisation ((x - 127.5) / 128) and runs at its native 160×160."""

    def __init__(self, input_size: int = 160):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1

        self.net = InceptionResnetV1(pretrained="vggface2").eval()
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * (127.5 / 128.0)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.net(x)


def download_buffalo_l(cache_dir: str) -> str:
    """Download InsightFace's buffalo_l model pack and return the path to the iResNet-50 ArcFace
    recogniser (trained on WebFace600K). Licence: non-commercial research use."""
    path = os.path.join(cache_dir, IRESNET50_ONNX)
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        zpath = os.path.join(cache_dir, "buffalo_l.zip")
        urllib.request.urlretrieve(BUFFALO_L_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extract(IRESNET50_ONNX, cache_dir)
        os.remove(zpath)
    return path


class ArcFaceTeacher(nn.Module):
    """InsightFace iResNet-50 ArcFace (ONNX) converted to PyTorch with onnx2torch so it runs on GPU
    with AMP. Expects 112×112 ArcFace-aligned RGB input normalised to [-1, 1] (same as our convention)."""

    def __init__(self, onnx_path: str | None = None, cache_dir: str = "~/.cache/fpgaface"):
        super().__init__()
        from onnx2torch import convert

        onnx_path = onnx_path or download_buffalo_l(os.path.expanduser(cache_dir))
        self.net = convert(onnx_path).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 112:
            x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        return self.net(x)


def build_teacher(name: str, **kwargs) -> nn.Module:
    if name == "facenet":
        return FaceNetTeacher(**kwargs)
    if name in ("arcface_r50", "iresnet50"):
        return ArcFaceTeacher(**kwargs)
    raise ValueError(f"unknown teacher {name!r}")
