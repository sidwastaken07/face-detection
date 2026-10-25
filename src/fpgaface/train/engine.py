"""Training loop pieces shared by FP32 student training (Phase 2) and QAT (Phase 3)."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from ..data.bins import load_bin
from ..eval.embed import embed_images
from ..eval.verification import evaluate_pairs
from .losses import EmbeddingDistill, MarginHead, relational_distill


# ----------------------------------------------------------------------------- distributed
def setup_distributed() -> tuple[int, int, torch.device]:
    """(rank, world size, device). Multi-GPU when launched with torchrun (e.g. Kaggle 2×T4)."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return rank, world, torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_main(rank: int) -> bool:
    return rank == 0


# ----------------------------------------------------------------------------- model wrapper
class FaceTrainModule(nn.Module):
    """Student + margin head + distillation projector in one module so DDP syncs all of them."""

    def __init__(self, student: nn.Module, emb_dim: int, num_classes: int, loss_cfg: dict, teacher_dim: int):
        super().__init__()
        self.student = student
        self.head = MarginHead(emb_dim, num_classes, loss_cfg.get("head", "arcface"), loss_cfg.get("s", 64.0),
                               loss_cfg.get("m"))
        self.kd = loss_cfg.get("kd", "proj_cos")
        self.kd_weight = float(loss_cfg.get("kd_weight", 0.0)) if self.kd else 0.0
        self.distill = EmbeddingDistill(emb_dim, teacher_dim) if self.kd == "proj_cos" else None

    def forward(self, x, y, teacher=None):
        emb = self.student(x)
        ce = nn.functional.cross_entropy(self.head(emb, y), y)
        kd = torch.zeros((), device=x.device)
        if self.kd_weight > 0 and teacher is not None:
            kd = self.distill(emb, teacher) if self.distill is not None else relational_distill(emb, teacher)
        return ce + self.kd_weight * kd, ce.detach(), kd.detach()


def param_groups(module: nn.Module, weight_decay: float):
    """No weight decay on BN parameters and biases."""
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(total_steps: int, warmup_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


# ----------------------------------------------------------------------------- evaluation
class BenchEvaluator:
    """Verification accuracy on InsightFace .bin benchmarks (images preloaded once)."""

    def __init__(self, bin_paths: dict[str, str], names: list[str], input_size: int = 112):
        self.data = {n: load_bin(bin_paths[n]) for n in names if n in bin_paths}
        self.input_size = input_size

    def __call__(self, model: nn.Module, device, amp: bool = True, batch_size: int = 512) -> dict:
        out = {}
        for name, (imgs, issame) in self.data.items():
            if self.input_size != imgs.shape[1]:
                imgs = _resize_batch(imgs, self.input_size)
            emb = embed_images(model, imgs, batch_size, device, flip=True, amp=amp and device.type == "cuda")
            out[name] = evaluate_pairs(emb, issame)
        if out:
            out["mean_acc"] = float(np.mean([v["acc"] for k, v in out.items()]))
        return out


def _resize_batch(imgs: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(i).resize((size, size), Image.BILINEAR)) for i in imgs])


# ----------------------------------------------------------------------------- checkpointing
def save_checkpoint(path: str, **state) -> None:
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a killed session never leaves a corrupt checkpoint


def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.parallel.DistributedDataParallel) else m


class TimeGuard:
    """Stops training cleanly before Kaggle's 12-hour session limit."""

    def __init__(self, max_hours: float):
        self.start = time.time()
        self.limit = max_hours * 3600

    def elapsed(self) -> float:
        return time.time() - self.start

    def expired(self) -> bool:
        return self.elapsed() > self.limit

    def can_fit(self, seconds: float) -> bool:
        return self.elapsed() + seconds * 1.1 < self.limit


class JsonlLogger:
    def __init__(self, path: str, enabled: bool = True):
        self.path, self.enabled = path, enabled

    def __call__(self, **row) -> None:
        if self.enabled:
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
