"""Margin-based softmax heads (ArcFace, CosFace, AdaFace) and distillation losses."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginHead(nn.Module):
    """Normalised-weight classifier with an additive margin on the target logit.

    * arcface: cos(θ + m)                      (Deng et al., 2019)
    * cosface: cos(θ) − m                      (Wang et al., 2018)
    * adaface: margin adapted per sample by its feature norm (image-quality proxy) (Kim et al., 2022)
    """

    def __init__(self, emb_dim: int, num_classes: int, kind: str = "arcface", s: float = 64.0,
                 m: float | None = None, h: float = 0.333, t_alpha: float = 0.01):
        super().__init__()
        self.kind = kind
        self.s = s
        self.m = m if m is not None else {"arcface": 0.5, "cosface": 0.35, "adaface": 0.4}[kind]
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.h = h
        self.t_alpha = t_alpha
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = emb.float()
        norms = emb.norm(dim=1, keepdim=True)
        cos = F.linear(emb / norms.clamp_min(1e-6), F.normalize(self.weight.float(), dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        onehot = F.one_hot(labels, cos.shape[1]).bool()

        if self.kind == "cosface":
            logits = torch.where(onehot, cos - self.m, cos)
        elif self.kind == "arcface":
            sin = (1 - cos**2).sqrt()
            phi = cos * math.cos(self.m) - sin * math.sin(self.m)
            # Keep the target logit monotonic when θ + m > π (InsightFace's "hard" margin fallback).
            phi = torch.where(cos > math.cos(math.pi - self.m), phi, cos - math.sin(math.pi - self.m) * self.m)
            logits = torch.where(onehot, phi, cos)
        elif self.kind == "adaface":
            safe = norms.detach().clamp(0.001, 100)
            if self.training:
                with torch.no_grad():
                    self.batch_mean.mul_(1 - self.t_alpha).add_(self.t_alpha * safe.mean())
                    self.batch_std.mul_(1 - self.t_alpha).add_(self.t_alpha * safe.std())
            scaler = ((safe - self.batch_mean) / (self.batch_std + 1e-3) * self.h).clamp(-1, 1)
            theta = cos.acos()
            theta = torch.where(onehot, (theta - self.m * scaler).clamp(1e-3, math.pi - 1e-3), theta)
            logits = torch.where(onehot, theta.cos() - (self.m + self.m * scaler), theta.cos())
        else:
            raise ValueError(self.kind)
        return logits * self.s


class EmbeddingDistill(nn.Module):
    """Cosine distillation to cached teacher embeddings through a small linear projector (training
    only; the deployed student keeps its own 128-d embedding)."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False) if student_dim != teacher_dim else nn.Identity()

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return (1 - F.cosine_similarity(self.proj(student.float()), teacher.float(), dim=1)).mean()


def relational_distill(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match the in-batch cosine-similarity matrices (dimension-free alternative to projection)."""
    s = F.normalize(student.float(), dim=1)
    t = F.normalize(teacher.float(), dim=1)
    return F.mse_loss(s @ s.T, t @ t.T)
