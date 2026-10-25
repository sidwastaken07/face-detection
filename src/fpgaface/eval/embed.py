"""Batched embedding extraction for uint8 image arrays."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def to_input(images: np.ndarray, device) -> torch.Tensor:
    """uint8 [N, H, W, 3] RGB -> float [N, 3, H, W] in [-1, 1] (project-wide convention)."""
    x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float()
    return (x - 127.5) / 127.5


@torch.no_grad()
def embed_images(model: torch.nn.Module, images: np.ndarray, batch_size: int = 256, device="cpu",
                 flip: bool = True, amp: bool = False) -> np.ndarray:
    """L2-normalised embeddings. With `flip`, the embedding is the sum of the original and the
    horizontally flipped view before normalisation (standard InsightFace evaluation)."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = to_input(images[i : i + batch_size], device)
        with torch.autocast(device_type=torch.device(device).type, enabled=amp):
            e = model(x).float()
            if flip:
                e = e + model(torch.flip(x, dims=[3])).float()
        out.append(F.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy()
