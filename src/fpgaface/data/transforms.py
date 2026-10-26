from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class ToNormalizedTensor:
    """PIL RGB -> float tensor [3, H, W] in [-1, 1] (project-wide input convention)."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(img, np.uint8).copy()).permute(2, 0, 1).float()
        return (x - 127.5) / 127.5
