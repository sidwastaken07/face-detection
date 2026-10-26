"""5-point face alignment to the ArcFace 112×112 template (similarity transform, Umeyama)."""
from __future__ import annotations

import numpy as np
from PIL import Image

# Left eye, right eye, nose tip, left mouth corner, right mouth corner (x, y) in a 112×112 crop.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    np.float64,
)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation) mapping src→dst.
    Returns a 2×3 matrix M with dst ≈ M @ [x, y, 1]."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rot = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / s_c.var(0).sum()
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - scale * rot @ mu_s
    return m


def align_face(img: Image.Image, landmarks: np.ndarray, size: int = 112) -> Image.Image:
    """Warp `img` so its 5 landmarks land on the ArcFace template (scaled to `size`)."""
    template = ARCFACE_TEMPLATE * (size / 112.0)
    m = similarity_transform(landmarks, template)
    # PIL's AFFINE takes the inverse map (output pixel -> input pixel).
    full = np.vstack([m, [0, 0, 1]])
    inv = np.linalg.inv(full)[:2].ravel()
    return img.transform((size, size), Image.AFFINE, tuple(inv), resample=Image.BILINEAR)


def center_crop(img: Image.Image, frac: float = 0.5, size: int = 112) -> Image.Image:
    """Fallback when no face is detected: central square crop (LFW faces are roughly centred)."""
    w, h = img.size
    c = min(w, h) * frac
    box = ((w - c) / 2, (h - c) / 2, (w + c) / 2, (h + c) / 2)
    return img.resize((size, size), Image.BILINEAR, box=box)
