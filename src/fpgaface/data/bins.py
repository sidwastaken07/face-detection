"""InsightFace verification benchmark files (lfw.bin, cfp_fp.bin, agedb_30.bin, calfw.bin, cplfw.bin).

Each .bin is a pickle of (list of encoded images, list of is-same flags). Images come in
consecutive pairs: (2k, 2k+1) is pair k.
"""
from __future__ import annotations

import io
import os
import pickle

import numpy as np
from PIL import Image

BENCHMARKS = ("lfw", "cfp_fp", "agedb_30", "calfw", "cplfw")


def load_bin(path: str, size: int = 112) -> tuple[np.ndarray, np.ndarray]:
    """Returns uint8 RGB images [2N, size, size, 3] and bool is-same flags [N]."""
    with open(path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")
    images = np.empty((len(bins), size, size, 3), np.uint8)
    for i, b in enumerate(bins):
        img = Image.open(io.BytesIO(bytes(b))).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        images[i] = np.asarray(img)
    return images, np.asarray(issame, bool)


def find_benchmarks(root: str, names=BENCHMARKS) -> dict[str, str]:
    """Search `root` recursively for `<name>.bin` files."""
    found = {}
    for dirpath, _, files in os.walk(root):
        for name in names:
            if f"{name}.bin" in files and name not in found:
                found[name] = os.path.join(dirpath, f"{name}.bin")
    return found
