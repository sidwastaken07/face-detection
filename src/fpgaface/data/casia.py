"""CASIA-WebFace (InsightFace `faces_webface_112x112` package) as a PyTorch dataset."""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image

from .recordio import IndexedRecordReader, unpack


def _first(label) -> float:
    return float(label[0]) if isinstance(label, np.ndarray) else float(label)


def image_keys(reader: IndexedRecordReader) -> list[int]:
    """Keys of image records. InsightFace packs a header at key 0 whose label holds
    (first identity-record key, end key); image records are keys 1 .. label[0]-1."""
    header = reader.read_header(0)
    if header.flag > 0:
        return list(range(1, int(header.label[0])))
    return reader.keys()


def scan_labels(reader: IndexedRecordReader, keys: list[int]) -> np.ndarray:
    """Read only the record headers to get every image's identity label (~1 min for CASIA)."""
    labels = np.empty(len(keys), np.int64)
    for i, k in enumerate(keys):
        labels[i] = int(_first(reader.read_header(k).label))
        if i % 100_000 == 0 and i:
            print(f"  labels {i}/{len(keys)}", flush=True)
    return labels


def load_or_scan_labels(rec_path: str, cache_path: str | None = None) -> tuple[list[int], np.ndarray]:
    reader = IndexedRecordReader(rec_path)
    keys = image_keys(reader)
    if cache_path and os.path.exists(cache_path):
        labels = np.load(cache_path)
        if len(labels) == len(keys):
            return keys, labels
    labels = scan_labels(reader, keys)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, labels)
    return keys, labels


def identity_subset(labels: np.ndarray, num_ids: int, seed: int = 0) -> np.ndarray:
    """Indices of all images belonging to `num_ids` randomly chosen identities, with labels
    remapped to 0..num_ids-1 by the dataset (see `CasiaDataset(indices=...)`)."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.unique(labels), size=num_ids, replace=False)
    return np.flatnonzero(np.isin(labels, ids))


class CasiaDataset:
    """Returns (image, label, index). `index` is the position in the full dataset and is used to
    look up cached teacher embeddings for distillation."""

    def __init__(self, rec_path: str, labels_cache: str | None = None, transform=None,
                 indices: np.ndarray | None = None):
        self.reader = IndexedRecordReader(rec_path)
        self.keys, all_labels = load_or_scan_labels(rec_path, labels_cache)
        self.indices = np.arange(len(self.keys)) if indices is None else np.asarray(indices)
        uniq = np.unique(all_labels[self.indices])
        remap = np.full(all_labels.max() + 1, -1, np.int64)
        remap[uniq] = np.arange(len(uniq))
        self.labels = remap[all_labels[self.indices]]
        self.num_classes = len(uniq)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def load_image(self, i: int) -> Image.Image:
        _, payload = unpack(self.reader.read_idx(self.keys[self.indices[i]]))
        return Image.open(io.BytesIO(payload)).convert("RGB")

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), int(self.indices[i])
