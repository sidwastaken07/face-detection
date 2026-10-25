import io
import pickle

import numpy as np
import pytest
from PIL import Image

from fpgaface.data.recordio import IRHeader, RecordWriter, pack


def _jpeg(rng, size=112) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def fake_casia(tmp_path):
    """InsightFace-style .rec: header at key 0, images at 1..N, identity records after."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(5), 4)  # 5 identities x 4 images
    rec = tmp_path / "train.rec"
    n = len(labels)
    with RecordWriter(str(rec)) as w:
        w.write_idx(0, pack(IRHeader(0, [n + 1, n + 1 + 5], 0, 0), b""))
        for i, y in enumerate(labels, start=1):
            w.write_idx(i, pack(IRHeader(0, float(y), i, 0), _jpeg(rng)))
        for c in range(5):
            start = 1 + 4 * c
            w.write_idx(n + 1 + c, pack(IRHeader(0, [start, start + 4], 0, 0), b""))
    return str(rec), labels


@pytest.fixture
def fake_bin(tmp_path):
    rng = np.random.default_rng(1)
    issame = [True, False] * 10
    bins = [_jpeg(rng) for _ in range(2 * len(issame))]
    path = tmp_path / "lfw.bin"
    with open(path, "wb") as f:
        pickle.dump((bins, issame), f)
    return str(path), np.array(issame)
