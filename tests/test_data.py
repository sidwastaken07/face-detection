import numpy as np

from fpgaface.data.bins import find_benchmarks, load_bin
from fpgaface.data.casia import CasiaDataset, identity_subset
from fpgaface.data.recordio import IndexedRecordReader, unpack
from fpgaface.data.transforms import ToNormalizedTensor


def test_recordio_header_record(fake_casia):
    rec, labels = fake_casia
    reader = IndexedRecordReader(rec)
    header, payload = unpack(reader.read_idx(0))
    assert header.flag == 2 and int(header.label[0]) == len(labels) + 1
    assert payload == b""
    assert reader.read_header(3).label == labels[2]


def test_casia_dataset(fake_casia, tmp_path):
    rec, labels = fake_casia
    ds = CasiaDataset(rec, str(tmp_path / "labels.npy"), transform=ToNormalizedTensor())
    assert len(ds) == len(labels) and ds.num_classes == 5
    x, y, idx = ds[5]
    assert x.shape == (3, 112, 112) and -1 <= x.min() and x.max() <= 1
    assert y == labels[5] and idx == 5
    # Second construction uses the cached labels.
    assert (tmp_path / "labels.npy").exists()
    assert np.array_equal(CasiaDataset(rec, str(tmp_path / "labels.npy")).labels, labels)


def test_casia_subset_remaps_labels(fake_casia):
    rec, labels = fake_casia
    sub = identity_subset(labels, 2, seed=0)
    ds = CasiaDataset(rec, indices=sub)
    assert ds.num_classes == 2 and set(ds.labels) == {0, 1} and len(ds) == 8
    assert ds[0][2] == sub[0]


def test_load_bin(fake_bin, tmp_path):
    path, issame = fake_bin
    imgs, same = load_bin(path)
    assert imgs.shape == (2 * len(issame), 112, 112, 3) and imgs.dtype == np.uint8
    assert np.array_equal(same, issame)
    assert find_benchmarks(str(tmp_path)) == {"lfw": path}
