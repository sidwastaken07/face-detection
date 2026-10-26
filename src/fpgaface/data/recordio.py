"""Minimal reader/writer for MXNet RecordIO (.rec/.idx) files, as used by InsightFace datasets.

Avoids a dependency on MXNet (unmaintained, not installable on recent Python).

Record framing:  uint32 magic | uint32 (cflag << 29 | length) | payload | pad to 4 bytes
Image payload:   IRHeader (uint32 flag, float32 label, uint64 id, uint64 id2)
                 [flag * float32 extra labels]  encoded image bytes
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0xCED7230A
_IR_FORMAT = "<IfQQ"
_IR_SIZE = struct.calcsize(_IR_FORMAT)


@dataclass
class IRHeader:
    flag: int
    label: float | np.ndarray
    id: int
    id2: int


def unpack(record: bytes) -> tuple[IRHeader, bytes]:
    flag, label, id_, id2 = struct.unpack(_IR_FORMAT, record[:_IR_SIZE])
    payload = record[_IR_SIZE:]
    if flag > 0:
        label = np.frombuffer(payload, np.float32, flag).copy()
        payload = payload[flag * 4 :]
    return IRHeader(flag, label, id_, id2), payload


def pack(header: IRHeader, payload: bytes) -> bytes:
    label = header.label
    if isinstance(label, (list, tuple, np.ndarray)):
        label = np.asarray(label, np.float32)
        head = struct.pack(_IR_FORMAT, len(label), 0.0, header.id, header.id2)
        return head + label.tobytes() + payload
    return struct.pack(_IR_FORMAT, 0, float(label), header.id, header.id2) + payload


def read_index(idx_path: str) -> dict[int, int]:
    index = {}
    with open(idx_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                index[int(parts[0])] = int(parts[1])
    return index


class IndexedRecordReader:
    """Random-access reader. The file handle is opened lazily per process so the reader can be
    shared with DataLoader worker processes."""

    def __init__(self, rec_path: str, idx_path: str | None = None):
        self.rec_path = rec_path
        self.idx_path = idx_path or os.path.splitext(rec_path)[0] + ".idx"
        self.index = read_index(self.idx_path)
        self._fh = None
        self._pid = None

    def _file(self):
        if self._fh is None or self._pid != os.getpid():
            self._fh = open(self.rec_path, "rb")
            self._pid = os.getpid()
        return self._fh

    def read_idx(self, key: int) -> bytes:
        f = self._file()
        f.seek(self.index[key])
        magic, lrec = struct.unpack("<II", f.read(8))
        if magic != MAGIC:
            raise IOError(f"bad RecordIO magic at key {key} in {self.rec_path}")
        cflag, length = lrec >> 29, lrec & ((1 << 29) - 1)
        if cflag != 0:
            raise NotImplementedError("multi-part RecordIO records are not supported")
        return f.read(length)

    def read_header(self, key: int) -> IRHeader:
        f = self._file()
        f.seek(self.index[key] + 8)
        flag, label, id_, id2 = struct.unpack(_IR_FORMAT, f.read(_IR_SIZE))
        if flag > 0:
            label = np.frombuffer(f.read(flag * 4), np.float32).copy()
        return IRHeader(flag, label, id_, id2)

    def keys(self) -> list[int]:
        return sorted(self.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_pid"] = None
        return state


class RecordWriter:
    """Sequential writer producing .rec + .idx (used for tests and for building small subsets)."""

    def __init__(self, rec_path: str, idx_path: str | None = None):
        self.rec = open(rec_path, "wb")
        self.idx = open(idx_path or os.path.splitext(rec_path)[0] + ".idx", "w")

    def write_idx(self, key: int, record: bytes) -> None:
        self.idx.write(f"{key}\t{self.rec.tell()}\n")
        self.rec.write(struct.pack("<II", MAGIC, len(record)))
        self.rec.write(record)
        self.rec.write(b"\x00" * (-len(record) % 4))

    def close(self) -> None:
        self.rec.close()
        self.idx.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
