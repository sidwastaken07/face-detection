"""Locate and verify datasets, index CASIA-WebFace labels, and build the aligned LFW gallery.

Writes `<out>/data_manifest.json`, which every later script reads.

    python scripts/prepare_data.py --out /kaggle/working/data
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from fpgaface.data.bins import BENCHMARKS, find_benchmarks, load_bin
from fpgaface.data.casia import load_or_scan_labels
from fpgaface.data.lfw import build_gallery
from fpgaface.utils import DEFAULT_SEARCH_ROOTS, find_dir, find_file, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/data")
    ap.add_argument("--casia-rec", help="path to CASIA train.rec (default: search input roots)")
    ap.add_argument("--bins-root", help="directory containing *.bin benchmarks (default: search input roots)")
    ap.add_argument("--lfw-dir", help="LFW image folder (default: search for lfw-deepfunneled)")
    ap.add_argument("--min-images", type=int, default=5)
    ap.add_argument("--skip-gallery", action="store_true")
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {}

    # CASIA-WebFace
    rec = args.casia_rec or find_file("train.rec")
    if rec:
        t = time.time()
        keys, labels = load_or_scan_labels(rec, os.path.join(args.out, "casia_labels.npy"))
        counts = np.bincount(labels)
        manifest["casia"] = {
            "rec": rec,
            "labels": "casia_labels.npy",
            "num_images": len(keys),
            "num_ids": int((counts > 0).sum()),
            "imgs_per_id_median": float(np.median(counts[counts > 0])),
        }
        print(f"CASIA: {len(keys)} images, {manifest['casia']['num_ids']} identities "
              f"(label scan {time.time() - t:.0f}s)")
    else:
        print("WARNING: CASIA train.rec not found — attach the faces_webface_112x112 dataset.")

    # Verification benchmarks
    bins = {}
    for root in ([args.bins_root] if args.bins_root else DEFAULT_SEARCH_ROOTS):
        if root and os.path.isdir(root):
            bins.update({k: v for k, v in find_benchmarks(root).items() if k not in bins})
    for name in BENCHMARKS:
        if name in bins:
            imgs, issame = load_bin(bins[name])
            print(f"{name}: {len(issame)} pairs ({issame.sum()} same), images {imgs.shape}")
        else:
            print(f"{name}: not found")
    manifest["bins"] = bins

    # LFW identification gallery
    if not args.skip_gallery:
        lfw_dir = args.lfw_dir or find_dir("lfw-deepfunneled", must_contain="George_W_Bush")
        if lfw_dir:
            name = f"lfw_gallery_min{args.min_images}.npz"
            out = os.path.join(args.out, name)
            if not os.path.exists(out):
                stats = build_gallery(lfw_dir, out, args.min_images, args.device)
                print(f"LFW gallery: {stats}")
            manifest["lfw_gallery"] = name
        else:
            print("WARNING: LFW image folder not found — gallery skipped.")

    save_json(manifest, os.path.join(args.out, "data_manifest.json"))
    print(f"manifest -> {os.path.join(args.out, 'data_manifest.json')}")


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
