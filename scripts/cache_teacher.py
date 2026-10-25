"""Pre-compute teacher embeddings for every CASIA image (distillation targets).

Output: `<out>/teacher_<name>.f16.npy`, an fp16 memmap [N, 512] of L2-normalised embeddings, row i =
CASIA dataset index i. Resumable: progress is checkpointed so a killed session continues where it
stopped. Only the original (unflipped) view is cached; faces are near-symmetric and the flip-augmented
student is distilled towards the same target.

    python scripts/cache_teacher.py --data <prepared data dir> --out /kaggle/working/teacher
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fpgaface.data.casia import CasiaDataset
from fpgaface.data.transforms import ToNormalizedTensor
from fpgaface.models.teachers import build_teacher
from fpgaface.utils import load_manifest, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/working/data")
    ap.add_argument("--out", default="/kaggle/working/teacher", help="output dir (data dir may be read-only)")
    ap.add_argument("--teacher", default="arcface_r50")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N images (smoke test)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    casia = load_manifest(args.data)["casia"]
    ds = CasiaDataset(casia["rec"], casia["labels"], transform=ToNormalizedTensor())
    n = min(len(ds), args.limit or len(ds))
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"teacher_{args.teacher}.f16.npy")
    progress_path = out_path + ".progress"

    model = build_teacher(args.teacher).to(device).eval()
    dim = model(torch.zeros(1, 3, 112, 112, device=device)).shape[1]
    start = int(open(progress_path).read()) if os.path.exists(progress_path) and os.path.exists(out_path) else 0
    mode = "r+" if start > 0 else "w+"
    emb = np.lib.format.open_memmap(out_path, mode=mode, dtype=np.float16, shape=(n, dim))
    print(f"{args.teacher}: {n} images, dim {dim}, resuming at {start}")

    loader = DataLoader(Subset(ds, range(start, n)), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device == "cuda")
    t0, done = time.time(), start
    with torch.no_grad():
        for step, (x, _, idx) in enumerate(loader):
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                e = model(x.to(device, non_blocking=True)).float()
            emb[idx.numpy()] = F.normalize(e, dim=1).cpu().numpy().astype(np.float16)
            done = int(idx[-1]) + 1
            if step % 50 == 0:
                emb.flush()
                with open(progress_path, "w") as f:
                    f.write(str(done))
                rate = (done - start) / (time.time() - t0)
                print(f"  {done}/{n}  {rate:.0f} img/s  eta {(n - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)
    emb.flush()
    with open(progress_path, "w") as f:
        f.write(str(done))
    save_json({"teacher": args.teacher, "num_images": n, "dim": int(dim), "path": out_path,
               "seconds": time.time() - t0}, out_path + ".json")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
