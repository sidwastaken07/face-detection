"""FP32 baselines: the original notebook pipeline vs. fixed FaceNet vs. the ArcFace iResNet-50 teacher.

* Verification (identity-disjoint, standard): LFW / CFP-FP / AgeDB-30 / CALFW / CPLFW .bin files.
* Identification on the LFW >=5-image gallery:
    - closed-set top-1 with the original notebook's protocol (stratified 80/20 image split, seed 42)
    - open-set DIR@FAR with 20 % of identities held out as unknowns.

    python scripts/baselines.py --data /kaggle/working/data --out /kaggle/working/results
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fpgaface.data.bins import load_bin
from fpgaface.eval.embed import embed_images
from fpgaface.eval.identification import closed_set, open_set
from fpgaface.eval.verification import evaluate_pairs
from fpgaface.models.teachers import build_teacher
from fpgaface.utils import load_manifest, save_json


def original_notebook_baseline(gallery, device, split) -> dict:
    """Reproduces FaceNet__SVM.ipynb: whole 250×250 image resized to 160, [0,1] input (no FaceNet
    standardisation), no L2-norm, StandardScaler + SVC(linear)."""
    from facenet_pytorch import InceptionResnetV1

    net = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    feats = []
    paths = gallery["paths"]
    with torch.no_grad():
        for i in range(0, len(paths), 128):
            batch = np.stack([np.asarray(Image.open(p).convert("RGB").resize((160, 160), Image.BILINEAR))
                              for p in paths[i : i + 128]])
            x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float() / 255.0
            feats.append(net(x).cpu())
    x = torch.cat(feats).numpy()
    y = gallery["labels"]
    tr, te = split
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear")).fit(x[tr], y[tr])
    return {"closed_set": {"svm_top1": float(np.mean(clf.predict(x[te]) == y[te]))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/working/data")
    ap.add_argument("--out", default="/kaggle/working/results")
    ap.add_argument("--teachers", default="facenet,arcface_r50")
    ap.add_argument("--skip-original", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    manifest = load_manifest(args.data)
    results = {}

    gallery = None
    if "lfw_gallery" in manifest:
        gallery = dict(np.load(manifest["lfw_gallery"]))
        idx = np.arange(len(gallery["labels"]))
        split = train_test_split(idx, test_size=0.2, stratify=gallery["labels"], random_state=42)
        if not args.skip_original:
            t = time.time()
            results["original_notebook"] = original_notebook_baseline(gallery, device, split)
            print(f"original notebook pipeline: {results['original_notebook']} ({time.time() - t:.0f}s)")

    bins = {name: load_bin(path) for name, path in manifest.get("bins", {}).items()}
    for name in args.teachers.split(","):
        model = build_teacher(name).to(device).eval()
        res = {"verification": {}}
        for bname, (imgs, issame) in bins.items():
            t = time.time()
            emb = embed_images(model, imgs, args.batch_size, device, flip=True, amp=amp)
            res["verification"][bname] = evaluate_pairs(emb, issame)
            print(f"{name} {bname}: acc {res['verification'][bname]['acc']:.4f} "
                  f"TAR@1e-3 {res['verification'][bname]['tar@far=1e-3']:.4f} ({time.time() - t:.0f}s)")
        if gallery is not None:
            crops = gallery["facenet160"] if name == "facenet" else gallery["aligned112"]
            emb = embed_images(model, crops, args.batch_size, device, flip=True, amp=amp)
            tr, te = split
            y = gallery["labels"]
            res["closed_set"] = closed_set(emb[tr], y[tr], emb[te], y[te])
            res["open_set"] = open_set(emb, y)
            print(f"{name} LFW gallery: {res['closed_set']} | open-set {res['open_set']}")
        results[name] = res
        del model
        torch.cuda.empty_cache()

    save_json(results, os.path.join(args.out, "baselines.json"))
    print_table(results)


def print_table(results: dict) -> None:
    benches = sorted({b for r in results.values() for b in r.get("verification", {})})
    print("\n| model | " + " | ".join(benches) + " | LFW-gallery SVM top-1 | cosine-NN top-1 | DIR@FAR=1% |")
    print("|---" * (len(benches) + 4) + "|")
    for name, r in results.items():
        v = [f"{r['verification'][b]['acc'] * 100:.2f}" if b in r.get("verification", {}) else "–" for b in benches]
        cs, os_ = r.get("closed_set", {}), r.get("open_set", {})
        fmt = lambda d, k: f"{d[k] * 100:.2f}" if k in d else "–"
        print(f"| {name} | " + " | ".join(v) +
              f" | {fmt(cs, 'svm_top1')} | {fmt(cs, 'cosine_nn_top1')} | {fmt(os_, 'dir@far=0.01')} |")


if __name__ == "__main__":
    main()
