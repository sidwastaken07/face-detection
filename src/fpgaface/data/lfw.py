"""LFW identity gallery (people with >= k images) for the identification baseline and the demo.

Only used for closed-/open-set identification experiments and the enrolment demo — never for
training. Standard LFW *verification* uses lfw.bin instead (see bins.py).
"""
from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image

from .align import align_face, center_crop


def list_identities(lfw_dir: str, min_images: int = 5) -> tuple[list[str], np.ndarray, list[str]]:
    """Returns (image paths, integer labels, sorted person names), labels read from folder names."""
    persons = sorted(
        p for p in os.listdir(lfw_dir)
        if os.path.isdir(os.path.join(lfw_dir, p)) and len(os.listdir(os.path.join(lfw_dir, p))) >= min_images
    )
    paths, labels = [], []
    for i, p in enumerate(persons):
        for f in sorted(os.listdir(os.path.join(lfw_dir, p))):
            path = os.path.join(lfw_dir, p, f)
            if os.path.isfile(path):
                paths.append(path)
                labels.append(i)
    return paths, np.array(labels, np.int64), persons


def _pick_central(boxes: np.ndarray, size: tuple[int, int]) -> int:
    cx, cy = size[0] / 2, size[1] / 2
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    return int(np.argmin(((centers - [cx, cy]) ** 2).sum(1)))


def build_gallery(lfw_dir: str, out_path: str, min_images: int = 5, device: str = "cpu",
                  batch_size: int = 64) -> dict:
    """Detect faces with MTCNN and store two crops per image in one .npz:
      * `aligned112`: 5-point ArcFace alignment (input for the teacher and our students)
      * `facenet160`: MTCNN box crop at 160×160 (FaceNet's native preprocessing)
    LFW images contain other people too, so the detection closest to the image centre is used."""
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(image_size=160, margin=0, post_process=False, keep_all=True, device=device)
    paths, labels, persons = list_identities(lfw_dir, min_images)
    a112 = np.empty((len(paths), 112, 112, 3), np.uint8)
    f160 = np.empty((len(paths), 160, 160, 3), np.uint8)
    detected = np.zeros(len(paths), bool)
    t0 = time.time()
    # LFW images are all 250×250, so MTCNN can detect a whole batch in one call.
    for b in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[b : b + batch_size]]
        if len({im.size for im in imgs}) == 1:
            boxes_b, _, points_b = mtcnn.detect(imgs, landmarks=True)
        else:
            boxes_b, points_b = zip(*[(r[0], r[2]) for r in (mtcnn.detect(im, landmarks=True) for im in imgs)])
        for k, (img, boxes, points) in enumerate(zip(imgs, boxes_b, points_b)):
            i = b + k
            if boxes is not None and len(boxes):
                j = _pick_central(boxes, img.size)
                a112[i] = np.asarray(align_face(img, points[j], 112))
                f160[i] = np.asarray(img.crop(tuple(boxes[j])).resize((160, 160), Image.BILINEAR))
                detected[i] = True
            else:
                a112[i] = np.asarray(center_crop(img, 0.5, 112))
                f160[i] = np.asarray(center_crop(img, 0.5, 160))
        done = min(b + batch_size, len(paths))
        if (b // batch_size) % 10 == 0 or done == len(paths):
            el = time.time() - t0
            print(f"  MTCNN align {done}/{len(paths)}  {done / el:.1f} img/s  eta {(len(paths) - done) / (done / el) / 60:.1f} min",
                  flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, aligned112=a112, facenet160=f160, labels=labels, detected=detected,
             persons=np.array(persons), paths=np.array(paths))
    return {"images": len(paths), "identities": len(persons), "detected": int(detected.sum())}
