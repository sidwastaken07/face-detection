"""Closed-set and open-set identification on a gallery of enrolled identities."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def class_templates(emb: np.ndarray, labels: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Mean embedding per identity, re-normalised (the enrolment 'gallery')."""
    n_classes = n_classes or int(labels.max()) + 1
    t = np.zeros((n_classes, emb.shape[1]), np.float32)
    np.add.at(t, labels, emb)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def closed_set(train_emb, train_y, test_emb, test_y) -> dict:
    """Top-1 accuracy of (a) linear SVM and (b) cosine nearest class template."""
    svm = LinearSVC(C=1.0, max_iter=5000).fit(train_emb, train_y)
    templates = class_templates(train_emb, train_y)
    nn_pred = np.argmax(test_emb @ templates.T, axis=1)
    return {
        "svm_top1": float(np.mean(svm.predict(test_emb) == test_y)),
        "cosine_nn_top1": float(np.mean(nn_pred == test_y)),
    }


def open_set(emb: np.ndarray, labels: np.ndarray, unknown_frac: float = 0.2, fars=(0.01, 0.1),
             seed: int = 0) -> dict:
    """Open-set identification with identity-disjoint unknowns.

    A fraction of identities is never enrolled. Known identities: half their images enrol a
    template, the rest are probes. Detection & identification rate (DIR) at a false-alarm rate (FAR):
    threshold chosen so that `far` of unknown probes are (wrongly) accepted; DIR = fraction of known
    probes whose best match is correct *and* above the threshold."""
    rng = np.random.default_rng(seed)
    ids = np.unique(labels)
    unknown = rng.choice(ids, size=int(round(len(ids) * unknown_frac)), replace=False)
    is_unknown = np.isin(labels, unknown)

    known_idx = np.flatnonzero(~is_unknown)
    enrol, probe = train_test_split(known_idx, test_size=0.5, stratify=labels[known_idx], random_state=seed)
    known_ids = np.unique(labels[known_idx])
    remap = {c: i for i, c in enumerate(known_ids)}
    y_enrol = np.array([remap[c] for c in labels[enrol]])
    y_probe = np.array([remap[c] for c in labels[probe]])
    templates = class_templates(emb[enrol], y_enrol, len(known_ids))

    s_known = emb[probe] @ templates.T
    best_known, pred = s_known.max(1), s_known.argmax(1)
    best_unknown = (emb[is_unknown] @ templates.T).max(1)

    out = {"rank1": float(np.mean(pred == y_probe)), "n_known_ids": len(known_ids),
           "n_unknown_ids": len(unknown)}
    for far in fars:
        th = np.quantile(best_unknown, 1 - far)
        out[f"dir@far={far}"] = float(np.mean((pred == y_probe) & (best_known > th)))
        out[f"threshold@far={far}"] = float(th)
    return out
