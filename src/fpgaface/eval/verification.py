"""Face verification metrics on pair benchmarks (LFW-style 10-fold protocol)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold


def pair_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity of consecutive pairs (2k, 2k+1); embeddings must be L2-normalised."""
    return np.sum(embeddings[0::2] * embeddings[1::2], axis=1)


def best_threshold(sims: np.ndarray, issame: np.ndarray) -> float:
    """Threshold maximising accuracy of `sims >= t`, in O(n log n). Only midpoints between sorted
    similarities change the accuracy, so those are the candidates."""
    order = np.argsort(sims)
    s, y = sims[order], issame[order]
    # Threshold below element k: elements [0, k) predicted different, [k, n) predicted same.
    neg_below = np.concatenate([[0], np.cumsum(~y)])
    pos_at_or_above = np.concatenate([[0], np.cumsum(y[::-1])])[::-1]
    k = int(np.argmax(neg_below + pos_at_or_above))
    if k == 0:
        return float(s[0] - 1e-6)
    if k == len(s):
        return float(s[-1] + 1e-6)
    return float((s[k - 1] + s[k]) / 2)


def kfold_accuracy(sims: np.ndarray, issame: np.ndarray, n_folds: int = 10) -> tuple[float, float, float]:
    """Pick the best threshold on 9 folds, measure accuracy on the 10th (InsightFace protocol:
    unshuffled folds). Returns (mean acc, std acc, mean threshold)."""
    accs, ths = [], []
    for train, test in KFold(n_splits=n_folds, shuffle=False).split(sims):
        t = best_threshold(sims[train], issame[train])
        accs.append(np.mean((sims[test] >= t) == issame[test]))
        ths.append(t)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(ths))


def tar_at_far(sims: np.ndarray, issame: np.ndarray, far: float) -> float:
    fpr, tpr, _ = roc_curve(issame.astype(int), sims)
    ok = fpr <= far
    return float(tpr[ok].max()) if ok.any() else 0.0


def evaluate_pairs(embeddings: np.ndarray, issame: np.ndarray) -> dict:
    sims = pair_similarity(embeddings)
    acc, std, th = kfold_accuracy(sims, issame)
    return {
        "acc": acc,
        "acc_std": std,
        "threshold": th,
        "auc": float(roc_auc_score(issame.astype(int), sims)),
        "tar@far=1e-2": tar_at_far(sims, issame, 1e-2),
        "tar@far=1e-3": tar_at_far(sims, issame, 1e-3),
    }
