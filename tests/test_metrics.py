import numpy as np
import torch.nn as nn

from fpgaface.eval.cost import count_macs, count_params, weight_bytes
from fpgaface.eval.identification import closed_set, open_set
from fpgaface.eval.verification import best_threshold, evaluate_pairs


def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _pairs(rng, n, noise):
    a = _norm(rng.normal(size=(n, 64)))
    issame = rng.random(n) < 0.5
    b = np.where(issame[:, None], a + noise * rng.normal(size=a.shape), rng.normal(size=a.shape))
    emb = np.empty((2 * n, 64))
    emb[0::2], emb[1::2] = a, _norm(b)
    return emb, issame


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    sims, y = rng.normal(size=200), rng.random(200) < 0.5
    t = best_threshold(sims, y)
    brute = max(np.mean((sims >= c) == y) for c in np.concatenate([sims, [sims.max() + 1]]))
    assert np.isclose(np.mean((sims >= t) == y), brute)


def test_verification_separable_vs_random():
    rng = np.random.default_rng(0)
    assert evaluate_pairs(*_pairs(rng, 600, 0.1))["acc"] > 0.99
    rand = _norm(rng.normal(size=(1200, 64)))
    assert evaluate_pairs(rand, rng.random(600) < 0.5)["acc"] < 0.65


def test_identification():
    rng = np.random.default_rng(0)
    centers = _norm(rng.normal(size=(30, 64)))
    y = np.repeat(np.arange(30), 10)
    emb = _norm(centers[y] + 0.1 * rng.normal(size=(300, 64)))
    cs = closed_set(emb[::2], y[::2], emb[1::2], y[1::2])
    assert cs["svm_top1"] > 0.95 and cs["cosine_nn_top1"] > 0.95
    os_ = open_set(emb, y)
    assert os_["rank1"] > 0.95 and os_["dir@far=0.1"] > 0.8 and os_["n_unknown_ids"] == 6


def test_cost():
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(8 * 4 * 4, 10))
    assert count_params(m) == 3 * 8 * 9 + 8 + 128 * 10 + 10
    assert count_macs(m, (1, 3, 4, 4)) == 16 * 8 * 27 + 1280
    assert weight_bytes(m, 4) == (216 + 1280) / 2
    assert weight_bytes(m, {"0": 2, "2": 8}) == 216 / 4 + 1280
