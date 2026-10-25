"""YAML configs with single inheritance (`base: other.yaml`) and command-line overrides
(`--set train.epochs=4 model.width=0.5`)."""
from __future__ import annotations

import copy
import os

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        cfg = _merge(base, cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg
