"""Markdown table of all finished runs (results.json) under one or more run directories.

    python scripts/summarize_runs.py /kaggle/working/runs [--out docs/results_fp32.md]
"""
from __future__ import annotations

import argparse
import glob
import os

from fpgaface.utils import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = []
    for root in args.roots:
        for path in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
            r = load_json(path)
            cfg = load_json(os.path.join(os.path.dirname(path), "config.json"))
            rows.append((r, cfg))
    benches = sorted({b for r, _ in rows for b, v in r["verification"].items() if isinstance(v, dict)})
    lines = ["| run | arch | width | input | loss | KD | ids | params | MMACs | " + " | ".join(benches) +
             " | mean | W4A4 Z7020 BRAM % | hours |", "|---" * (12 + len(benches)) + "|"]
    for r, c in rows:
        v = r["verification"]
        lines.append(
            f"| {r['run_name']} | {c['model']['arch']} | {c['model']['width']} | {c['model']['input_size']} | "
            f"{c['loss']['head']} | {c['loss']['kd'] or '–'} | {c['data']['num_ids'] or 'all'} | "
            f"{r['cost']['params'] / 1e6:.2f}M | {r['cost']['macs'] / 1e6:.0f} | " +
            " | ".join(f"{v[b]['acc'] * 100:.2f}" if b in v else "–" for b in benches) +
            f" | {v.get('mean_acc', 0) * 100:.2f} | {r['budget_w4a4_z7020']['bram_util'] * 100:.0f} | {r['train_hours']:.2f} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
