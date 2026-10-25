"""Sweep student sizes × bit-widths through the on-chip memory estimate.

    python scripts/budget.py --device z7020 [--out docs/budget.md]
"""
from __future__ import annotations

import argparse
import itertools

from fpgaface.eval.cost import count_macs, count_params
from fpgaface.hw.budget import DEVICES, estimate
from fpgaface.models.students import build_student

CANDIDATES = [
    ("mobilefacenet", 1.0, 112), ("mobilefacenet", 0.75, 112), ("mobilefacenet", 0.5, 112),
    ("mobilefacenet", 1.0, 64), ("mobilefacenet", 0.75, 64),
    ("vgg", 1.0, 112), ("vgg", 0.75, 112), ("vgg", 0.5, 112),
]
BITS = [(8, 8), (4, 4), (2, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="z7020", choices=list(DEVICES))
    ap.add_argument("--out")
    args = ap.parse_args()

    lines = [f"On-chip memory estimate for **{DEVICES[args.device].name}** (first-order, see `fpgaface/hw/budget.py`)",
             "", "| model | width | input | params | MMACs | W/A bits | weights (Kb) | thresholds (Kb) | "
             "buffers (Kb) | BRAM18 | BRAM % | LUTRAM % | fits |", "|---" * 13 + "|"]
    for (arch, width, size), (wb, ab) in itertools.product(CANDIDATES, BITS):
        model = build_student(arch, width=width, input_size=size)
        r = estimate(model, size, wb, ab, args.device)
        lines.append(
            f"| {arch} | {width} | {size} | {count_params(model) / 1e6:.2f}M | "
            f"{count_macs(model, (1, 3, size, size)) / 1e6:.0f} | W{wb}A{ab} | {r['weights_kbit']:.0f} | "
            f"{r['thresholds_kbit']:.0f} | {r['buffers_kbit']:.0f} | {r['bram18']} | {r['bram_util'] * 100:.0f} | "
            f"{r['lutram_util'] * 100:.0f} | {'✅' if r['fits'] else '❌'} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
