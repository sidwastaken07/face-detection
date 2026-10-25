"""First-order on-chip memory budget for a FINN streaming-dataflow implementation.

FINN instantiates every layer as its own hardware block and keeps *all* parameters on chip, so a
network fits only if its weights, activation thresholds and stream buffers fit in BRAM/LUTRAM.
This estimate is used to size students before training; FINN's own estimate reports (Phase 6)
replace it once a model is exported.

Per conv layer (bit-widths W weights, A activations):
* weights:     Cout · Cin/groups · k² · W bits
* thresholds:  a quantised activation with A bits = (2^A − 1) thresholds per output channel, each as
               wide as the accumulator (W + A_in + log2(fan-in) bits). BN and the activation fold into
               these thresholds, so they exist even for "linear" layers whose output is re-quantised.
               This is why 8-bit activations and wide layers are expensive in FINN.
* line buffer: sliding-window generator for k > 1 keeps ~k rows of the input: k · W_in · Cin · A bits
* residual:    skip-connection FIFO ≈ 2 input rows · W_in · C · A bits per residual block
Memories smaller than LUTRAM_MAX_BITS map to LUTRAM, larger ones to BRAM18 at BRAM_EFFICIENCY packing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

BRAM18_BITS = 18 * 1024
BRAM_EFFICIENCY = 0.75     # width/depth packing losses with typical PE/SIMD folding
LUTRAM_MAX_BITS = 4 * 1024


@dataclass(frozen=True)
class Device:
    name: str
    bram18: int
    uram: int          # 288 Kb UltraRAM blocks
    luts: int
    lutram_luts: int   # LUTs usable as 64-bit distributed RAM (SLICEM)
    dsps: int

    @property
    def onchip_bits(self) -> int:
        return self.bram18 * BRAM18_BITS + self.uram * 288 * 1024


DEVICES = {
    "z7020": Device("Zynq-7020 (PYNQ-Z2)", 280, 0, 53_200, 17_400, 220),
    "zu3eg": Device("ZU3EG (Ultra96-V2)", 432, 0, 70_560, 28_800, 360),
    "kv260": Device("K26 (Kria KV260)", 288, 64, 117_120, 57_600, 1_248),
}


@dataclass
class LayerMem:
    name: str
    weight_bits: int
    threshold_bits: int
    buffer_bits: int


def _bits_for(bits, name: str, default: int) -> int:
    return bits if isinstance(bits, int) else bits.get(name, default)


@torch.no_grad()
def layer_memories(model: nn.Module, input_size: int, wbits=4, abits=4, input_bits: int = 8) -> list[LayerMem]:
    """`wbits`/`abits` are ints or {layer name: bits} (mixed precision; missing names use 8)."""
    shapes = {}
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(lambda mod, inp, out: shapes.__setitem__(mod, inp[0].shape)))
    was_training = model.training
    model.eval()
    model(torch.zeros(1, 3, input_size, input_size))
    model.train(was_training)
    for h in hooks:
        h.remove()

    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    mems = []
    for i, m in enumerate(convs):
        name = getattr(m, "fpga_name", f"conv{i}")
        _, cin, h, w = shapes[m]
        k = m.kernel_size[0]
        wb = _bits_for(wbits, name, 8)
        a_in = input_bits if i == 0 else _bits_for(abits, name, 8)
        a_out = _bits_for(abits, name + ".act", _bits_for(abits, name, 8))
        weight = m.weight.numel() * wb
        fan_in = cin // m.groups * k * k
        acc_bits = wb + a_in + math.ceil(math.log2(fan_in))
        thresholds = (2 ** a_out - 1) * m.out_channels * acc_bits
        buffer = k * w * cin * a_in if k > 1 and k < h else 0
        if k >= h:  # GDC head: window covers the whole map, buffer the full feature map
            buffer = h * w * cin * a_in
        mems.append(LayerMem(name, weight, thresholds, buffer))

    # Residual skip FIFOs: one per residual bottleneck, sized by that block's input.
    for mod in model.modules():
        if getattr(mod, "residual", False):
            first = next(c for c in mod.modules() if isinstance(c, nn.Conv2d))
            _, cin, h, w = shapes[first]
            a = _bits_for(abits, getattr(first, "fpga_name", ""), 8)
            mems.append(LayerMem(getattr(first, "fpga_name", "res") + ".skip", 0, 0, 2 * w * cin * a))
    return mems


def _place(bits: int) -> tuple[int, int]:
    """(BRAM18 blocks, LUTs used as LUTRAM) for one memory."""
    if bits == 0:
        return 0, 0
    if bits <= LUTRAM_MAX_BITS:
        return 0, math.ceil(bits / 64)
    return math.ceil(bits / (BRAM18_BITS * BRAM_EFFICIENCY)), 0


def estimate(model: nn.Module, input_size: int, wbits=4, abits=4, device: str = "z7020") -> dict:
    mems = layer_memories(model, input_size, wbits, abits)
    dev = DEVICES[device]
    bram = lutram = 0
    for m in mems:
        for bits in (m.weight_bits, m.threshold_bits, m.buffer_bits):
            b, l = _place(bits)
            bram += b
            lutram += l
    tot = lambda f: sum(f(m) for m in mems)
    return {
        "device": dev.name,
        "weights_kbit": tot(lambda m: m.weight_bits) / 1024,
        "thresholds_kbit": tot(lambda m: m.threshold_bits) / 1024,
        "buffers_kbit": tot(lambda m: m.buffer_bits) / 1024,
        "bram18": bram,
        "bram18_available": dev.bram18 + dev.uram * 16,  # 1 URAM ≈ 16 BRAM18 of capacity
        "bram_util": bram / (dev.bram18 + dev.uram * 16),
        "lutram_luts": lutram,
        "lutram_util": lutram / dev.lutram_luts,
        "fits": bram <= dev.bram18 + dev.uram * 16 and lutram <= dev.lutram_luts,
    }
