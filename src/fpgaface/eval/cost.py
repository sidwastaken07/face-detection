"""Model cost: parameters, MACs, and weight memory at a given bit-width."""
from __future__ import annotations

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, input_size=(1, 3, 112, 112)) -> int:
    """Multiply-accumulates of Conv2d and Linear layers for one forward pass of batch size 1."""
    macs = 0

    def conv_hook(m: nn.Conv2d, _, out):
        nonlocal macs
        macs += out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m: nn.Linear, _, out):
        nonlocal macs
        macs += m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    was_training = model.training
    model.eval()
    p = next(model.parameters())
    model(torch.zeros(input_size, device=p.device, dtype=p.dtype))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs


def weight_bytes(model: nn.Module, bits: int | dict[str, int] = 32) -> float:
    """Weight storage of Conv/Linear layers in bytes. `bits` is a single bit-width or a mapping
    from module name to bit-width (mixed precision; unlisted layers default to 8). BatchNorm is
    excluded because it folds into thresholds/biases at deployment."""
    total = 0.0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            b = bits if isinstance(bits, int) else bits.get(name, 8)
            total += m.weight.numel() * b / 8
    return total


def summary(model: nn.Module, input_size=(1, 3, 112, 112)) -> dict:
    return {
        "params": count_params(model),
        "macs": count_macs(model, input_size),
        "weights_fp32_MB": weight_bytes(model, 32) / 2**20,
        "weights_int8_KB": weight_bytes(model, 8) / 2**10,
        "weights_int4_KB": weight_bytes(model, 4) / 2**10,
    }
