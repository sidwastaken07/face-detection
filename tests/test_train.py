import pytest
import torch
import torch.nn as nn

from fpgaface.config import load_config
from fpgaface.hw.budget import estimate
from fpgaface.models.students import build_student
from fpgaface.train.augment import TrainTransform
from fpgaface.train.losses import EmbeddingDistill, MarginHead, relational_distill


@pytest.mark.parametrize("kind", ["arcface", "cosface", "adaface"])
def test_margin_head_learns(kind):
    torch.manual_seed(0)
    centers = torch.randn(10, 16) * 3
    y = torch.arange(10).repeat(40)
    x = centers[y] + torch.randn(400, 16)
    net = nn.Linear(16, 32)
    head = MarginHead(32, 10, kind)
    opt = torch.optim.SGD(list(net.parameters()) + list(head.parameters()), lr=0.05, momentum=0.9)
    for _ in range(150):
        loss = nn.functional.cross_entropy(head(net(x), y), y)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        cos = nn.functional.normalize(net(x)) @ nn.functional.normalize(head.weight).T
    assert torch.isfinite(loss) and (cos.argmax(1) == y).float().mean() > 0.95


def test_distill_losses():
    s, t = torch.randn(8, 128), torch.randn(8, 512)
    kd = EmbeddingDistill(128, 512)
    assert 0 <= kd(s, t) <= 2
    assert relational_distill(t, t) < 1e-6


@pytest.mark.parametrize("arch,size", [("mobilefacenet", 112), ("mobilefacenet", 64), ("vgg", 112)])
def test_students_shapes(arch, size):
    m = build_student(arch, width=0.5, input_size=size).eval()
    assert m(torch.randn(2, 3, size, size)).shape == (2, 128)
    names = [getattr(c, "fpga_name", None) for c in m.modules() if isinstance(c, nn.Conv2d)]
    assert all(names) and len(set(names)) == len(names)


def test_mobilefacenet_matches_paper_size():
    m = build_student("mobilefacenet", width=1.0)
    assert 0.95e6 < sum(p.numel() for p in m.parameters()) < 1.05e6


def test_budget_monotonic_in_bits():
    m = build_student("mobilefacenet", width=0.5)
    b8, b4, b2 = (estimate(m, 112, b, b)["bram18"] for b in (8, 4, 2))
    assert b8 > b4 > b2


def test_train_transform():
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (112, 112, 3), np.uint8))
    for size in (112, 64):
        x = TrainTransform(size, lowres_p=1.0, blur_p=1.0)(img)
        assert x.shape == (3, size, size) and x.min() >= -1 and x.max() <= 1


def test_config_inheritance():
    cfg = load_config("configs/ablations/abl_res64.yaml", ["train.epochs=1"])
    assert cfg["model"]["input_size"] == 64 and cfg["model"]["width"] == 0.75
    assert cfg["data"]["num_ids"] == 2000 and cfg["train"]["epochs"] == 1 and cfg["run_name"] == "abl_res64"
    assert cfg["loss"]["head"] == "arcface"
