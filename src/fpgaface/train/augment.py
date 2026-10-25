"""Training augmentation for aligned 112×112 face crops (PIL in, normalised tensor out)."""
from __future__ import annotations

import random

from PIL import Image, ImageEnhance, ImageFilter

from ..data.transforms import ToNormalizedTensor


class RandomLowRes:
    """Downscale to a random lower resolution and back up — simulates small/distant faces."""

    def __init__(self, p: float = 0.2, min_size: int = 16):
        self.p, self.min_size = p, min_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = random.randint(self.min_size, w)
        return img.resize((s, s * h // w), Image.BILINEAR).resize((w, h), Image.BILINEAR)


class TrainTransform:
    def __init__(self, input_size: int = 112, flip: bool = True, crop_scale: float = 0.9,
                 jitter: float = 0.2, lowres_p: float = 0.2, blur_p: float = 0.1):
        self.input_size = input_size
        self.flip = flip
        self.crop_scale = crop_scale
        self.jitter = jitter
        self.blur_p = blur_p
        self.lowres = RandomLowRes(lowres_p)
        self.to_tensor = ToNormalizedTensor()

    def __call__(self, img: Image.Image) -> "torch.Tensor":
        w, h = img.size
        if self.flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.crop_scale < 1:
            s = random.uniform(self.crop_scale, 1.0)
            cw, ch = int(w * s), int(h * s)
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BILINEAR)
        if self.jitter:
            for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                img = enh(img).enhance(random.uniform(1 - self.jitter, 1 + self.jitter))
        if self.blur_p and random.random() < self.blur_p:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
        img = self.lowres(img)
        if img.size != (self.input_size, self.input_size):
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        return self.to_tensor(img)


class EvalTransform:
    def __init__(self, input_size: int = 112):
        self.input_size = input_size
        self.to_tensor = ToNormalizedTensor()

    def __call__(self, img: Image.Image):
        if img.size != (self.input_size, self.input_size):
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        return self.to_tensor(img)
