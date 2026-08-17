"""Literal M6 view transforms plus a tiny deterministic golden fixture."""
from __future__ import annotations

from io import BytesIO
import struct

import PIL
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from benchmark.m6.p5_protocol import VIEWS


PINNED_PILLOW = "11.3.0"
FIXTURE = bytes([index % 256 for index in range(8 * 6 * 3)])


def _jpeg_round_trip(image: Image.Image, *, quality: int) -> Image.Image:
    output = BytesIO()
    image.convert("RGB").save(
        output,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=False,
        subsampling=2,
    )
    output.seek(0)
    with Image.open(output) as decoded:
        return decoded.convert("RGB")


def apply_view(image: Image.Image, view: str) -> Image.Image:
    if PIL.__version__ != PINNED_PILLOW:
        raise RuntimeError("M6 view transforms require the pinned Pillow 11.3.0 runtime")
    if view not in VIEWS:
        raise ValueError("unknown M6 view")
    source = ImageOps.exif_transpose(image).convert("RGB")
    width, height = source.size
    if view == "original":
        return source
    if view == "screenshot":
        border = max(1, round(min(width, height) * 0.08))
        return ImageOps.expand(source, border=border, fill=(248, 248, 248))
    if view == "social-q75":
        return _jpeg_round_trip(source, quality=75)
    if view == "social-heavy":
        smaller = source.resize((max(1, round(width * 0.75)), max(1, round(height * 0.75))), Image.Resampling.LANCZOS)
        compressed = _jpeg_round_trip(smaller, quality=55)
        return compressed.resize((width, height), Image.Resampling.BICUBIC).filter(ImageFilter.SHARPEN)
    if view == "forum-repost":
        bordered = ImageOps.expand(ImageEnhance.Contrast(source).enhance(1.03), border=max(1, round(min(width, height) * 0.04)), fill=(245, 245, 245))
        return _jpeg_round_trip(bordered, quality=82)
    if view == "search-thumbnail":
        thumbnail = source.copy()
        thumbnail.thumbnail((max(1, round(width * 0.5)), max(1, round(height * 0.5))), Image.Resampling.LANCZOS)
        return thumbnail.resize((width, height), Image.Resampling.BILINEAR)
    return _jpeg_round_trip(source, quality=82)


def render(view: str) -> tuple[bytes, tuple[int, int]]:
    image = apply_view(Image.frombytes("RGB", (8, 6), FIXTURE), view)
    width, height = image.size
    payload = b"seroslop-m6-view-fixture-v1\0" + struct.pack(">II", width, height) + image.tobytes()
    return payload, image.size


# Generated only from FIXTURE through apply_view under PINNED_PILLOW.
GOLDENS = {
    "original": {"sha256": "27c3d5caed3009a1b8b3093755404ec76b94d58f39090c365e63e60e580c6c2b", "dimensions": [8, 6]},
    "screenshot": {"sha256": "62068f2fac720eede3d2db70f5ccb0461048272db25c2f69fe9244d8dd222d52", "dimensions": [10, 8]},
    "social-q75": {"sha256": "0166d2c962ac0983d20c593eabc26fdc44e9e074ce6d12898aa2efa084fbad12", "dimensions": [8, 6]},
    "social-heavy": {"sha256": "a2726e34b469e1e5ad91407ae66d773bb059d378b233d1b05ba2311eed959122", "dimensions": [8, 6]},
    "forum-repost": {"sha256": "8f6a236d900cea72703a9655e17a9c146c2018b3f2110399a898cdf32c7bb3ed", "dimensions": [10, 8]},
    "search-thumbnail": {"sha256": "d54193e6c0f2b0b82e88b9880552babe88db74acef54c04cf2b2ad8212b09b56", "dimensions": [8, 6]},
    "provider-cdn": {"sha256": "c2ae567db2741afd362ff3f152fff88a250e7ed0c80738103baf7c4640a47a7a", "dimensions": [8, 6]},
}
