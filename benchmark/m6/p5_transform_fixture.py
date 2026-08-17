"""Literal M6 view transforms plus a tiny deterministic golden fixture."""
from __future__ import annotations

from io import BytesIO

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
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=6)
    return output.getvalue(), image.size


# Generated only from FIXTURE through apply_view under PINNED_PILLOW.
GOLDENS = {
    "original": {"sha256": "26aadcd12d6d013670b19cb5925c9dde800a3de211763e33ecdb437c492743c6", "dimensions": [8, 6]},
    "screenshot": {"sha256": "cb8995e9424a97d012ee5ffe90c13f899a71c92da85975ccbe1c34ee71f3202d", "dimensions": [10, 8]},
    "social-q75": {"sha256": "3842e1f4567279c80bc98147d8eaf04deb2bc95a232afc1cc6b2f64daf9a340a", "dimensions": [8, 6]},
    "social-heavy": {"sha256": "720816a012718eed1716b1aad25a154575efcf5bddc3e91b30387797c9d1e9aa", "dimensions": [8, 6]},
    "forum-repost": {"sha256": "691d6607e6b69c54aa8ebca8bc78bcf3834c49489b04bec651f34d93f1c42a8d", "dimensions": [10, 8]},
    "search-thumbnail": {"sha256": "6ee1fa8906d4b28208b9035827a7ba4d5d02353e18b0c1a54b7282947ada9692", "dimensions": [8, 6]},
    "provider-cdn": {"sha256": "b6a769372f1b51f91ff545008a7180bfe52562f31c2952bd4174b8edd76dd418", "dimensions": [8, 6]},
}
