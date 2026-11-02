"""Render test scenes as real JPEGs carrying EXIF (requires Pillow)."""

from __future__ import annotations

import io

from exifbuilder import build_tiff, standard_tags


def scene(kind: int = 0, gain: float = 1.0, size=(320, 214)):
    from PIL import Image, ImageDraw

    w, h = size
    im = Image.new("RGB", size)
    d = ImageDraw.Draw(im)
    for x in range(w):  # sky-like gradient, different per scene kind
        v = int(40 + 140 * x / w) if kind == 0 else int(180 - 140 * x / w)
        d.line([(x, 0), (x, h)], fill=(v, v, min(255, v + 30)))
    if kind == 0:
        d.rectangle([40, 120, 140, 200], fill=(90, 60, 30))
        d.ellipse([200, 30, 260, 90], fill=(230, 220, 160))
    else:
        d.polygon([(20, 200), (160, 40), (300, 200)], fill=(40, 90, 50))
        d.rectangle([230, 20, 300, 70], fill=(200, 60, 60))
    if gain != 1.0:
        im = im.point(lambda p: max(0, min(255, int(p * gain))))
    return im


def jpeg_bytes(im, exposure=(1, 250), bias=(0, 1), mode=2, second=0, subsec="00", quality=90) -> bytes:
    ifd0, exif = standard_tags(exposure=exposure, bias=bias, mode=mode, dto=f"2024:05:01 12:00:{second:02d}", subsec=subsec)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, exif=b"Exif\x00\x00" + build_tiff(ifd0, exif))
    return buf.getvalue()
