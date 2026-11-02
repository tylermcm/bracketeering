"""Locate the embedded JPEG preview inside image files.

Every major RAW format carries a camera-rendered JPEG, usually full size.
Decoding it (with DCT downscaling) costs a few milliseconds, against a
second or more to demosaic the RAW. That makes it the right input for
triage-grade pixel analysis.

The preview has the camera's tone curve applied, so highlights clip in it
somewhat *before* they clip in the RAW. Clipping measured here is
conservative (it may flag frames that are recoverable from the RAW), never
optimistic.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import BinaryIO

from .metadata.native import _iter_boxes, _Tiff

_RENDERED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
_PRVW_UUID = bytes.fromhex("eaf42b5e1c984b88b9fbb7dc406e4d16")
_CR3_META_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")
_MAX_PREVIEW = 64 * 1024 * 1024
_BASELINE_SOF = {0xC0, 0xC1, 0xC2}
_OTHER_SOF = {0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _decodable_jpeg(f: BinaryIO, off: int) -> bool:
    """True for a baseline/progressive JPEG (rejects lossless-JPEG raw data)."""
    f.seek(off)
    if f.read(2) != b"\xff\xd8":
        return False
    for _ in range(64):
        hdr = f.read(4)
        if len(hdr) < 4 or hdr[0] != 0xFF:
            return False
        marker, length = hdr[1], struct.unpack(">H", hdr[2:])[0]
        if marker in _BASELINE_SOF:
            return True
        if marker in _OTHER_SOF or marker in (0xD9, 0xDA):
            return False
        f.seek(length - 2, io.SEEK_CUR)
    return False


def _tiff_candidates(f: BinaryIO, base: int = 0) -> list[tuple[int, int]]:
    """(absolute offset, length) of JPEG streams referenced from a TIFF structure."""
    t = _Tiff(f, base)
    seen: set[int] = set()
    out: list[tuple[int, int]] = []
    queue = [t.ifd0]
    while queue and len(seen) < 16:
        off = queue.pop(0)
        if not off or off in seen:
            continue
        seen.add(off)
        try:
            tags = t.ifd(off)
            queue.append(t.next_ifd(off))
        except (ValueError, struct.error):
            continue
        sub = tags.get(0x014A)
        queue.extend(sub if isinstance(sub, tuple) else [sub] if isinstance(sub, int) else [])
        if isinstance(tags.get(0x0201), int) and isinstance(tags.get(0x0202), int):
            out.append((base + tags[0x0201], tags[0x0202]))  # type: ignore[operator]
        if isinstance(tags.get(0x0111), int) and isinstance(tags.get(0x0117), int) and tags.get(0x0103) in (6, 7):
            out.append((base + tags[0x0111], tags[0x0117]))  # type: ignore[operator]
        rw2 = tags.get(0x002E)
        if isinstance(rw2, tuple) and len(rw2) == 2:
            out.append((base + rw2[0], rw2[1]))
    return out


def _cr3_candidates(f: BinaryIO, file_end: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for typ, s, e in _iter_boxes(f, 0, file_end):
        if typ == b"uuid":
            f.seek(s)
            uid = f.read(16)
            if uid == _PRVW_UUID:
                # The PRVW box follows the UUID after a short header (8 bytes in current bodies).
                pos = f.read(64).find(b"PRVW")
                if pos >= 4:
                    out += _box_jpegs(f, s + 16 + pos - 4, e, b"PRVW")
        elif typ == b"moov":
            for ctyp, cs, ce in _iter_boxes(f, s, e):
                if ctyp == b"uuid":
                    f.seek(cs)
                    if f.read(16) == _CR3_META_UUID:
                        out += _box_jpegs(f, cs + 16, ce, b"THMB")
    return out


def _box_jpegs(f: BinaryIO, start: int, end: int, want: bytes) -> list[tuple[int, int]]:
    out = []
    for typ, s, e in _iter_boxes(f, start, end):
        if typ == want:
            f.seek(s)
            head = f.read(64)
            pos = head.find(b"\xff\xd8")
            if pos >= 0:
                out.append((s + pos, e - s - pos))
    return out


def preview_source(path: Path) -> Path | bytes | None:
    """Return the file itself (rendered formats) or the bytes of its best embedded JPEG."""
    if path.suffix.lower() in _RENDERED:
        with open(path, "rb") as fh:
            head = fh.read(4)
        if head[:2] == b"\xff\xd8" or path.suffix.lower() not in (".tif", ".tiff"):
            return path
    try:
        with open(path, "rb") as f:
            head = f.read(16)
            f.seek(0, io.SEEK_END)
            size = f.tell()
            if head.startswith(b"FUJIFILMCCD-RAW"):
                f.seek(84)
                cands = [struct.unpack(">II", f.read(8))]
            elif head[4:8] == b"ftyp":
                cands = _cr3_candidates(f, size)
            elif head[:2] in (b"II", b"MM"):
                cands = _tiff_candidates(f)
            else:
                return None
            for off, length in sorted(set(cands), key=lambda c: -c[1]):
                if 0 < length <= _MAX_PREVIEW and off + length <= size and _decodable_jpeg(f, off):
                    f.seek(off)
                    return f.read(length)
    except (OSError, ValueError, struct.error):
        return None
    # A plain (non-RAW) TIFF with no embedded JPEG: let the decoder read it directly.
    return path if path.suffix.lower() in (".tif", ".tiff") else None
