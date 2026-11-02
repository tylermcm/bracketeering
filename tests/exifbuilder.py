"""Build minimal but structurally valid EXIF containers for parser tests."""

from __future__ import annotations

import struct

ASCII, SHORT, LONG, RATIONAL, SRATIONAL = 2, 3, 4, 5, 10


def _encode(e: str, typ: int, value) -> tuple[bytes, int]:
    if typ == ASCII:
        data = value.encode() + b"\x00"
        return data, len(data)
    vals = value if isinstance(value, list) else [value]
    if typ == SHORT:
        return struct.pack(f"{e}{len(vals)}H", *vals), len(vals)
    if typ == LONG:
        return struct.pack(f"{e}{len(vals)}I", *vals), len(vals)
    if typ in (RATIONAL, SRATIONAL):
        flat = [x for pair in vals for x in pair]
        return struct.pack(f"{e}{len(flat)}{'I' if typ == RATIONAL else 'i'}", *flat), len(vals)
    raise ValueError(typ)


def _ifd(e: str, entries: list[tuple[int, int, object]], offset: int) -> bytes:
    """Serialise one IFD placed at `offset` (relative to TIFF start), data following it."""
    entries = sorted(entries)
    data_start = offset + 2 + 12 * len(entries) + 4
    head = struct.pack(f"{e}H", len(entries))
    blob = b""
    for tag, typ, value in entries:
        raw, count = _encode(e, typ, value)
        if len(raw) <= 4:
            field = raw.ljust(4, b"\x00")
        else:
            field = struct.pack(f"{e}I", data_start + len(blob))
            blob += raw + (b"\x00" if len(raw) % 2 else b"")
        head += struct.pack(f"{e}HHI", tag, typ, count) + field
    return head + struct.pack(f"{e}I", 0) + blob


def build_tiff(ifd0: list, exif: list, big_endian: bool = False, magic: int = 42, extra_ifd0: list | None = None) -> bytes:
    e = ">" if big_endian else "<"
    ifd0 = list(ifd0) + list(extra_ifd0 or [])
    # Lay out IFD0 with a placeholder Exif pointer, then append the Exif IFD.
    probe = _ifd(e, ifd0 + [(0x8769, LONG, 0)], 8)
    exif_off = 8 + len(probe)
    exif_off += exif_off % 2
    first = _ifd(e, ifd0 + [(0x8769, LONG, exif_off)], 8)
    hdr = (b"MM" if big_endian else b"II") + struct.pack(f"{e}HI", magic, 8)
    body = hdr + first
    body = body.ljust(exif_off, b"\x00")
    return body + _ifd(e, exif, exif_off)


def build_tiff_flat(entries: list, big_endian: bool = False) -> bytes:
    """A TIFF whose IFD0 *is* the tag list (used for CR3 CMT boxes)."""
    e = ">" if big_endian else "<"
    return (b"MM" if big_endian else b"II") + struct.pack(f"{e}HI", 42, 8) + _ifd(e, entries, 8)


def jpeg_with_exif(tiff: bytes) -> bytes:
    app1 = b"Exif\x00\x00" + tiff
    return (
        b"\xff\xd8"
        + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
        + b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00" + b"\x00" * 16 + b"\xff\xd9"
    )


def raf_with_exif(tiff: bytes) -> bytes:
    jpeg = jpeg_with_exif(tiff)
    header = b"FUJIFILMCCD-RAW 0201FF383501".ljust(84, b"\x00")
    jpeg_off = 160
    header += struct.pack(">II", jpeg_off, len(jpeg))
    return header.ljust(jpeg_off, b"\x00") + jpeg


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + typ + payload


def cr3_with_exif(ifd0: list, exif: list) -> bytes:
    uuid = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")
    inner = _box(b"CNCV", b"CanonCR3_001/00.09.00/00.00.00") + _box(b"CMT1", build_tiff_flat(ifd0)) + _box(b"CMT2", build_tiff_flat(exif))
    moov = _box(b"moov", _box(b"uuid", uuid + inner))
    ftyp = _box(b"ftyp", b"crx \x00\x00\x00\x01crx isom")
    return ftyp + moov + _box(b"mdat", b"\x00" * 32)


def rw2_with_embedded_jpeg(make: str, exif_tiff: bytes) -> bytes:
    """Panasonic-style: IFD0 without an Exif pointer, full EXIF in JpgFromRaw (0x002E)."""
    jpeg = jpeg_with_exif(exif_tiff)
    raw_make = make.encode() + b"\x00"
    ifd_size = 2 + 2 * 12 + 4
    make_off = 8 + ifd_size
    jpeg_off = make_off + len(raw_make) + (len(raw_make) % 2)
    ifd = struct.pack("<H", 2)
    ifd += struct.pack("<HHII", 0x002E, 7, len(jpeg), jpeg_off)  # entries sorted by tag
    ifd += struct.pack("<HHII", 0x010F, ASCII, len(raw_make), make_off)
    ifd += struct.pack("<I", 0)
    body = b"IIU\x00" + struct.pack("<I", 8) + ifd + raw_make
    return body.ljust(jpeg_off, b"\x00") + jpeg


def standard_tags(exposure=(1, 250), fnum=(8, 1), iso=100, bias=(0, 1), mode=2, dto="2024:05:01 12:00:00", subsec="123", offset="+02:00"):
    ifd0 = [(0x010F, ASCII, "Canon"), (0x0110, ASCII, "Canon EOS R5"), (0x0112, SHORT, 1)]
    exif = [
        (0x829A, RATIONAL, exposure), (0x829D, RATIONAL, fnum), (0x8822, SHORT, 3), (0x8827, SHORT, iso),
        (0x9003, ASCII, dto), (0x9011, ASCII, offset), (0x9291, ASCII, subsec),
        (0x9204, SRATIONAL, bias), (0x9209, SHORT, 16), (0x920A, RATIONAL, (24, 1)),
        (0xA402, SHORT, mode), (0xA403, SHORT, 0), (0xA431, ASCII, "012345678901"),
        (0xA434, ASCII, "RF24-70mm F2.8 L IS USM"),
    ]
    return ifd0, exif
