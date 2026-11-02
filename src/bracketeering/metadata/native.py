"""Dependency-free EXIF reader.

Reads only the bytes it needs (IFD entries and their values), seeking within
the file rather than loading it, so a 60 MB RAW costs a few kilobytes of I/O.

Supported containers:
  * JPEG (APP1 Exif)
  * TIFF and TIFF-derived RAWs: DNG, CR2, NEF, NRW, ARW, SR2, PEF, ORF, RW2, 3FR, IIQ, ...
  * Fujifilm RAF (via the embedded JPEG)
  * Canon CR3 (ISO-BMFF CMT1/CMT2 boxes)

Maker notes are not decoded here; use the exiftool backend for those.
"""

from __future__ import annotations

import io
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO

from ..model import FileMeta

# --- tag ids ---------------------------------------------------------------
T_MAKE, T_MODEL, T_ORIENT, T_DATETIME = 0x010F, 0x0110, 0x0112, 0x0132
T_EXIF_IFD, T_RW2_JPEG = 0x8769, 0x002E
T_EXPTIME, T_FNUMBER, T_PROGRAM, T_ISO = 0x829A, 0x829D, 0x8822, 0x8827
T_REI, T_ISOSPEED = 0x8832, 0x8833
T_DTO, T_OFFSET_DTO, T_SUBSEC_DTO = 0x9003, 0x9011, 0x9291
T_BIAS, T_SUBJ_DIST, T_FLASH, T_FOCAL = 0x9204, 0x9206, 0x9209, 0x920A
T_EXPMODE, T_WB = 0xA402, 0xA403
T_BODY_SERIAL, T_LENS_MODEL = 0xA431, 0xA434
T_MAKERNOTE = 0x927C
# Nikon type-3 maker note tags (same ids as exiftool's Nikon::Main table)
N_EXPOSURE_BRACKET_VALUE, N_SHOOTING_MODE, N_SHUTTER_COUNT, N_SERIAL = 0x0019, 0x0089, 0x00A7, 0x001D
N_FOCUS_MODE = 0x0007  # "AF-S", "AF-C", "MF", ... or "Fshift" during focus-shift shooting
N_MODE_CONTINUOUS, N_MODE_EXPOSURE_BRACKETING = 1 << 0, 1 << 4

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
_MAX_IFD_ENTRIES = 1024
_CR3_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")


class _Tiff:
    """Lazy TIFF IFD reader over a seekable stream; `base` is the TIFF header offset."""

    def __init__(self, f: BinaryIO, base: int = 0) -> None:
        self.f, self.base = f, base
        f.seek(base)
        hdr = f.read(8)
        if len(hdr) < 8 or hdr[:2] not in (b"II", b"MM"):
            raise ValueError("not a TIFF structure")
        self.e = "<" if hdr[:2] == b"II" else ">"
        # Magic is 42 for TIFF but ORF (0x4F52/0x5352) and RW2 (0x55) differ; accept any.
        (self.ifd0,) = struct.unpack(self.e + "I", hdr[4:8])

    def _read(self, off: int, n: int) -> bytes:
        self.f.seek(self.base + off)
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError("truncated TIFF")
        return data

    def next_ifd(self, off: int) -> int:
        """Offset of the IFD chained after the one at `off` (0 when none)."""
        if off <= 0:
            return 0
        (count,) = struct.unpack(self.e + "H", self._read(off, 2))
        if count > _MAX_IFD_ENTRIES:
            return 0
        (nxt,) = struct.unpack(self.e + "I", self._read(off + 2 + count * 12, 4))
        return nxt

    def ifd(self, off: int) -> dict[int, object]:
        if off <= 0:
            return {}
        (count,) = struct.unpack(self.e + "H", self._read(off, 2))
        if count > _MAX_IFD_ENTRIES:
            raise ValueError("implausible IFD")
        raw = self._read(off + 2, count * 12)
        out: dict[int, object] = {}
        for i in range(count):
            tag, typ, cnt = struct.unpack(self.e + "HHI", raw[i * 12 : i * 12 + 8])
            size = _TYPE_SIZE.get(typ)
            if size is None:
                continue
            nbytes = size * cnt
            field = raw[i * 12 + 8 : i * 12 + 12]
            if tag in (T_RW2_JPEG, T_MAKERNOTE) and typ == 7:  # keep embedded blobs as (offset, length)
                (voff,) = struct.unpack(self.e + "I", field)
                out[tag] = (voff, nbytes)
                continue
            if nbytes > 4096:  # skip large blobs (maker notes, tables)
                continue
            if nbytes <= 4:
                data = field[:nbytes]
            else:
                (voff,) = struct.unpack(self.e + "I", field)
                try:
                    data = self._read(voff, nbytes)
                except ValueError:
                    continue
            out[tag] = self._decode(typ, cnt, data)
        return out

    def _decode(self, typ: int, cnt: int, data: bytes) -> object:
        e = self.e
        if cnt == 0:
            return None
        if typ == 2:
            return data.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
        if typ in (1, 7):
            return data if cnt > 1 else data[0]
        if typ == 6:
            return struct.unpack(f"{e}{cnt}b", data)[0]
        fmt = {3: "H", 4: "I", 8: "h", 9: "i", 11: "f", 12: "d", 13: "I"}.get(typ)
        if fmt:
            vals = struct.unpack(f"{e}{cnt}{fmt}", data)
            return vals[0] if cnt == 1 else vals
        if typ in (5, 10):
            vals = struct.unpack(f"{e}{cnt * 2}{'I' if typ == 5 else 'i'}", data)
            rats = [vals[i] / vals[i + 1] if vals[i + 1] else None for i in range(0, len(vals), 2)]
            return rats[0] if cnt == 1 else rats
        return None


# --- container front-ends ---------------------------------------------------

def _jpeg_exif(f: BinaryIO, start: int = 0) -> bytes | None:
    """Return the TIFF payload of the first APP1 Exif segment of a JPEG at `start`."""
    f.seek(start)
    if f.read(2) != b"\xff\xd8":
        return None
    while True:
        marker = f.read(2)
        if len(marker) < 2 or marker[0] != 0xFF:
            return None
        m = marker[1]
        if m == 0xFF:  # padding
            f.seek(-1, io.SEEK_CUR)
            continue
        if m in (0xD9, 0xDA):  # EOI / start of scan: no more metadata
            return None
        (length,) = struct.unpack(">H", f.read(2))
        if m == 0xE1:
            seg = f.read(length - 2)
            if seg.startswith(b"Exif\x00\x00"):
                return seg[6:]
        else:
            f.seek(length - 2, io.SEEK_CUR)


def _iter_boxes(f: BinaryIO, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack(">I4s", hdr)
        hlen = 8
        if size == 1:
            (size,) = struct.unpack(">Q", f.read(8))
            hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen:
            return
        yield typ, pos + hlen, pos + size
        pos += size


def _cr3_tags(f: BinaryIO) -> tuple[dict, dict]:
    f.seek(0, io.SEEK_END)
    file_end = f.tell()
    for typ, s, e in _iter_boxes(f, 0, file_end):
        if typ != b"moov":
            continue
        for ctyp, cs, ce in _iter_boxes(f, s, e):
            if ctyp != b"uuid":
                continue
            f.seek(cs)
            if f.read(16) != _CR3_UUID:
                continue
            ifd0: dict = {}
            exif: dict = {}
            for gtyp, gs, ge in _iter_boxes(f, cs + 16, ce):
                if gtyp == b"CMT1":
                    t = _Tiff(f, gs)
                    ifd0 = t.ifd(t.ifd0)
                elif gtyp == b"CMT2":
                    t = _Tiff(f, gs)
                    exif = t.ifd(t.ifd0)
            return ifd0, exif
    raise ValueError("CR3 metadata boxes not found")


def _tiff_tags(f: BinaryIO, base: int = 0, depth: int = 0) -> tuple[dict, dict]:
    t = _Tiff(f, base)
    ifd0 = t.ifd(t.ifd0)
    exif_off = ifd0.get(T_EXIF_IFD)
    exif = t.ifd(exif_off) if isinstance(exif_off, int) else {}
    if not exif and T_RW2_JPEG in ifd0 and depth == 0:
        # Panasonic RW2 keeps full EXIF in an embedded JPEG.
        off, _ = ifd0[T_RW2_JPEG]  # type: ignore[misc]
        payload = _jpeg_exif(f, base + off)
        if payload:
            sub0, subexif = _tiff_tags(io.BytesIO(payload), 0, depth + 1)
            ifd0 = {**sub0, **ifd0}
            exif = subexif
    make = ifd0.get(T_MAKE)
    note = exif.get(T_MAKERNOTE)
    if isinstance(make, str) and make.upper().startswith("NIKON") and isinstance(note, tuple):
        try:
            exif["nikon"] = _nikon_makernote(f, base + note[0])
        except (ValueError, struct.error):
            pass
    return ifd0, exif


def _nikon_makernote(f: BinaryIO, start: int) -> dict:
    """Read a few tags from a Nikon type-3 maker note ("Nikon\0" + version + embedded TIFF header).

    Only unencrypted tags are used: ShootingMode (bitmask), ExposureBracketValue (this
    frame's slot in the bracket program) and ShutterCount.
    """
    f.seek(start)
    if f.read(6) != b"Nikon\x00":
        raise ValueError("not a Nikon type-3 maker note")
    t = _Tiff(f, start + 10)  # offsets inside the note are relative to its own TIFF header
    tags = t.ifd(t.ifd0)
    keep = (N_EXPOSURE_BRACKET_VALUE, N_SHOOTING_MODE, N_SHUTTER_COUNT, N_SERIAL, N_FOCUS_MODE)
    return {k: tags[k] for k in keep if k in tags}


def read_tags(path: Path) -> tuple[dict, dict]:
    """Return (IFD0, ExifIFD) tag dictionaries for an image file."""
    with open(path, "rb", buffering=0) as raw:
        f = io.BufferedReader(raw, buffer_size=16384)
        head = f.read(16)
        if head[:2] == b"\xff\xd8":
            payload = _jpeg_exif(f)
            if payload is None:
                raise ValueError("no EXIF segment")
            return _tiff_tags(io.BytesIO(payload))
        if head[:2] in (b"II", b"MM"):
            return _tiff_tags(f)
        if head.startswith(b"FUJIFILMCCD-RAW"):
            f.seek(84)
            (jpeg_off,) = struct.unpack(">I", f.read(4))
            payload = _jpeg_exif(f, jpeg_off)
            if payload is None:
                raise ValueError("RAF without embedded EXIF")
            return _tiff_tags(io.BytesIO(payload))
        if head[4:8] == b"ftyp" and head[8:12] == b"crx ":
            return _cr3_tags(f)
        if not head.strip(b"\x00"):
            raise ValueError("file starts with binary zeros (empty or corrupt)")
        raise ValueError("unsupported container")


# --- normalisation ------------------------------------------------------------

def _parse_time(dto: object, subsec: object, offset: object) -> tuple[float | None, bool]:
    if not isinstance(dto, str) or len(dto) < 19:
        return None, False
    try:
        dt = datetime.strptime(dto[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None, False
    tz = timezone.utc
    if isinstance(offset, str) and len(offset) >= 6 and offset[0] in "+-":
        try:
            sign = 1 if offset[0] == "+" else -1
            tz = timezone(sign * timedelta(hours=int(offset[1:3]), minutes=int(offset[4:6])))
        except ValueError:
            pass
    ts = dt.replace(tzinfo=tz).timestamp()
    precise = False
    if isinstance(subsec, str):
        digits = "".join(ch for ch in subsec if ch.isdigit())
        if digits:
            ts += int(digits) / (10 ** len(digits))
            precise = True
    return ts, precise


def _offset(v: object) -> float | None:
    if isinstance(v, str) and len(v) >= 6 and v[0] in "+-" and v[3] == ":":
        try:
            return (1 if v[0] == "+" else -1) * (int(v[1:3]) * 3600 + int(v[4:6]) * 60)
        except ValueError:
            return None
    return None


def _num(v: object) -> float | None:
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if isinstance(v, (int, float)) and v == v:
        return float(v)
    return None


def _str(v: object) -> str | None:
    return v if isinstance(v, str) and v else None


def meta_from_tags(path: Path, ifd0: dict, exif: dict) -> FileMeta:
    t, precise = _parse_time(exif.get(T_DTO) or ifd0.get(T_DATETIME), exif.get(T_SUBSEC_DTO), exif.get(T_OFFSET_DTO))
    iso = _num(exif.get(T_ISO))
    if iso is None or iso >= 65535:  # ISO overflows the SHORT field above 65535
        iso = _num(exif.get(T_ISOSPEED)) or _num(exif.get(T_REI)) or iso
    flash = _num(exif.get(T_FLASH))
    dist = _num(exif.get(T_SUBJ_DIST))
    nikon = exif.get("nikon") or {}
    bracket_kind = drive = None
    slot = shutter_count = None
    mode = nikon.get(N_SHOOTING_MODE)
    if isinstance(mode, int):
        bracket_kind = "exposure" if mode & N_MODE_EXPOSURE_BRACKETING else "none"
        drive = "continuous" if mode & N_MODE_CONTINUOUS else "single"
        if bracket_kind == "exposure":
            v = _num(nikon.get(N_EXPOSURE_BRACKET_VALUE))
            slot = v if v is not None and abs(v) <= 10 else None
    focus_mode = nikon.get(N_FOCUS_MODE)
    if isinstance(focus_mode, str) and "shift" in focus_mode.lower() and bracket_kind != "exposure":
        bracket_kind = "focus"
    sc = nikon.get(N_SHUTTER_COUNT)
    if isinstance(sc, int) and sc > 0:
        shutter_count = sc
    serial = _str(exif.get(T_BODY_SERIAL)) or _str(nikon.get(N_SERIAL))
    return FileMeta(
        path=path,
        make=_str(ifd0.get(T_MAKE)),
        model=_str(ifd0.get(T_MODEL)),
        serial=serial,
        lens=_str(exif.get(T_LENS_MODEL)),
        capture_time=t,
        time_precise=precise,
        exposure_time=_num(exif.get(T_EXPTIME)),
        f_number=_num(exif.get(T_FNUMBER)),
        iso=iso,
        exposure_bias=_num(exif.get(T_BIAS)),
        exposure_mode=int(m) if (m := _num(exif.get(T_EXPMODE))) is not None else None,
        exposure_program=int(p) if (p := _num(exif.get(T_PROGRAM))) is not None else None,
        focal_length=_num(exif.get(T_FOCAL)),
        white_balance=int(w) if (w := _num(exif.get(T_WB))) is not None else None,
        flash_fired=bool(int(flash) & 1) if flash is not None else None,
        focus_distance=dist if dist is not None and 0 < dist < 1e6 else None,
        bracket_kind=bracket_kind,
        bracket_value=slot,
        drive_mode=drive,
        shutter_count=shutter_count,
        utc_offset=_offset(exif.get(T_OFFSET_DTO)),
        reader="native",
    )


def read_native(path: Path) -> FileMeta:
    try:
        ifd0, exif = read_tags(path)
    except (OSError, ValueError, struct.error) as exc:
        return FileMeta(path=path, reader="native", error=str(exc) or type(exc).__name__)
    return meta_from_tags(path, ifd0, exif)
