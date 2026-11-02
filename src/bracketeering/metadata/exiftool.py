"""exiftool backend: broadest format coverage plus maker-note bracket hints.

Files are processed in batches, one exiftool process per batch, with several
batches running in parallel. Maker-note tags that describe bracketing (Canon
BracketMode/BracketShotNumber, Sony ReleaseMode, Nikon ShootingMode, Fujifilm
AutoBracketing, Olympus DriveMode, Pentax AutoBracketing, Panasonic
BracketSettings, ...) are normalised into `bracket_kind` / `bracket_position`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..model import FileMeta
from .native import _parse_time

TAGS = [
    "-Make", "-Model", "-SerialNumber", "-InternalSerialNumber", "-LensModel", "-LensID",
    "-DateTimeOriginal", "-SubSecTimeOriginal", "-OffsetTimeOriginal",
    "-ExposureTime#", "-FNumber#", "-ISO#", "-ExposureCompensation#",
    "-EXIF:ExposureMode#", "-ExposureProgram#", "-FocalLength#", "-EXIF:WhiteBalance#",
    "-Flash#", "-FocusDistance#", "-SubjectDistance#", "-ApproximateFocusDistance#",
    "-*Bracket*", "-SequenceNumber", "-*ShotNumber*", "-DriveMode", "-ReleaseMode*",
    "-ShootingMode", "-FocusShift*", "-ContinuousDrive", "-BurstMode",
]

_MODE_KEYS = {"drivemode", "releasemode", "releasemode2", "releasemode3", "shootingmode", "bracketsettings"}
_POSITION_KEYS = {"bracketshotnumber", "sequencenumber", "shotnumberinsequence"}
_OFF_VALUES = {
    "", "off", "no", "none", "n/a", "no bracket", "0", "0 ev", "single", "single frame", "single shot",
    "disable", "disabled",
}
_SHOT_RE = re.compile(r"shot\s*(\d+)", re.I)


def available() -> bool:
    return shutil.which("exiftool") is not None


def _kind_from_text(text: str) -> str:
    t = text.lower()
    if "focus" in t:
        return "focus"
    if "white balance" in t or re.search(r"\bwb\b", t):
        return "wb"
    if "flash" in t or "feb" in t:
        return "flash"
    if re.search(r"\biso\b", t):
        return "iso"
    return "exposure"


_DRIVE_KEYS = ("continuousdrive", "drivemode", "releasemode", "releasemode2", "releasemode3", "shootingmode", "burstmode")
_CONTINUOUS_RE = re.compile(r"continuous|burst|sequential|high[- ]speed|low[- ]speed|\bc[hlm]\b|\bon\b|infinite|unlimited", re.I)
_SINGLE_RE = re.compile(r"single|normal|\boff\b|timer|remote", re.I)


def classify_drive(record: dict) -> tuple[str | None, int | None]:
    """Derive (drive_mode, sequence_number) from maker-note drive tags.

    Canon ContinuousDrive, Sony ReleaseMode*, Nikon ShootingMode, Fujifilm and
    Pentax DriveMode, Olympus DriveMode ("Continuous Shooting, Shot 3"),
    Panasonic BurstMode; SequenceNumber is the frame index within a burst.
    """
    lowered = {k.lower(): str(v) for k, v in record.items()}
    drive: str | None = None
    for key in _DRIVE_KEYS:
        v = lowered.get(key)
        if v is None:
            continue
        if _CONTINUOUS_RE.search(v) and not re.search(r"no burst|\bsingle\b", v, re.I):
            drive = "continuous"
            break
        if _SINGLE_RE.search(v):
            drive = drive or "single"
    seq: int | None = None
    raw = lowered.get("sequencenumber")
    if raw is not None:
        try:
            n = int(float(raw))
            seq = n if n > 0 else None
        except ValueError:
            seq = None
    if seq is None:
        for key in ("drivemode", "releasemode2", "releasemode3"):
            if (v := lowered.get(key)) and (m := _SHOT_RE.search(v)):
                seq = int(m.group(1))
                break
    if drive == "single":
        seq = None
    return drive, seq


def classify_bracket(record: dict) -> tuple[str | None, int | None]:
    """Derive (bracket_kind, bracket_position) from exiftool maker-note tags."""
    active: list[str] = []
    saw_off = False
    position: int | None = None
    for key, value in record.items():
        k = key.lower()
        is_mode = k in _MODE_KEYS
        if "bracket" not in k and not is_mode and k not in _POSITION_KEYS:
            continue
        v = str(value).strip()
        vl = v.lower()
        if k in _POSITION_KEYS or "shotnumber" in k:
            try:
                n = int(float(v))
                if n > 0:
                    position = n
            except ValueError:
                pass
            continue
        if m := _SHOT_RE.search(v):
            position = int(m.group(1))
        if is_mode and k != "bracketsettings":
            if "bracket" in vl:
                active.append(_kind_from_text(vl))
            continue
        # A tag whose *name* mentions bracketing.
        if vl in _OFF_VALUES or vl.startswith("off") or vl.startswith("no "):
            saw_off = True
            continue
        if k in ("aebbracketvalue", "exposurebracketvalue", "bracketvalue"):
            try:
                if abs(float(v.replace("EV", "").strip())) > 1e-6:
                    active.append("exposure")
                else:
                    saw_off = True
            except ValueError:
                pass
            continue
        if k == "bracketmode":
            active.append({"aeb": "exposure", "feb": "flash", "iso": "iso", "wb": "wb"}.get(vl, _kind_from_text(vl)))
            continue
        if any(s in k for s in ("step", "count", "increment", "width", "value", "shotnumber")):
            continue  # parameters, not on/off switches
        name_kind = _kind_from_text(k)
        active.append(name_kind if name_kind != "exposure" else _kind_from_text(vl))
    if active:
        for pref in ("exposure", "focus", "flash", "iso", "wb"):
            if pref in active:
                kind = pref
                break
        return kind, position if kind in ("exposure", "focus") else None
    return ("none" if saw_off else None), None


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def meta_from_record(rec: dict) -> FileMeta:
    path = Path(rec["SourceFile"])
    t, precise = _parse_time(
        rec.get("DateTimeOriginal"), str(rec.get("SubSecTimeOriginal", "")) or None, rec.get("OffsetTimeOriginal")
    )
    kind, pos = classify_bracket(rec)
    drive, seq = classify_drive(rec)
    flash = _f(rec.get("Flash"))
    dist = None
    for k in ("FocusDistance", "ApproximateFocusDistance", "SubjectDistance"):
        if (d := _f(rec.get(k))) is not None and 0 < d < 1e6:
            dist = d
            break
    serial = rec.get("SerialNumber") or rec.get("InternalSerialNumber")
    em, ep, wb = _f(rec.get("ExposureMode")), _f(rec.get("ExposureProgram")), _f(rec.get("WhiteBalance"))
    return FileMeta(
        path=path,
        make=rec.get("Make"),
        model=rec.get("Model"),
        serial=str(serial) if serial is not None else None,
        lens=rec.get("LensModel") or rec.get("LensID"),
        capture_time=t,
        time_precise=precise,
        exposure_time=_f(rec.get("ExposureTime")),
        f_number=_f(rec.get("FNumber")),
        iso=_f(rec.get("ISO")),
        exposure_bias=_f(rec.get("ExposureCompensation")),
        exposure_mode=int(em) if em is not None else None,
        exposure_program=int(ep) if ep is not None else None,
        focal_length=_f(rec.get("FocalLength")),
        white_balance=int(wb) if wb is not None else None,
        flash_fired=bool(int(flash) & 1) if flash is not None else None,
        focus_distance=dist,
        bracket_kind=kind,
        bracket_position=pos,
        drive_mode=drive,
        sequence_number=seq,
        reader="exiftool",
    )


def _run_batch(paths: list[Path]) -> list[FileMeta]:
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(str(p) for p in paths))
        argfile = fh.name
    try:
        proc = subprocess.run(
            ["exiftool", "-json", "-fast", "-charset", "filename=utf8", "-api", "LargeFileSupport=1", *TAGS, "-@", argfile],
            capture_output=True, check=False,
        )
    finally:
        os.unlink(argfile)
    try:
        records = json.loads(proc.stdout.decode("utf-8", "replace") or "[]")
    except json.JSONDecodeError:
        records = []
    by_path = {}
    for rec in records:
        if "SourceFile" in rec:
            m = meta_from_record(rec)
            by_path[str(m.path)] = m
    return [by_path.get(str(p)) or FileMeta(path=p, reader="exiftool", error="exiftool returned no data") for p in paths]


def read_exiftool(paths: list[Path], batch_size: int = 256, workers: int | None = None) -> list[FileMeta]:
    if not paths:
        return []
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
    workers = workers or min(len(batches), os.cpu_count() or 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [m for batch in pool.map(_run_batch, batches) for m in batch]
