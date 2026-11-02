"""Core data types: per-file metadata, captures (shots), and detected groups."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

RAW_EXTENSIONS = frozenset(
    {
        ".3fr", ".arw", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf", ".fff",
        ".iiq", ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef",
        ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f",
    }
)
RENDERED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif", ".hif"})
IMAGE_EXTENSIONS = RAW_EXTENSIONS | RENDERED_EXTENSIONS

_FILE_NUMBER_RE = re.compile(r"(\d{3,})(?!.*\d)")


def file_number_from_name(name: str) -> int | None:
    """Extract the camera frame counter from a filename (IMG_1234 -> 1234)."""
    m = _FILE_NUMBER_RE.search(Path(name).stem)
    return int(m.group(1)) if m else None


@dataclass(slots=True)
class FileMeta:
    """Normalised metadata for a single image file.

    Every field is optional: readers fill what they can and the engine
    degrades gracefully when data is missing.
    """

    path: Path
    make: str | None = None
    model: str | None = None
    serial: str | None = None
    lens: str | None = None
    capture_time: float | None = None  # seconds; offset-aware when the file has an offset
    time_precise: bool = False  # True when sub-second time was available
    exposure_time: float | None = None  # seconds
    f_number: float | None = None
    iso: float | None = None
    exposure_bias: float | None = None  # EV
    exposure_mode: int | None = None  # EXIF 0xA402: 0 auto, 1 manual, 2 auto-bracket
    exposure_program: int | None = None  # EXIF 0x8822
    focal_length: float | None = None  # mm
    white_balance: int | None = None  # EXIF 0xA403: 0 auto, 1 manual
    flash_fired: bool | None = None
    focus_distance: float | None = None  # metres
    # Maker-note derived hints (exiftool backend only).
    bracket_kind: str | None = None  # "exposure", "focus", "flash", "wb", "iso", "none"
    bracket_position: int | None = None  # 1-based position within a bracket
    bracket_value: float | None = None  # this frame's slot in the camera's bracket program, EV (maker notes)
    bracket_program: int | None = None  # frames per bracket cycle the camera was set to (Nikon BracketProgram)
    drive_mode: str | None = None  # "continuous" or "single" (maker notes)
    shutter_count: int | None = None  # body's actuation counter (maker notes): identifies a capture
    interval_shooting: bool | None = None  # camera's interval timer was on (maker notes)
    utc_offset: float | None = None  # seconds east of UTC that the camera's local time was in
    sequence_number: int | None = None  # frame index within a continuous-drive burst (maker notes)
    file_number: int | None = None
    reader: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.file_number is None:
            self.file_number = file_number_from_name(self.path.name)

    @property
    def is_raw(self) -> bool:
        return self.path.suffix.lower() in RAW_EXTENSIONS

    def completeness(self) -> int:
        return sum(
            v is not None
            for v in (
                self.capture_time, self.exposure_time, self.f_number, self.iso,
                self.exposure_bias, self.exposure_mode, self.focal_length, self.make,
            )
        )


# Cameras write *nominal* 1/3-stop values: rounded labels for exact powers of
# two. f/13 is really f/12.7 and 1/125 s is really 1/128 s. Between f/9 and
# f/16 the labels are off by up to 0.15 EV, enough to make a 1/3-stop
# aperture bracket look irregular. These tables map each label back to its
# exact stop (in thirds). Values not in a table (half-stop labels like f/4.8
# or 1/90, or exact values some bodies write) are used as recorded.
_NOMINAL_F = (
    1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 9.0,
    10, 11, 13, 14, 16, 18, 20, 22, 25, 29, 32, 36, 40, 45, 51, 57, 64,
)
_NOMINAL_T_FAST = (  # denominators of 1/x s, from 1/8000 s down to 1/4 s
    8000, 6400, 5000, 4000, 3200, 2500, 2000, 1600, 1250, 1000, 800, 640, 500, 400, 320, 250, 200, 160,
    125, 100, 80, 60, 50, 40, 30, 25, 20, 15, 13, 10, 8, 6, 5, 4,
)
_NOMINAL_T_SLOW = (0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.3, 1.6, 2, 2.5, 3.2, 4, 5, 6, 8, 10, 13, 15, 20, 25, 30)
_NOMINAL_ISO = (
    50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 4000,
    5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600, 32000, 40000, 51200, 64000, 80000, 102400,
)


def _thirds(x: float) -> float:
    return round(x * 3) / 3


def log2_shutter(t: float) -> float:
    lt = math.log2(t)
    if t < 0.3:
        return _thirds(lt) if any(abs(1 / t - d) <= 0.005 * d for d in _NOMINAL_T_FAST) else lt
    return _thirds(lt) if any(abs(t - n) <= 0.005 * n for n in _NOMINAL_T_SLOW) else lt


def log2_aperture_stops(n: float) -> float:
    """Stops of light lost to the aperture, 2*log2(N), snapped for nominal f-numbers."""
    stops = 2.0 * math.log2(n)
    return _thirds(stops) if any(abs(n - f) <= 0.005 * f for f in _NOMINAL_F) else stops


def log2_iso(iso: float) -> float:
    li = math.log2(iso / 100.0)
    return _thirds(li) if any(abs(iso - i) <= 0.005 * i for i in _NOMINAL_ISO) else li


def _relative_ev(m: FileMeta) -> float | None:
    """Relative image brightness in stops (higher = brighter).

    log2(t) + log2(ISO/100) - 2*log2(N), with nominal labels snapped to exact
    1/3 stops. Unlike "EV" this captures every leg of the exposure triangle,
    so shutter, aperture (shutter-priority AEB) and ISO brackets all register.
    Only comparable between shots with the same exposure_mask.
    """
    if m.exposure_time is None or m.exposure_time <= 0:
        return None
    v = log2_shutter(m.exposure_time)
    if m.iso is not None and m.iso > 0:
        v += log2_iso(m.iso)
    if m.f_number is not None and m.f_number > 0:
        v -= log2_aperture_stops(m.f_number)
    return v


@dataclass(slots=True)
class Shot:
    """One physical capture. May be backed by several files (RAW + JPEG sidecar)."""

    files: list[FileMeta]
    meta: FileMeta = field(init=False)
    ev: float | None = field(init=False)
    exposure_mask: tuple[bool, bool, bool] = field(init=False)
    pixels: object | None = field(default=None, init=False)  # pixels.PixelStats when analysed

    def __post_init__(self) -> None:
        # The RAW is the most trustworthy source; otherwise the richest file.
        self.meta = max(self.files, key=lambda f: (f.is_raw, -len(f.path.parts), f.completeness()))
        m = self.meta
        self.exposure_mask = (m.exposure_time is not None, m.f_number is not None, m.iso is not None)
        self.ev = _relative_ev(m)

    @property
    def paths(self) -> list[Path]:
        return [f.path for f in self.files]

    @property
    def name(self) -> str:
        return self.meta.path.name

    @property
    def camera_key(self) -> tuple[str, str, str]:
        m = self.meta
        return ((m.make or "").strip().lower(), (m.model or "").strip().lower(), (m.serial or "").strip())

    @property
    def time(self) -> float | None:
        return self.meta.capture_time


def frame_step(a: Shot, b: Shot) -> int | None:
    """How many captures apart b is from a: shutter count when both have one, else file number.

    The body's shutter count is continuous; file numbers restart (new folder, focus-shift
    sequences, 9999 -> 0001) or switch series (_DSC8xxx / _DSC0xxx), which mustn't break a burst.
    """
    ca, cb = a.meta.shutter_count, b.meta.shutter_count
    if ca is not None and cb is not None and a.camera_key == b.camera_key:
        return cb - ca
    fa, fb = a.meta.file_number, b.meta.file_number
    if fa is None or fb is None:
        return None
    d = fb - fa
    if d < 0 and fa > 9000 and fb < 1000:
        d += 10000 if fa < 10000 else 100000
    return d


@dataclass(slots=True)
class Group:
    """A detected bracket (or a single, ungrouped shot when kind == "single")."""

    kind: str  # "exposure", "focus", "burst", "single"
    shots: list[Shot]
    score: float = 0.0
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    representative: int = 0  # index into shots of the frame that stands for the group
    best_exposed: int | None = None  # exposure brackets with pixel stats: exposed-to-the-right pick
    scene: int | None = None  # near-duplicate set id (pixel analysis)

    @property
    def size(self) -> int:
        return len(self.shots)

    @property
    def slots(self) -> list[float] | None:
        """The camera's own bracket offsets for each frame (maker notes), if every frame has one."""
        vals = [s.meta.bracket_value for s in self.shots]
        return None if any(v is None for v in vals) else vals  # type: ignore[return-value]

    @property
    def ev_offsets(self) -> list[float]:
        """Exposure of each frame in capture order, in EV.

        From the camera's bracket program when known (offsets from the metered frame, immune to
        Auto ISO re-metering), otherwise measured, relative to the group's median frame.
        """
        if (slots := self.slots) is not None:
            return [round(v, 2) + 0.0 for v in slots]
        evs = [s.ev for s in self.shots]
        if any(e is None for e in evs):
            return []
        ordered = sorted(evs)  # type: ignore[arg-type]
        mid = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
        return [round(e - mid, 2) + 0.0 for e in evs]  # type: ignore[operator]

    @property
    def step_ev(self) -> float | None:
        vals = self.slots or [s.ev for s in self.shots if s.ev is not None]
        evs = sorted(set(round(v, 3) for v in vals))  # type: ignore[arg-type]
        if len(evs) < 2:
            return None
        diffs = sorted(b - a for a, b in zip(evs, evs[1:]))
        if self.slots is not None:
            return round(diffs[0], 2)  # program step: smallest gap between slots
        return round(diffs[len(diffs) // 2], 2)
