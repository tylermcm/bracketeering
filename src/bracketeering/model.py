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


def _relative_ev(m: FileMeta) -> float | None:
    """Relative image brightness in stops (higher = brighter).

    log2(t) + log2(ISO/100) - 2*log2(N). Unlike "EV" this captures every leg
    of the exposure triangle, so shutter, aperture and ISO brackets all
    register. Only comparable between shots with the same exposure_mask.
    """
    if m.exposure_time is None or m.exposure_time <= 0:
        return None
    v = math.log2(m.exposure_time)
    if m.iso is not None and m.iso > 0:
        v += math.log2(m.iso / 100.0)
    if m.f_number is not None and m.f_number > 0:
        v -= 2.0 * math.log2(m.f_number)
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
        self.meta = max(self.files, key=lambda f: (f.is_raw, f.completeness()))
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
    def ev_offsets(self) -> list[float]:
        """Exposure of each frame relative to the bracket's median frame, in capture order."""
        evs = [s.ev for s in self.shots]
        if any(e is None for e in evs):
            return []
        ordered = sorted(evs)  # type: ignore[arg-type]
        mid = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
        return [round(e - mid, 2) for e in evs]  # type: ignore[operator]

    @property
    def step_ev(self) -> float | None:
        evs = sorted(s.ev for s in self.shots if s.ev is not None)
        if len(evs) < 2:
            return None
        diffs = sorted(b - a for a, b in zip(evs, evs[1:]))
        return round(diffs[len(diffs) // 2], 2)
