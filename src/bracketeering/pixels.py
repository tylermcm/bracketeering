"""Optional pixel analysis on embedded previews (requires Pillow).

Adds, on top of the metadata-only engine:

* per-frame exposure stats: highlight clipping, crushed shadows, mean level,
  plus a coarse ok / over / under flag;
* `best_exposed` for each exposure bracket: the brightest frame whose
  highlights still hold ("expose to the right"), which has the cleanest
  shadows of the usable frames;
* near-duplicate *scenes*: consecutive groups (singles, bursts, brackets)
  whose representative frames look alike, linked into one scene id.

Stats come from the camera's embedded JPEG (see preview.py), so highlight
clipping is measured conservatively relative to the RAW.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .model import Group, Shot
from .preview import preview_source


@dataclass(slots=True, frozen=True)
class PixelConfig:
    size: int = 256  # analyse previews downscaled to fit this box
    highlight_level: int = 250  # 8-bit value at or above which a channel counts as clipped
    shadow_level: int = 5  # luma at or below which a pixel counts as crushed
    over_fraction: float = 0.02  # > 2% clipped highlights -> "over"
    under_fraction: float = 0.40  # > 40% crushed shadows -> "under"
    under_mean: float = 0.07  # or mean level below this -> "under"
    ettr_tolerance: float = 0.005  # best_exposed: highlights clipped must stay below 0.5%
    scene_window: float = 120.0  # s: max pause between groups of one near-duplicate scene
    hash_threshold: int = 12  # max differing bits (of 64) between near-duplicates


@dataclass(slots=True, frozen=True)
class PixelStats:
    dhash: int
    highlights: float  # fraction of pixels with any channel clipped
    shadows: float  # fraction of pixels with crushed luma
    mean: float  # mean luma, 0..1
    flag: str  # "ok", "over", "under"

    def to_dict(self) -> dict:
        return {"highlights_clipped": round(self.highlights, 4), "shadows_crushed": round(self.shadows, 4),
                "mean": round(self.mean, 3), "flag": self.flag}


def available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def _dhash(gray) -> int:
    from PIL import Image

    small = gray.resize((9, 8), Image.Resampling.BOX)
    px = small.tobytes()  # mode L: one byte per pixel, row-major
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def analyse_file(path: Path, cfg: PixelConfig = PixelConfig()) -> PixelStats | None:
    from PIL import Image, ImageChops

    src = preview_source(path)
    if src is None:
        return None
    try:
        im = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src)
        im.draft("RGB", (cfg.size, cfg.size))  # JPEG: decode at 1/2..1/8 scale directly
        im = im.convert("RGB")
        im.thumbnail((cfg.size, cfg.size), Image.Resampling.BILINEAR)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    r, g, b = im.split()
    peak = ImageChops.lighter(ImageChops.lighter(r, g), b)
    gray = im.convert("L")
    n = im.width * im.height
    hist_peak, hist_gray = peak.histogram(), gray.histogram()
    highlights = sum(hist_peak[cfg.highlight_level :]) / n
    shadows = sum(hist_gray[: cfg.shadow_level + 1]) / n
    mean = sum(i * c for i, c in enumerate(hist_gray)) / n / 255
    if highlights > cfg.over_fraction:
        flag = "over"
    elif shadows > cfg.under_fraction or mean < cfg.under_mean:
        flag = "under"
    else:
        flag = "ok"
    return PixelStats(_dhash(gray), highlights, shadows, mean, flag)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def analyse_shots(shots: list[Shot], cfg: PixelConfig = PixelConfig(), workers: int | None = None) -> None:
    """Attach PixelStats to every shot (run before grouping, so bursts can split on subject changes)."""
    if not available():
        raise RuntimeError("pixel analysis needs Pillow: pip install pillow")
    workers = workers or min(32, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:  # Pillow releases the GIL while decoding
        for s, st in zip(shots, pool.map(lambda s: analyse_file(s.meta.path, cfg), shots)):
            s.pixels = st


def finish_groups(groups: list[Group], cfg: PixelConfig = PixelConfig()) -> None:
    """Fill best_exposed and scene on groups whose shots already carry PixelStats."""
    for g in groups:
        if g.kind == "exposure":
            g.best_exposed = pick_best_exposed(g, cfg)
    assign_scenes(groups, cfg)


def analyse_groups(groups: list[Group], cfg: PixelConfig = PixelConfig(), workers: int | None = None) -> None:
    """Pixel-analyse already-grouped shots, then fill best_exposed and scene."""
    analyse_shots([s for g in groups for s in g.shots], cfg, workers)
    finish_groups(groups, cfg)


def pick_best_exposed(g: Group, cfg: PixelConfig = PixelConfig()) -> int | None:
    """Brightest frame with highlights intact; the darkest frame if every frame clips."""
    stats = [s.pixels for s in g.shots]
    evs = [s.ev for s in g.shots]
    if any(st is None for st in stats) or any(e is None for e in evs):
        return None
    order = sorted(range(g.size), key=lambda i: evs[i])  # type: ignore[arg-type,return-value]
    usable = [i for i in order if stats[i].highlights <= cfg.ettr_tolerance]  # type: ignore[union-attr]
    return usable[-1] if usable else order[0]


def assign_scenes(groups: list[Group], cfg: PixelConfig = PixelConfig()) -> None:
    """Link consecutive look-alike groups (per camera body) into near-duplicate scenes."""
    last_by_cam: dict[tuple, Group] = {}
    next_id = 1
    for g in sorted(groups, key=lambda g: (g.shots[0].time is None, g.shots[0].time or 0.0, g.shots[0].name)):
        cam = g.shots[0].camera_key
        rep = g.shots[g.representative].pixels
        prev = last_by_cam.get(cam)
        joined = False
        if prev is not None and rep is not None:
            prev_rep = prev.shots[prev.representative].pixels
            t_prev, t_now = prev.shots[-1].time, g.shots[0].time
            close = t_prev is not None and t_now is not None and t_now - t_prev <= cfg.scene_window
            if close and prev_rep is not None and _hamming(prev_rep.dhash, rep.dhash) <= cfg.hash_threshold:  # type: ignore[union-attr,attr-defined]
                g.scene = prev.scene
                joined = True
        if not joined:
            g.scene = next_id
            next_id += 1
        last_by_cam[cam] = g
