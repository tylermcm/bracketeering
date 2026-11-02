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
import math
from array import array
from functools import lru_cache
import operator
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
    scene_lookback: int = 4  # compare with this many recent groups, not just the previous one
    hash_threshold: int = 12  # dHash fallback only: bits (of 64) that map to scene_similarity
    scene_similarity: float = 0.5  # preview similarity (-1..1) at or above which groups are one scene (= SAME_SCENE)
    anchor_similarity: float = 0.35  # ...and never links a group clearly unlike the scene's first one (= DIFFERENT)


@dataclass(slots=True, frozen=True)
class PixelStats:
    dhash: int
    highlights: float  # fraction of pixels with any channel clipped
    shadows: float  # fraction of pixels with crushed luma
    mean: float  # mean luma, 0..1
    flag: str  # "ok", "over", "under"
    thumb: bytes | None = None  # THUMB_W x THUMB_H grayscale, for shift-tolerant similarity

    def to_dict(self) -> dict:
        return {"highlights_clipped": round(self.highlights, 4), "shadows_crushed": round(self.shadows, 4),
                "mean": round(self.mean, 3), "flag": self.flag}


# --------------------------------------------------------------------------- similarity
#
# "Same composition, drifted a bit" vs "recomposed / different subject". Whole-image correlation at
# low resolution mostly compares layout (sky above, ground below), which a recompose of the same
# place keeps, and dHash flips bits on fine texture. Instead, like phase correlation, compare
# *structure*: band-pass the thumbnail (keep detail at the scale of objects, drop overall
# brightness/layout and pixel-level texture), then take the best normalised cross-correlation over
# shifts of up to ~12 % of the frame (coarse search at half resolution, refined at full). Frames that
# register score high; recomposed frames and other subjects don't. Brightness/contrast changes
# (exposure brackets) cancel out in the normalisation.

THUMB_W, THUMB_H = 64, 44
COARSE = 2  # coarse search at 1/COARSE resolution
SHIFT = 4  # coarse pixels each way at 32x22: +-4/32 = +-12 % of the frame
SAME_SCENE = 0.5  # similarity at or above: same composition / subject
DIFFERENT = 0.35  # similarity below: recomposed or a different subject


def band_pass(gray) -> bytes:
    """THUMB_W x THUMB_H band-passed thumbnail of a grayscale PIL image (128 = no structure)."""
    from PIL import Image, ImageChops, ImageFilter

    small = gray.resize((THUMB_W, THUMB_H), Image.Resampling.BOX).filter(ImageFilter.GaussianBlur(0.7))
    return ImageChops.subtract(small, small.filter(ImageFilter.BoxBlur(4)), scale=1, offset=128).tobytes()


@lru_cache(maxsize=8192)
def _coarse(thumb: bytes) -> bytes:
    w, h = THUMB_W // COARSE, THUMB_H // COARSE
    out = bytearray()
    for y in range(h):
        r0, r1 = 2 * y * THUMB_W, (2 * y + 1) * THUMB_W
        for x in range(w):
            c = 2 * x
            out.append((thumb[r0 + c] + thumb[r0 + c + 1] + thumb[r1 + c] + thumb[r1 + c + 1]) // 4)
    return bytes(out)


@lru_cache(maxsize=2048)  # ~12 KB per fine window: bounded memory on any shoot size
def _window(img: bytes, width: int, margin_x: int, margin_y: int, dx: int, dy: int) -> array:
    """Mean-removed, unit-norm window of `img` (row width `width`) inset by the margins, shifted."""
    height = len(img) // width
    vals: list[int] = []
    for y in range(margin_y + dy, height - margin_y + dy):
        row = y * width
        vals.extend(img[row + margin_x + dx : row + width - margin_x + dx])
    n = len(vals)
    mean = sum(vals) / n
    centred = [v - mean for v in vals]
    norm = math.sqrt(sum(v * v for v in centred)) or 1.0
    return array("d", (v / norm for v in centred))


def _ncc(a: bytes, b: bytes, width: int, margin_x: int, margin_y: int, dx: int, dy: int) -> float:
    return sum(map(operator.mul, _window(a, width, margin_x, margin_y, 0, 0), _window(b, width, margin_x, margin_y, dx, dy)))


def thumb_similarity(a: bytes, b: bytes) -> float:
    """How well two band-passed thumbnails register (-1..1), allowing ~12 % drift."""
    if len(a) != len(b) or len(a) != THUMB_W * THUMB_H:
        return -1.0
    ca, cb = _coarse(a), _coarse(b)
    cw = THUMB_W // COARSE
    best, bx, by = -2.0, 0, 0
    for dy in range(-SHIFT // 2, SHIFT // 2 + 1):  # frames drift sideways more than vertically
        for dx in range(-SHIFT, SHIFT + 1):
            v = _ncc(ca, cb, cw, SHIFT, SHIFT // 2, dx, dy)
            if v > best:
                best, bx, by = v, dx, dy
    fine = -2.0
    mx, my = SHIFT * COARSE + 1, (SHIFT // 2) * COARSE + 1
    for dy in range(by * COARSE - 1, by * COARSE + 2):
        for dx in range(bx * COARSE - 1, bx * COARSE + 2):
            fine = max(fine, _ncc(a, b, THUMB_W, mx, my, dx, dy))
    return fine


def similarity(a, b) -> float | None:
    """Similarity of two shots' previews, or None when either wasn't analysed."""
    ta, tb = getattr(a.pixels, "thumb", None), getattr(b.pixels, "thumb", None)
    if ta is None or tb is None:
        return None
    return thumb_similarity(ta, tb)


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
    thumb = band_pass(gray)
    return PixelStats(_dhash(gray), highlights, shadows, mean, flag, thumb)


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


def _looks_alike(a: Group, b: Group, cfg: PixelConfig) -> float | None:
    """Similarity of two groups' representatives in 0..1-ish (None when unknown)."""
    sa, sb = a.shots[a.representative], b.shots[b.representative]
    sim = similarity(sa, sb)
    if sim is not None:
        return sim
    pa, pb = sa.pixels, sb.pixels
    if pa is None or pb is None:
        return None
    # dHash fallback (e.g. stats computed elsewhere without a thumbnail): map bit distance to a score
    return 1.0 - _hamming(pa.dhash, pb.dhash) / (cfg.hash_threshold * 2.5)  # type: ignore[attr-defined]


def assign_scenes(groups: list[Group], cfg: PixelConfig = PixelConfig()) -> None:
    """Link look-alike groups shot close together (per camera body) into near-duplicate scenes.

    Each group is compared with the last `scene_lookback` groups from the same body within
    `scene_window` seconds, not just the previous one, so one odd frame (a recompose, someone
    walking through) doesn't cut a scene in two. The best match above SAME_SCENE wins.
    """
    recent: dict[tuple, list[Group]] = {}
    anchors: dict[int, Group] = {}  # scene id -> its first group
    next_id = 1
    threshold = cfg.scene_similarity
    for g in sorted(groups, key=lambda g: (g.shots[0].time is None, g.shots[0].time or 0.0, g.shots[0].name)):
        cam = g.shots[0].camera_key
        window = recent.setdefault(cam, [])
        t_now = g.shots[0].time
        best, best_sim = None, threshold
        for prev in reversed(window[-cfg.scene_lookback :]):
            t_prev = prev.shots[-1].time
            if t_prev is None or t_now is None or t_now - t_prev > cfg.scene_window:
                continue
            sim = _looks_alike(prev, g, cfg)
            if sim is not None and sim >= best_sim:
                best, best_sim = prev, sim
        if best is not None:
            # Guard against chaining: A~B, B~C, C~D can walk across subjects over many minutes.
            anchor_sim = _looks_alike(anchors[best.scene], g, cfg) if best.scene in anchors else None  # type: ignore[index]
            if anchor_sim is not None and anchor_sim < cfg.anchor_similarity:
                best = None
        if best is not None:
            g.scene = best.scene
        else:
            g.scene = next_id
            anchors[next_id] = g
            next_id += 1
        window.append(g)
