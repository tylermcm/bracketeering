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
    anchor_similarity: float = 0.2  # ...and never links a group clearly unlike the scene's first one
    weak_similarity: float = 0.15  # structure this weak still links if the layout matches strongly...
    layout_similarity: float = 0.85  # ...at least this much...
    layout_window: float = 20.0  # ...and the frames are this close in time (s); also the featureless fallback


@dataclass(slots=True, frozen=True)
class PixelStats:
    dhash: int
    highlights: float  # fraction of pixels with any channel clipped
    shadows: float  # fraction of pixels with crushed luma
    mean: float  # mean luma, 0..1
    flag: str  # "ok", "over", "under"
    thumb: bytes | None = None  # THUMB_W x THUMB_H band-passed: structure, for registration
    layout: bytes | None = None  # LAYOUT_W x LAYOUT_H plain grayscale: overall tonal layout

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


MIN_DETAIL = 4.0  # std of the band-passed thumbnail (0-255): below this a frame is featureless


@lru_cache(maxsize=8192)
def detail(thumb: bytes) -> float:
    """How much structure a band-passed thumbnail holds (moon in an empty sky, a black misfire: ~0-3)."""
    n = len(thumb)
    mean = sum(thumb) / n
    return math.sqrt(sum((v - mean) ** 2 for v in thumb) / n)


def similarity(a, b) -> float | None:
    """Do two shots' previews register? -1..1, or None when unknown.

    Unknown when either wasn't analysed or is featureless: a moon in an empty sky or a black
    misfire has nothing to register, which says nothing about whether the shot changed.
    """
    ta, tb = getattr(a.pixels, "thumb", None), getattr(b.pixels, "thumb", None)
    if ta is None or tb is None or detail(ta) < MIN_DETAIL or detail(tb) < MIN_DETAIL:
        return None
    return thumb_similarity(ta, tb)


LAYOUT_W, LAYOUT_H = 28, 20
_LS = 2  # layout shift search, pixels each way (+-8 %)


def layout_similarity(a, b) -> float | None:
    """Correlation of the overall tonal layout (sky/ground/subject placement), -1..1, or None.

    Tolerant of handheld drift on fine texture where structure registration fails (ivy, foliage),
    but too tolerant on its own (a recompose of the same place keeps its layout), so it only ever
    supports a structural match, never replaces one.
    """
    la, lb = getattr(a.pixels, "layout", None), getattr(b.pixels, "layout", None)
    if la is None or lb is None or len(la) != len(lb):
        return None
    best = -1.0
    for dy in range(-_LS, _LS + 1):
        for dx in range(-_LS, _LS + 1):
            best = max(best, _ncc(la, lb, LAYOUT_W, _LS, _LS, dx, dy))
    return best


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
    layout = gray.resize((LAYOUT_W, LAYOUT_H), Image.Resampling.BOX).tobytes()
    return PixelStats(_dhash(gray), highlights, shadows, mean, flag, thumb, layout)


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
    """Similarity of two groups' representatives (structure first), or None when unknown."""
    sa, sb = a.shots[a.representative], b.shots[b.representative]
    sim = similarity(sa, sb)
    if sim is not None:
        return sim
    pa, pb = sa.pixels, sb.pixels
    if pa is None or pb is None or getattr(pa, "thumb", None) is not None:
        return None  # analysed but featureless: unknown
    # dHash fallback (stats computed elsewhere without a thumbnail): map bit distance to a score
    return 1.0 - _hamming(pa.dhash, pb.dhash) / (cfg.hash_threshold * 2.5)  # type: ignore[attr-defined]


def _same_scene(prev: Group, g: Group, cfg: PixelConfig) -> tuple[bool, float]:
    """(link?, strength) for two groups shot within the scene window."""
    sim = _looks_alike(prev, g, cfg)
    sa, sb = prev.shots[prev.representative], g.shots[g.representative]
    gap = (g.shots[0].time - prev.shots[-1].time) if g.shots[0].time is not None and prev.shots[-1].time is not None else None
    if sim is not None:
        if sim >= cfg.scene_similarity:
            return True, sim
        # Handheld drift on fine texture (ivy, foliage) defeats structure registration; a strong
        # layout match a few seconds later still means the same view.
        lay = layout_similarity(sa, sb)
        if sim >= cfg.weak_similarity and lay is not None and lay >= cfg.layout_similarity and gap is not None and gap <= cfg.layout_window:
            return True, sim
        return False, sim
    # Featureless frames (moon, black sky): fall back to camera settings and timing.
    ma, mb = sa.meta, sb.meta
    same_lens = (ma.focal_length is None or mb.focal_length is None
                 or abs(ma.focal_length - mb.focal_length) <= 0.05 * max(ma.focal_length, mb.focal_length))
    close_ev = sa.ev is None or sb.ev is None or abs(sa.ev - sb.ev) <= 1.5
    if gap is not None and gap <= cfg.layout_window and same_lens and close_ev:
        return True, 0.0
    return False, -1.0


def assign_scenes(groups: list[Group], cfg: PixelConfig = PixelConfig()) -> None:
    """Link look-alike groups shot close together (per camera body) into near-duplicate scenes.

    Each group is compared with the last `scene_lookback` groups from the same body within
    `scene_window` seconds, not just the previous one, so one odd frame (a recompose, someone
    walking through) doesn't cut a scene in two. The best match above SAME_SCENE wins.
    """
    recent: dict[tuple, list[Group]] = {}
    anchors: dict[int, Group] = {}  # scene id -> its first group
    next_id = 1
    for g in sorted(groups, key=lambda g: (g.shots[0].time is None, g.shots[0].time or 0.0, g.shots[0].name)):
        cam = g.shots[0].camera_key
        window = recent.setdefault(cam, [])
        t_now = g.shots[0].time
        best, best_sim = None, -2.0
        for prev in reversed(window[-cfg.scene_lookback :]):
            t_prev = prev.shots[-1].time
            if t_prev is None or t_now is None or t_now - t_prev > cfg.scene_window:
                continue
            link, strength = _same_scene(prev, g, cfg)
            if link and strength > best_sim:
                best, best_sim = prev, strength
        if best is not None:
            # Guard against chaining: A~B, B~C, C~D can walk across subjects over many minutes.
            # Only a *clear* mismatch with the scene's first group (structure and layout) breaks it.
            anchor = anchors.get(best.scene)  # type: ignore[arg-type]
            if anchor is not None and anchor is not best:
                a_sim = _looks_alike(anchor, g, cfg)
                a_lay = layout_similarity(anchor.shots[anchor.representative], g.shots[g.representative])
                if a_sim is not None and a_sim < cfg.anchor_similarity and (a_lay is None or a_lay < cfg.layout_similarity):
                    best = None
        if best is not None:
            g.scene = best.scene
        else:
            g.scene = next_id
            anchors[next_id] = g
            next_id += 1
        window.append(g)
