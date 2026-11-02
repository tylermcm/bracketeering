"""Burst detection: segment ungrouped frames into continuous-drive bursts.

A burst here means one press-and-hold of the shutter in continuous drive:
frames fired by the camera's drive mechanism, which a culler treats as
alternatives of one moment. Quick separate presses ("taps"), interval-timer
sequences and unrelated shots are *not* bursts, and two presses separated by
a short pause are two bursts.

Model
-----
Frames that could belong together are chained (same body and lens, frame
counter consecutive, <= max_link_gap apart). Every interval of a chain is
then explained as one of:

  drive  the camera's drive fired the next frame. The interval sits near
         the burst's *local* cadence: a few % jitter, plus occasional
         slowdowns (AF hunting, buffer filling). The first interval of a
         burst instead draws its cadence from a prior over drive rates
         (~1-60 fps).
  break  a separate press: a re-press after releasing the shutter, a tap, or
         an unrelated shot. Humans rarely re-press faster than ~0.2 s, and
         usually take 0.3-2 s.

Both are probability densities over the same intervals, so a labelling's log
likelihood compares directly across segmentations. The engine finds the
most likely labelling (bursts vs singles, and where bursts start and end)
by dynamic programming. Comparing with the local cadence (neighbouring
intervals) keeps a burst that slows down as the buffer fills together, and
still splits at an isolated long pause.

Without sub-second timestamps, intervals are whole seconds. The same model
then uses the probability of landing in the same or the next second.

Metadata adds log-odds when present: drive mode (continuous vs single),
the maker-note sequence number (a reset to 1 forces a new burst), zoom
changes, exposure jumps, frame-counter gaps, and (with --pixels) subject
changes between consecutive previews.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

from .model import Group, Shot, frame_step

_LN_SQRT_2PI = 0.5 * math.log(2 * math.pi)


@dataclass(frozen=True, slots=True)
class BurstModel:
    # drive: ln(interval / local cadence) ~ steady jitter + slowdown mixture
    jitter: float = 0.12
    slow_weight: float = 0.15
    slow_mu: float = 0.55
    slow_sd: float = 0.25
    # drive: prior over cadence (s/frame), log-normal
    drive_mu: float = math.log(0.12)
    drive_sd: float = 0.9
    max_drive_interval: float = 1.1  # s: continuous drive is essentially never slower than ~1 fps
    # break: separate presses, log-normal pause with a human floor
    break_mu: float = math.log(0.9)
    break_sd: float = 0.6
    human_floor: float = 0.2
    human_soft: float = 0.15
    burst_cost: float = 1.0  # prior cost of starting a burst
    pace_change: float = 1.0  # cost of a burst changing cadence (buffer full, drive mode switch)
    # per-interval metadata log-odds for "same press" (applied when an interval is labelled drive)
    seq_step: float = 2.0  # sequence number advances by exactly 1
    seq_mismatch: float = 2.0  # sequence number present but doesn't advance by 1 (penalty)
    drive_continuous: float = 1.0  # both frames shot in continuous drive
    drive_single: float = 3.0  # either frame shot in single drive (penalty)
    zoom: float = 0.7  # focal length changed (penalty)
    ev_jump: float = 2.0  # exposure jumped (penalty)
    ev_jump_threshold: float = 1.5  # EV
    frame_skip: float = 0.3  # per missing frame number (penalty)
    hash_same: float = 0.3  # previews look alike (pixels)
    hash_change: float = 1.0  # previews don't register (pixels, penalty): weak, since panning with a
    # moving subject or a fast-changing scene also fails to register; one hold stays one burst
    hash_same_max: int = 12
    refocus: float = 1.5  # focus distance jumped between frames at a slow pace (penalty)
    refocus_ratio: float = 1.4
    slow_pace: float = 0.5  # s: intervals at least this long count as a slow pace
    hash_change_min: int = 24
    # whole-second timestamps: P(break interval = 0, 1, >=2 s)
    break_seconds: tuple[float, float, float] = (0.2, 0.45, 0.35)
    same_second_decay: float = 0.5  # P(another separate press in the same second) shrinks per press
    edge_cost_seconds: float = 2.0  # whole-second stamps: a separate press right next to a burst is unlikely
    edge_cost: float = 2.0  # sub-second stamps: same prior, weaker because timing is informative
    drive_late: float = 0.01  # P(drive interval >= 2 s ticks)
    # boundary candidates: an interval this many times off its neighbours' cadence
    split_ratio: float = 1.3


@dataclass(slots=True)
class BurstConfig:
    max_link_gap: float = 2.5  # s start-to-start: frames further apart are never one burst
    min_frames: int = 2
    max_atoms: int = 64  # DP look-back in atoms
    focal_tol: float = 0.02
    max_frame_skip: int = 3
    model: BurstModel = field(default_factory=BurstModel)
    # Set per camera: "continuous drive" is evidence only if the body was also used in single drive.
    # Many shooters leave continuous on all day, and then every frame says so.
    drive_informative: bool = True


def _ln_norm(x: float, mu: float, sd: float) -> float:
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd) - _LN_SQRT_2PI


def _ln_sigmoid(z: float) -> float:
    return -math.log1p(math.exp(-z)) if z > -30 else z


def _ln_jitter(r: float, m: BurstModel) -> float:
    a = math.log(1 - m.slow_weight) + _ln_norm(r, 0.0, m.jitter)
    b = math.log(m.slow_weight) + _ln_norm(r, m.slow_mu, m.slow_sd)
    hi = max(a, b)
    return hi + math.log(math.exp(a - hi) + math.exp(b - hi))


def _ln_drive_prior(c: float, m: BurstModel) -> float:
    lc = math.log(max(c, 1e-3))
    return _ln_norm(lc, m.drive_mu, m.drive_sd) + _ln_sigmoid((math.log(m.max_drive_interval) - lc) / 0.1)


def _ln_break(iv: float, m: BurstModel) -> float:
    lg = math.log(max(iv, 1e-3))
    return _ln_norm(lg, m.break_mu, m.break_sd) + _ln_sigmoid((lg - math.log(m.human_floor)) / m.human_soft)


def _ln_break_seconds(d: int, m: BurstModel) -> float:
    return math.log(m.break_seconds[min(max(d, 0), 2)])


def _ln_drive_seconds(d: int, c: float, m: BurstModel) -> float:
    """P(whole-second difference d | drive cadence c), c in seconds/frame."""
    if c >= 1:
        p = {0: 0.5 / c, 1: 1 - 0.5 / c}.get(d, m.drive_late)
    else:
        p = {0: 1 - c, 1: c}.get(d, m.drive_late)
    return math.log(max(p, m.drive_late))


def _ev_close(a: Shot, b: Shot, m: BurstModel) -> bool:
    if a.ev is None or b.ev is None or a.exposure_mask != b.exposure_mask:
        return True
    return abs(a.ev - b.ev) <= m.ev_jump_threshold


def _subject_change(a: Shot, b: Shot, m: BurstModel) -> bool | None:
    """Did the camera swing to a different subject between a and b? None when previews are unknown."""
    from .pixels import DIFFERENT, similarity

    sim = similarity(a, b)
    if sim is not None:
        return sim < DIFFERENT
    h = _hamming(a, b)
    return None if h is None else h >= m.hash_change_min


def _alike(a: Shot, b: Shot, m: BurstModel) -> bool:
    from .pixels import SAME_SCENE, similarity

    sim = similarity(a, b)
    if sim is not None:
        return sim >= SAME_SCENE
    h = _hamming(a, b)
    return h is not None and h <= m.hash_same_max


def _hamming(a: Shot, b: Shot) -> int | None:
    pa, pb = getattr(a.pixels, "dhash", None), getattr(b.pixels, "dhash", None)
    if pa is None or pb is None:
        return None
    return (pa ^ pb).bit_count()


def linked(a: Shot, b: Shot, cfg: BurstConfig) -> bool:
    """Could b follow a in the same burst? (hard constraints only)"""
    ma, mb = a.meta, b.meta
    if a.camera_key != b.camera_key:
        return False
    if ma.lens and mb.lens and ma.lens != mb.lens:
        return False
    step = frame_step(a, b)
    if step is not None and not (0 <= step <= cfg.max_frame_skip):
        return False
    if a.time is None or b.time is None:
        return step == 1
    return b.time - a.time <= cfg.max_link_gap + (0 if ma.time_precise and mb.time_precise else 1.0)


@dataclass(slots=True)
class _Pair:
    """Evidence about the step from frame i to frame i+1 of a chain."""
    interval: float | None
    meta: float  # metadata log-odds that both frames come from the same press
    hard_split: bool  # sequence-number reset: never the same press
    candidate: bool  # a boundary (or pace change) may be placed here
    precise: bool  # both timestamps have sub-seconds
    same_second_run: int = 0  # whole-second stamps: how many consecutive 0 s steps end here
    subject_change: bool = False  # previews look different (pixels)
    exposure: float = 0.0  # exposure time of the first frame: a person can only re-press after it ends


def _pairs(chain: list[Shot], cfg: BurstConfig) -> list[_Pair]:
    m = cfg.model
    out: list[_Pair] = []
    for a, b in zip(chain, chain[1:]):
        ma, mb = a.meta, b.meta
        iv = (b.time - a.time) if a.time is not None and b.time is not None else None  # type: ignore[operator]
        meta, hard, cand = 0.0, False, False
        if ma.focal_length and mb.focal_length and abs(ma.focal_length - mb.focal_length) > cfg.focal_tol * max(ma.focal_length, mb.focal_length):
            meta -= m.zoom
            cand = True
        if not _ev_close(a, b, m):
            meta -= m.ev_jump
            cand = True
        fs = frame_step(a, b)
        if fs is not None and fs > 1:
            meta -= m.frame_skip * (fs - 1)
            cand = True
        da, db = ma.focus_distance, mb.focus_distance
        if iv is not None and iv >= m.slow_pace and da and db and max(da, db) / min(da, db) >= m.refocus_ratio:
            # At a slow pace, focus jumping between frames (199 m -> 100 m -> 67 m) means the
            # shooter half-pressed and pressed again; a held drive tracks a subject smoothly.
            meta -= m.refocus
            cand = True
        if iv is not None and ma.time_precise and mb.time_precise and abs(iv) < 0.002:
            hard = cand = True  # identical capture times: duplicates or exports, never a 500 fps burst
        sa, sb = ma.sequence_number, mb.sequence_number
        if sa is not None and sb is not None:
            if sb == sa + 1:
                meta += m.seq_step
            elif sb <= sa or sb == 1:
                hard = cand = True
            else:
                meta -= m.seq_mismatch
                cand = True
        elif sb == 1:
            cand = True
        if ma.drive_mode == "continuous" and mb.drive_mode == "continuous" and cfg.drive_informative:
            meta += m.drive_continuous
        elif "single" in (ma.drive_mode, mb.drive_mode):
            meta -= m.drive_single
            cand = True
        change = _subject_change(a, b, m)
        if change is not None:
            if not change and _alike(a, b, m):
                meta += m.hash_same
            elif change:
                meta -= m.hash_change
                cand = True
        precise = ma.time_precise and mb.time_precise and iv is not None
        if not precise:
            cand = cand or iv is None or iv >= 1
        run = 0
        if not precise and iv is not None and round(iv) == 0:
            run = (out[-1].same_second_run if out and not out[-1].precise else 0) + 1
        out.append(_Pair(iv, meta, hard, cand, precise, run, bool(change), ma.exposure_time or 0.0))
    # timing: an interval far off its neighbours' cadence may be a boundary or a pace change
    # (measured on dead time when slow shutters dominate the interval)
    ivs = [p.interval for p in out]
    exps = [s.meta.exposure_time or 0.0 for s in chain[:-1]]
    known = [x for x in ivs if x is not None]
    if known and max(exps) > 0.25 * median(known):
        ivs = [None if x is None else max(x - e, 0.01) for x, e in zip(ivs, exps)]
    for i, p in enumerate(out):
        if p.candidate or ivs[i] is None:
            p.candidate = True
            continue
        x = ivs[i]
        before = [v for v in ivs[max(0, i - 3) : i] if v is not None]
        after = [v for v in ivs[i + 1 : i + 4] if v is not None]
        sides = [median(v) for v in (before, after) if v]
        if not sides or x > m.split_ratio * min(sides) or x * m.split_ratio < max(sides):
            p.candidate = True
    return out


def _break(p: _Pair, m: BurstModel) -> float:
    """Log likelihood of interval p as a break between separate presses (or subjects)."""
    if p.interval is None:
        return 0.0
    if p.precise:
        # The pause a person needs runs from the end of the exposure: 0.4 s exposures fired
        # 0.53 s apart leave 0.13 s, far too quick for a release and re-press.
        return _ln_break(max(p.interval - p.exposure, 0.001), m)
    d = round(p.interval)
    if d <= 0:
        # Each further separate press landing in the same second is rarer than the last.
        return math.log(m.break_seconds[0] * m.same_second_decay ** max(p.same_second_run - 1, 0))
    return _ln_break_seconds(d, m)


def _drive(p: _Pair, cadence: float, m: BurstModel) -> float:
    """Log likelihood of interval p as a drive interval at `cadence` (no metadata)."""
    if p.interval is None:
        return -math.inf
    if p.precise:
        return _ln_jitter(math.log(max(p.interval, 0.002) / cadence), m)
    return _ln_drive_seconds(round(p.interval), cadence, m)


def _edge(p: _Pair, m: BurstModel) -> float:
    """Prior cost of a break touching a burst (only needed when timing is coarse)."""
    if p.subject_change or p.hard_split:
        return 0.0
    return m.edge_cost if p.precise else m.edge_cost_seconds


@dataclass(slots=True)
class _Piece:
    """A stretch of one burst at one cadence."""
    score: float
    cadence: float
    fatal: bool = False


def _piece(chain: list[Shot], pairs: list[_Pair], lo: int, hi: int, m: BurstModel) -> _Piece:
    """Score frames chain[lo..hi] as drive frames at a single cadence (>= 2 frames)."""
    seg = pairs[lo:hi]
    if any(p.hard_split for p in seg):
        return _Piece(-math.inf, 0.0, True)
    if all(p.precise for p in seg):
        ivs = [max(p.interval, 0.002) for p in seg]  # type: ignore[type-var]
        cadence = median(ivs)
        if cadence > m.max_drive_interval * 1.15:
            return _Piece(-math.inf, cadence)  # slower than any continuous drive: separate presses
        score = _ln_drive_prior(cadence, m)  # explains one interval; local jitter the other n-2
        if len(ivs) > 1:
            # With slow shutters the drive fires "exposure + fixed overhead" apart, so
            # intervals track the metered exposure; regularity lives in the dead time.
            exps = [s.meta.exposure_time or 0.0 for s in chain[lo:hi]]
            if max(exps) > 0.25 * cadence:
                ivs = [max(iv - e, 0.01) for iv, e in zip(ivs, exps)]
            comp = 0.0
            for i, iv in enumerate(ivs):
                local = median(ivs[max(0, i - 3) : i] + ivs[i + 1 : i + 4])
                comp += _ln_jitter(math.log(iv / local), m)
            score += comp * (len(ivs) - 1) / len(ivs)
    else:
        t0, t1 = chain[lo].time, chain[hi].time
        if t0 is None or t1 is None:
            return _Piece(-math.inf, 0.0, True)
        cadence = min(max((t1 - t0 + 0.5) / (hi - lo), 0.02), 3.0)
        score = _ln_drive_prior(cadence, m) + math.log(0.5) + sum(_ln_drive_seconds(round(p.interval or 0.0), cadence, m) for p in seg)
    return _Piece(score + sum(p.meta for p in seg), cadence)


def _confidence(score: float) -> float:
    return round(1.0 / (1.0 + math.exp(-score / 2.0)), 3)


def segment_chain(chain: list[Shot], cfg: BurstConfig) -> list[Group]:
    """Most likely labelling of a chain's frames as bursts (with possible pace changes) and singles."""
    if len(chain) < 2:
        return [Group("single", [s], confidence=1.0) for s in chain]
    m = cfg.model
    pairs = _pairs(chain, cfg)
    atoms: list[tuple[int, int]] = []
    start = 0
    for i, p in enumerate(pairs):
        if p.candidate:
            atoms.append((start, i))
            start = i + 1
    atoms.append((start, len(chain) - 1))
    k = len(atoms)
    NEG = -math.inf
    # best_s[j] / best_b[j]: best score for atoms[:j] ending in a single / inside a burst
    best_s = [NEG] * (k + 1)
    best_b = [NEG] * (k + 1)
    best_s[0] = 0.0
    # back pointers: (prev_j, prev_state, piece_kind) with piece_kind in {"S", "N", "C"}
    back_s: list[tuple[int, str]] = [(0, "s")] * (k + 1)
    # back_b: (prev_j, prev_state, piece_kind, cadence, piece contribution to the log likelihood)
    back_b: list[tuple[int, str, str, float, float]] = [(0, "s", "N", 0.0, 0.0)] * (k + 1)

    def lead(j: int) -> _Pair | None:
        return pairs[atoms[j][0] - 1] if j > 0 else None

    for j in range(1, k + 1):
        lo, hi = atoms[j - 1]
        # S: atom j-1 as separate presses
        inner = sum(_break(p, m) for p in pairs[lo:hi])
        lp = lead(j - 1)
        for prev_state, prev in (("s", best_s[j - 1]), ("b", best_b[j - 1])):
            v = prev + (_break(lp, m) if lp else 0.0) + inner
            if prev_state == "b" and lp is not None:
                v -= _edge(lp, m)
            if v > best_s[j]:
                best_s[j], back_s[j] = v, (j - 1, prev_state)
        # N / C: atoms i..j-1 as one cadence piece
        for i in range(j - 1, max(-1, j - 1 - cfg.max_atoms), -1):
            plo = atoms[i][0]
            if hi - plo + 1 < cfg.min_frames:
                continue
            piece = _piece(chain, pairs, plo, hi, m)
            if piece.fatal:
                break  # a sequence reset inside stays inside every longer piece
            lp = lead(i)
            # N: a new press
            contrib = piece.score - m.burst_cost - (_edge(lp, m) if lp is not None else 0.0)
            for prev_state, prev in (("s", best_s[i]), ("b", best_b[i])):
                v = prev + (_break(lp, m) if lp else 0.0) + contrib
                if v > best_b[j]:
                    best_b[j], back_b[j] = v, (i, prev_state, "N", piece.cadence, contrib)
            # C: the same press continues at a new cadence
            if lp is not None and not lp.hard_split and best_b[i] > NEG:
                contrib = _drive(lp, piece.cadence, m) + lp.meta - m.pace_change + piece.score
                v = best_b[i] + contrib
                if v > best_b[j]:
                    best_b[j], back_b[j] = v, (i, "b", "C", piece.cadence, contrib)

    # backtrack into pieces, then merge N + C* into burst groups
    pieces: list[tuple[str, int, int, float, float]] = []  # (kind, lo, hi, cadence, contribution)
    j, state = k, ("s" if best_s[k] >= best_b[k] else "b")
    while j > 0:
        if state == "s":
            i, state = back_s[j]
            pieces.append(("S", atoms[i][0], atoms[j - 1][1], 0.0, 0.0))
        else:
            i, prev_state, kind, cad, contrib = back_b[j]
            pieces.append((kind, atoms[i][0], atoms[j - 1][1], cad, contrib))
            state = prev_state
        j = i
    pieces.reverse()

    out: list[Group] = []
    idx = 0
    while idx < len(pieces):
        kind, lo, hi, cad, contrib = pieces[idx]
        if kind == "S":
            out.extend(Group("single", [s], confidence=1.0) for s in chain[lo : hi + 1])
            idx += 1
            continue
        cadences, total = [cad], contrib
        while idx + 1 < len(pieces) and pieces[idx + 1][0] == "C":
            idx += 1
            hi = pieces[idx][2]
            cadences.append(pieces[idx][3])
            total += pieces[idx][4]
        idx += 1
        # log-odds: one burst vs the same frames as separate presses
        margin = total - sum(_break(p, m) for p in pairs[lo:hi])
        out.append(_burst_group(chain, pairs, lo, hi, cadences, margin, cfg))
    return out


def _burst_group(chain: list[Shot], pairs: list[_Pair], lo: int, hi: int, cadences: list[float], margin: float,
                 cfg: BurstConfig) -> Group:
    m = cfg.model
    shots = chain[lo : hi + 1]
    seg = pairs[lo:hi]
    fps = " -> ".join(f"{1 / c:.1f}" for c in cadences)
    evidence = [f"{len(shots)} frames at {fps} fps" + (" (pace changed mid-burst)" if len(cadences) > 1 else "")]
    if not all(p.precise for p in seg):
        evidence.append("timestamps lack sub-seconds: grouped by whole-second timing")
    if all(s.meta.drive_mode == "continuous" for s in shots):
        evidence.append("drive mode: continuous")
    seqs = [s.meta.sequence_number for s in shots]
    if all(q is not None for q in seqs):
        evidence.append(f"burst sequence numbers {seqs[0]}..{seqs[-1]}")
    if any(_alike(x, y, m) for x, y in zip(shots, shots[1:])):
        evidence.append("consecutive previews look alike")
    if lo > 0 and pairs[lo - 1].interval is not None:
        why = "subject changed" if pairs[lo - 1].subject_change else "sequence restarted" if pairs[lo - 1].hard_split else "separate press"
        evidence.append(f"{why}: starts {pairs[lo - 1].interval:.2f} s after the previous frame")
    return Group("burst", shots, score=round(margin, 2), confidence=_confidence(margin), evidence=evidence)


def form_bursts(groups: list[Group], cfg: BurstConfig) -> list[Group]:
    """Replace chains of linked singles (in capture order, one camera) with bursts.

    Runs after bracket and focus detection: those groups break chains, so a
    burst is never formed across a bracket.
    """
    out: list[Group] = []
    chain: list[Shot] = []
    warnings: dict[int, list[str]] = {}  # id(shot) -> warnings carried by its single group

    def flush() -> None:
        if chain:
            for g in segment_chain(list(chain), cfg):
                if g.kind == "single":
                    g.warnings = warnings.get(id(g.shots[0]), [])
                out.append(g)
            chain.clear()

    for g in groups:
        if g.kind == "single":
            s = g.shots[0]
            if chain and not linked(chain[-1], s, cfg):
                flush()
            chain.append(s)
            if g.warnings:
                warnings[id(s)] = g.warnings
            continue
        flush()
        out.append(g)
    flush()
    return out


# --------------------------------------------------------------------------------------
# Timelapse / interval-timer sequences
# --------------------------------------------------------------------------------------

@dataclass(slots=True)
class TimelapseConfig:
    min_frames: int = 5
    min_interval: float = 1.0  # s; faster regular sequences are drive bursts
    max_interval: float = 3600.0
    tolerance: float = 0.004  # timers are precise: max deviation as a fraction of the interval...
    tolerance_abs: float = 0.02  # ...plus this many seconds of timestamp rounding (1/100 s stamps)


def _regular(a: Shot, b: Shot, interval: float, cfg: TimelapseConfig) -> bool:
    if a.time is None or b.time is None or a.camera_key != b.camera_key:
        return False
    if frame_step(a, b) not in (1, None):
        return False
    d = b.time - a.time
    if a.meta.time_precise and b.meta.time_precise:
        return abs(d - interval) <= interval * cfg.tolerance + cfg.tolerance_abs
    # Whole-second stamps: an exact interval I shows up as floor(I) or ceil(I).
    return d >= 2 and abs(d - interval) <= 1.0


def form_timelapses(groups: list[Group], cfg: TimelapseConfig) -> list[Group]:
    """Group runs of singles fired at a fixed interval (interval timer / timelapse).

    A timelapse is not a set of alternatives: every frame is part of the
    sequence, so culling usually leaves it alone. Groups must be one camera, in
    capture order.
    """
    out: list[Group] = []
    i = 0
    while i < len(groups):
        g = groups[i]
        if g.kind != "single" or i + 1 >= len(groups) or groups[i + 1].kind != "single":
            out.append(g)
            i += 1
            continue
        a, b = g.shots[0], groups[i + 1].shots[0]
        if a.time is None or b.time is None:
            out.append(g)
            i += 1
            continue
        interval = b.time - a.time
        coarse = not (a.meta.time_precise and b.meta.time_precise)
        if not ((2.0 if coarse else cfg.min_interval) <= interval <= cfg.max_interval):
            out.append(g)
            i += 1
            continue
        j = i + 1
        while j + 1 < len(groups) and groups[j + 1].kind == "single" and _regular(groups[j].shots[0], groups[j + 1].shots[0], interval, cfg):
            j += 1
        # A timelapse doesn't start right on the heels of a frame shot at drive speed
        # (that is a burst slowing down as its buffer fills).
        prev = groups[i - 1].shots[-1] if i > 0 else None
        heels = (prev is not None and prev.time is not None and frame_step(prev, a) in (1, None)
                 and a.time - prev.time < min(0.8 * interval, 1.1))
        # The camera recorded that its interval timer was off (or these are focus-shift frames).
        timer_off = any(groups[k].shots[0].meta.interval_shooting is False or groups[k].shots[0].meta.bracket_kind == "focus"
                        for k in range(i, j + 1))
        if heels or timer_off or not _regular(a, b, interval, cfg) or j - i + 1 < cfg.min_frames:
            out.append(g)
            i += 1
            continue
        shots = [groups[k].shots[0] for k in range(i, j + 1)]
        span = shots[-1].time - shots[0].time  # type: ignore[operator]
        every = span / (len(shots) - 1)
        out.append(Group("timelapse", shots, confidence=0.95 if all(s.meta.time_precise for s in shots) else 0.8,
                         evidence=[f"{len(shots)} frames every {every:.2f} s (interval timer)"]))
        i = j + 1
    return out
