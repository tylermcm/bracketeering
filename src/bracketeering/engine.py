"""Bracket detection and grouping engine.

Pipeline (per camera body):

1. Order shots by capture time (sub-second when available), then frame counter.
2. Split into *runs*: maximal chains of consecutive shots that could belong to
   the same bracket (same body/lens/focal length, comparable exposure data, and
   a small gap between the end of one exposure and the start of the next).
3. Carve out focus stacks: constant-exposure blocks flagged by maker notes or
   with monotonic focus distance.
4. Segment each remaining run with dynamic programming. Every candidate group
   gets a log-odds style score from independent evidence (distinct and evenly
   spaced exposures, known camera firing orders, EXIF ExposureMode=auto-bracket,
   maker-note bracket mode and shot position, exposure compensation agreeing
   with measured deltas, time gaps at the boundaries). Singles score 0, so a
   group is formed only when the evidence outweighs the cost of grouping. The
   DP finds the globally best partition in O(n * max_size) group evaluations.
5. Run the DP a second time with the camera's dominant bracket signature
   (size, firing order, step) as a prior. Photographers rarely change bracket
   settings mid-shoot, so this resolves the ambiguous cases, such as two
   back-to-back brackets with nothing between them.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import median

from .bursts import BurstConfig, TimelapseConfig, form_bursts, form_timelapses
from .cycles import CycleConfig, bracket_cycles
from .model import Group, Shot, frame_step
from .pixels import similarity

NEG_INF = float("-inf")


@dataclass(slots=True)
class EngineConfig:
    max_gap: float = 2.0  # s from the end of one exposure to the start of the next
    manual_max_gap: float = 6.0  # s, for user-driven brackets in Manual exposure mode
    imprecise_slack: float = 1.0  # extra s when timestamps lack sub-seconds
    max_size: int = 9
    min_size: int = 2
    dup_tol: float = 0.16  # EV: frames closer than this count as the same exposure
    min_step: float = 0.28  # EV (1/3-stop brackets, allowing for rounding)
    min_step_unsupported: float = 0.45  # EV: smaller steps need bracket metadata
    max_step: float = 4.2  # EV
    max_irregularity: float = 0.45  # max deviation of a step from the median step, as a fraction
    grid_tol: float = 0.08  # EV: how far a step may sit from the 1/3- or 1/2-stop grid
    abs_tol: float = 0.17  # EV: max deviation of a delta from a whole number of steps
    max_missing: int = 2  # exposure levels that may be absent from a bracket (deleted frames)
    focal_tol: float = 0.02
    max_frame_skip: int = 3  # max frame-counter jump inside a run
    detect_focus: bool = True
    min_focus_frames: int = 3  # with maker-note evidence
    min_focus_frames_inferred: int = 5  # inferred from focus distance alone
    focus_max_gap: float = 1.5
    use_prior: bool = True
    detect_bursts: bool = True
    bursts: BurstConfig = field(default_factory=BurstConfig)
    detect_timelapse: bool = True
    cycles: CycleConfig = field(default_factory=CycleConfig)
    timelapse: TimelapseConfig = field(default_factory=TimelapseConfig)
    min_confidence: float = 0.5  # exposure brackets below this are reported as singles


@dataclass(slots=True, frozen=True)
class Weights:
    frame: float = 1.0  # per extra frame in a valid group
    group_cost: float = 2.0
    size_prior: tuple[tuple[int, float], ...] = (
        (2, -0.6), (3, 2.0), (4, 0.3), (5, 2.0), (6, 0.0), (7, 1.8), (8, -0.2), (9, 1.6),
    )
    irregular: float = 2.0  # x irregularity x (n-1)
    missing: float = 2.0  # per absent exposure level
    pattern: float = 0.5  # per extra frame when the firing order is a known one
    aeb: float = 1.0  # per frame with EXIF ExposureMode = auto bracket
    not_aeb: float = 1.5  # per frame explicitly *not* auto-bracket (camera known to report it)
    manual_bracket: float = 0.4  # per frame when all frames are in Manual exposure mode
    maker: float = 1.0  # per frame with maker-note exposure bracketing
    orphan: float = 1.5  # leaving an auto-bracket-flagged frame ungrouped
    maker_contra: float = 1.5  # per frame with maker-note saying "not an exposure bracket"
    sequence: float = 3.0  # maker-note shot positions run exactly 1..n
    bias: float = 0.5  # per extra frame when exposure compensation tracks the deltas
    multi_leg: float = 0.75  # per extra frame when 2+ of shutter/aperture/ISO vary
    small_step: float = 3.0  # step < 0.6 EV without metadata support
    off_grid: float = 3.0  # step off the 1/3|1/2-stop grid or uneven deltas, without metadata
    flash: float = 0.75  # per frame that fired a flash
    wb_change: float = 1.0
    prior: float = 0.6  # per extra frame matching the camera's dominant signature
    gap: float = 0.4  # x log2(gap ratio) at a boundary
    frame_skip: float = 0.5  # boundary bonus where the frame counter jumps


W = Weights()
_STEP_GRID = sorted({round(k / 3, 3) for k in range(1, 13)} | {round(k / 2, 3) for k in range(1, 9)})
_SIZE_PRIOR = dict(W.size_prior)


@dataclass(slots=True, frozen=True)
class Signature:
    size: int
    ranks: tuple[int, ...]
    step: float

    def matches(self, size: int, ranks: tuple[int, ...], step: float) -> bool:
        return size == self.size and ranks == self.ranks and abs(step - self.step) < 0.2


@dataclass(slots=True)
class _Eval:
    score: float
    fatal: str | None = None  # "dup"/"noev" are monotone: no larger group can recover
    ranks: tuple[int, ...] = ()
    step: float = 0.0
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta_support: bool = False


@dataclass(slots=True)
class _CameraContext:
    reports_aeb: bool  # this body writes ExposureMode=2 during AEB (seen in the data)


# --------------------------------------------------------------------------- utils

def _order_key(s: Shot):
    return (s.time if s.time is not None else math.inf, s.meta.file_number or 0, s.name)


def _gap(a: Shot, b: Shot) -> float | None:
    if a.time is None or b.time is None:
        return None
    return b.time - a.time - (a.meta.exposure_time or 0.0)


_frame_step = frame_step


def _is_manual(s: Shot) -> bool:
    return s.meta.exposure_mode == 1 or (s.meta.exposure_mode is None and s.meta.exposure_program == 1)


def _rel_close(a: float | None, b: float | None, tol: float) -> bool:
    if a is None or b is None:
        return True
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def _linked(a: Shot, b: Shot, cfg: EngineConfig) -> bool:
    ma, mb = a.meta, b.meta
    if ma.lens and mb.lens and ma.lens != mb.lens:
        return False
    if not _rel_close(ma.focal_length, mb.focal_length, cfg.focal_tol):
        return False
    if a.exposure_mask != b.exposure_mask:
        return False
    step = _frame_step(a, b)
    if step is not None and not (0 <= step <= cfg.max_frame_skip):
        return False
    g = _gap(a, b)
    if g is None:
        return step == 1
    limit = cfg.manual_max_gap if (_is_manual(a) and _is_manual(b)) else cfg.max_gap
    if not (ma.time_precise and mb.time_precise):
        limit += cfg.imprecise_slack
    return g <= limit


def split_runs(shots: list[Shot], cfg: EngineConfig) -> list[list[Shot]]:
    runs: list[list[Shot]] = []
    for s in shots:
        if runs and _linked(runs[-1][-1], s, cfg):
            runs[-1].append(s)
        else:
            runs.append([s])
    return runs


@lru_cache(maxsize=None)
def known_orders(n: int) -> dict[tuple[int, ...], str]:
    """Firing orders used by camera AEB implementations, as rank tuples.

    ranks[i] is the brightness rank (0 = darkest) of the i-th frame shot.
    """
    out: dict[tuple[int, ...], str] = {}
    asc = tuple(range(n))
    out[asc] = "-,0,+ (ascending)"
    out[asc[::-1]] = "+,0,- (descending)"
    if n >= 3 and n % 2 == 1:
        c = n // 2
        minus_first = [c]
        plus_first = [c]
        for k in range(1, c + 1):
            minus_first += [c - k, c + k]
            plus_first += [c + k, c - k]
        out.setdefault(tuple(minus_first), "0,-,+ (centre first)")
        out.setdefault(tuple(plus_first), "0,+,- (centre first)")
        out.setdefault(tuple([c, *range(c - 1, -1, -1), *range(c + 1, n)]), "0,-,--,+,++")
        out.setdefault(tuple([c, *range(0, c), *range(c + 1, n)]), "0,--,-,+,++")
    return out


def _fit_step(diffs: list[float], cfg: EngineConfig) -> tuple[float, list[int]]:
    """Estimate the bracket step and how many steps each sorted delta spans.

    Candidate steps are the camera grid (1/3 and 1/2 stops) plus the median
    delta. The fit prefers the fewest implied missing frames, then the smallest
    residual. The step is then refined as span / total levels, which averages
    out per-frame rounding of nominal shutter speeds.
    """
    best: tuple[float, float, list[int]] | None = None
    d_min = min(diffs)
    for c in (*_STEP_GRID, median(diffs)):
        # The smallest delta spans at least one step and, realistically, at most three.
        if c < cfg.min_step * 0.9 or c > d_min * 1.45 or c < d_min / 3.5:
            continue
        levels = [max(1, round(d / c)) for d in diffs]
        missing = sum(k - 1 for k in levels)
        dev = max(abs(d - k * c) for d, k in zip(diffs, levels))
        if dev / c > cfg.max_irregularity:
            continue
        cost = dev + 0.15 * missing  # rounding error is absolute (~0.1 EV), not relative
        if best is None or cost < best[0]:
            best = (cost, c, levels)
    if best is None:
        return median(diffs), [1] * len(diffs)
    levels = best[2]
    return sum(diffs) / sum(levels), levels


def _match_order(ranks: tuple[int, ...], levels: list[int], full: int) -> tuple[str, int] | None:
    """Name the known firing order these ranks follow, allowing for missing levels.

    Returns (name, levels_missing_at_the_ends). Interior gaps are visible in
    the exposure spacing; a missing darkest or brightest frame is not, so those
    are tried explicitly and only accepted when they complete a known order.
    """
    if full == len(ranks) and (name := known_orders(full).get(ranks)) is not None:
        return name, 0
    level_of_rank = [0]
    for k in levels:
        level_of_rank.append(level_of_rank[-1] + k)
    for low, high in ((0, 0), (1, 0), (0, 1)):
        size = full + low + high
        if size > 9 or (size == len(ranks)):
            continue
        observed = tuple(level_of_rank[r] + low for r in ranks)
        present = set(observed)
        for order, name in known_orders(size).items():
            if tuple(r for r in order if r in present) == observed:
                return name, low + high
    return None


def _single_score(s: Shot) -> float:
    """Cost of leaving a shot ungrouped: negative when metadata says it belongs to a bracket."""
    m = s.meta
    sc = 0.0
    if m.exposure_mode == 2:
        sc -= W.orphan
    if m.bracket_kind == "exposure":
        sc -= W.orphan
    return sc


def _log_ratio(g: float, ref: float) -> float:
    return math.log2((max(g, 0.0) + 0.05) / (max(ref, 0.0) + 0.05))


# --------------------------------------------------------------------- group scoring

def _score_exposure(shots: list[Shot], ctx: _CameraContext, cfg: EngineConfig) -> _Eval:
    """Prior-independent evidence for `shots` being one exposure bracket."""
    n = len(shots)
    evs = [s.ev for s in shots]
    if any(e is None for e in evs):
        return _Eval(NEG_INF, "noev")
    order = sorted(range(n), key=lambda i: evs[i])  # type: ignore[arg-type,return-value]
    sorted_evs = [evs[i] for i in order]
    diffs = [b - a for a, b in zip(sorted_evs, sorted_evs[1:])]  # type: ignore[operator]
    if min(diffs) < cfg.dup_tol:
        return _Eval(NEG_INF, "dup")
    if n < cfg.min_size or n > cfg.max_size:
        return _Eval(NEG_INF, "size")
    step, levels = _fit_step(diffs, cfg)
    if not (cfg.min_step <= step <= cfg.max_step):
        return _Eval(NEG_INF, "step")
    # A diff spanning k steps means k-1 exposure levels are missing (deleted frames).
    missing = sum(k - 1 for k in levels)
    if missing > cfg.max_missing or (missing and n < 3):
        return _Eval(NEG_INF, "irregular")
    irregular = max(abs(d - k * step) for d, k in zip(diffs, levels)) / step
    if irregular > cfg.max_irregularity:
        return _Eval(NEG_INF, "irregular")

    ranks = [0] * n
    for r, i in enumerate(order):
        ranks[i] = r
    rk = tuple(ranks)
    metas = [s.meta for s in shots]
    ev: list[str] = []
    warn: list[str] = []
    full = n + missing  # the bracket's size before frames were deleted
    score = W.frame * (n - 1) - W.group_cost + _SIZE_PRIOR.get(full, -0.5)
    score -= W.irregular * irregular * (n - 1)
    ev.append(f"{n} distinct exposures, {step:.2f} EV step")
    if missing:
        score -= W.missing * missing
        warn.append(f"{missing} exposure level(s) missing: frame(s) likely deleted")

    if (match := _match_order(rk, levels, full)) is not None:
        name, edge_missing = match
        if edge_missing:
            score += _SIZE_PRIOR.get(full + edge_missing, -0.5) - _SIZE_PRIOR.get(full, -0.5) - W.missing * edge_missing
            warn.append(f"{edge_missing} end frame(s) of a {full + edge_missing}-frame bracket missing")
        score += W.pattern * (n - 1)
        ev.append(f"firing order {name}")

    # --- metadata evidence
    meta_support = False
    modes = [m.exposure_mode for m in metas]
    n_aeb = sum(m == 2 for m in modes)
    if n_aeb == n:
        score += W.aeb * n
        meta_support = True
        ev.append("EXIF ExposureMode = auto bracket")
    elif all(_is_manual(s) for s in shots):
        if n >= 3:
            score -= W.manual_bracket * n
            ev.append("Manual exposure mode, no auto-bracket flag: likely adjusted by hand")
        else:
            return _Eval(NEG_INF, "manual2")
    elif ctx.reports_aeb:
        n_contra = sum(m is not None and m != 2 for m in modes)
        score -= W.not_aeb * n_contra
        if n_aeb:
            warn.append(f"only {n_aeb}/{n} frames flagged auto bracket")

    kinds = [m.bracket_kind for m in metas]
    if any(k == "none" for k in kinds):
        # The camera recorded that auto-bracketing was off for (some of) these frames.
        return _Eval(NEG_INF, "maker-off")
    if all(k == "exposure" for k in kinds):
        score += W.maker * n
        meta_support = True
        ev.append("maker note: exposure bracketing")
    else:
        contra = sum(k is not None and k != "exposure" for k in kinds)
        score -= W.maker_contra * contra

    positions = [m.bracket_position for m in metas]
    if all(p is not None for p in positions):
        if positions == list(range(1, n + 1)):
            score += W.sequence
            ev.append(f"maker-note shot positions 1..{n}")
        elif 1 in positions[1:]:
            return _Eval(NEG_INF, "seqreset")
        elif all(b == a + 1 for a, b in zip(positions, positions[1:])):
            warn.append(f"shot positions start at {positions[0]}: leading frames missing")
        else:
            score -= 1.0

    biases = [m.exposure_bias for m in metas]
    if all(b is not None for b in biases) and len({round(b, 2) for b in biases}) == n:  # type: ignore[arg-type]
        err = max(abs((biases[i] - biases[0]) - (evs[i] - evs[0])) for i in range(1, n))  # type: ignore[operator]
        if err < 0.25:
            score += W.bias * (n - 1)
            meta_support = True
            ev.append("exposure compensation tracks the deltas")

    legs = 0
    for attr in ("exposure_time", "f_number", "iso"):
        vals = [getattr(m, attr) for m in metas if getattr(m, attr) is not None]
        if vals and max(vals) > min(vals) * 1.02:
            legs += 1
    if legs >= 2 and n_aeb < n:
        score -= W.multi_leg * (n - 1)

    if not meta_support:
        # Without metadata vouching for the group, demand what camera AEB
        # produces: a step on the 1/3- or 1/2-stop grid, each delta an exact
        # multiple of it (up to nominal shutter/aperture rounding).
        if step < cfg.min_step_unsupported:
            # A 1/3-stop series with no bracket metadata is indistinguishable from
            # auto-exposure metering drift in a burst: never call it a bracket.
            return _Eval(NEG_INF, "smallstep")
        if step < 0.6:
            score -= W.small_step
        grid_dev = min(abs(step - g) for g in _STEP_GRID)
        abs_dev = max(abs(d - k * step) for d, k in zip(diffs, levels))
        if grid_dev > cfg.grid_tol or abs_dev > cfg.abs_tol:
            score -= W.off_grid
    fired = sum(bool(m.flash_fired) for m in metas)
    if fired:
        score -= W.flash * fired
    wbs = {m.white_balance for m in metas if m.white_balance is not None}
    if len(wbs) > 1:
        score -= W.wb_change

    return _Eval(score, None, rk, step, ev, warn, meta_support)


# ------------------------------------------------------------------------ focus stacks

def _same_exposure(a: Shot, b: Shot, cfg: EngineConfig) -> bool:
    if a.exposure_mask != b.exposure_mask:
        return False
    ea, eb = a.ev, b.ev
    if ea is None or eb is None:
        return False
    return abs(ea - eb) < cfg.dup_tol


def _focus_evidence(block: list[Shot], cfg: EngineConfig) -> tuple[float, list[str]] | None:
    n = len(block)
    flagged = sum(s.meta.bracket_kind == "focus" for s in block)
    if n >= cfg.min_focus_frames and flagged >= 0.8 * n:
        return 0.95, ["maker note: focus bracketing", f"{n} frames at constant exposure"]
    dists = [s.meta.focus_distance for s in block]
    if n >= cfg.min_focus_frames_inferred and all(d is not None for d in dists):
        deltas = [b - a for a, b in zip(dists, dists[1:])]  # type: ignore[operator]
        monotone = all(d >= 0 for d in deltas) or all(d <= 0 for d in deltas)
        distinct = len(set(dists))
        if monotone and distinct >= 0.6 * n and dists[0] != dists[-1]:
            return 0.75, [f"{n} frames at constant exposure", "focus distance steps monotonically"]
    return None


def _carve_focus(run: list[Shot], cfg: EngineConfig) -> list[tuple[str, list[Shot], float, list[str]]]:
    """Split a run into ("focus", block, conf, evidence) and ("pending", block, 0, [])."""
    out: list[tuple[str, list[Shot], float, list[str]]] = []
    pending: list[Shot] = []
    i, n = 0, len(run)
    while i < n:
        j = i + 1
        while j < n and _same_exposure(run[j - 1], run[j], cfg) and ((g := _gap(run[j - 1], run[j])) is None or g <= cfg.focus_max_gap):
            j += 1
        block = run[i:j]
        found = _focus_evidence(block, cfg) if cfg.detect_focus and len(block) >= cfg.min_focus_frames else None
        if found:
            if pending:
                out.append(("pending", pending, 0.0, []))
                pending = []
            out.append(("focus", block, found[0], found[1]))
        else:
            pending.extend(block)
        i = j
    if pending:
        out.append(("pending", pending, 0.0, []))
    return out


# --------------------------------------------------------------------------- segmentation

def _with_prior(e: _Eval, n: int, prior: Signature | None) -> _Eval:
    if prior is None or e.fatal or not prior.matches(n, e.ranks, e.step):
        return e
    return _Eval(e.score + W.prior * (n - 1), None, e.ranks, e.step,
                 [*e.evidence, "matches the camera's usual bracket"], e.warnings, e.meta_support)


_Cache = dict[tuple[int, int], _Eval]


def _segment_exposure(run: list[Shot], ctx: _CameraContext, cfg: EngineConfig, prior: Signature | None,
                      cache: _Cache) -> list[tuple[list[Shot], _Eval | None]]:
    n = len(run)
    if n == 1:
        return [(run, None)]
    gaps = [None] + [_gap(run[i - 1], run[i]) for i in range(1, n)]
    known = [g for g in gaps if g is not None]
    ref = median(known) if known else None
    boundary = [0.0] * n
    for i in range(1, n):
        b = 0.0
        if ref is not None and gaps[i] is not None:
            b += W.gap * max(-1.5, min(2.0, _log_ratio(gaps[i], ref)))  # type: ignore[arg-type]
        fs = _frame_step(run[i - 1], run[i])
        if fs is not None and fs > 1:
            b += W.frame_skip
        boundary[i] = b

    best = [NEG_INF] * (n + 1)
    best[0] = 0.0
    back: list[tuple[int, _Eval | None]] = [(0, None)] * (n + 1)
    for j in range(1, n + 1):
        for k in range(1, min(cfg.max_size, j) + 1):
            i = j - k
            if k == 1:
                e: _Eval | None = None
                sc = _single_score(run[i])
            else:
                key = (id(run[i]), k)
                if (base := cache.get(key)) is None:
                    base = cache[key] = _score_exposure(run[i:j], ctx, cfg)
                e = _with_prior(base, k, prior)
                if e.fatal in ("dup", "noev", "seqreset"):
                    break  # monotone: extending further back cannot help
                sc = e.score
                if sc == NEG_INF:
                    continue
            total = best[i] + sc + (boundary[i] if i > 0 else 0.0)
            if total > best[j]:
                best[j] = total
                back[j] = (i, e)
    out: list[tuple[list[Shot], _Eval | None]] = []
    j = n
    while j > 0:
        i, e = back[j]
        if e is not None and e.score <= 0:  # only kept for boundary reasons: too weak to report
            out.extend(([s], None) for s in reversed(run[i:j]))
        else:
            out.append((run[i:j], e))
        j = i
    out.reverse()
    return _burst_veto(out)


def _in_burst(group: list[Shot], e: _Eval, neighbour: Shot, before: bool) -> bool:
    """Is `neighbour` an ungrouped frame shot at the group's cadence and exposure level?"""
    gaps = [g for a, b in zip(group, group[1:]) if (g := _gap(a, b)) is not None]
    g = _gap(neighbour, group[0]) if before else _gap(group[-1], neighbour)
    if g is not None and gaps:
        coarse = not all(s.meta.time_precise for s in (*group, neighbour))
        # whole-second stamps: a neighbour one tick later can still be at the same cadence
        if g > 1.25 * max(gaps) + (0.5 if coarse else 0.05):
            return False
    else:
        adjacent = _frame_step(neighbour, group[0]) if before else _frame_step(group[-1], neighbour)
        if adjacent != 1:
            return False
    evs = [s.ev for s in group]
    nev = neighbour.ev
    if nev is None or any(x is None for x in evs):
        return False
    return min(evs) - 0.5 * e.step <= nev <= max(evs) + 0.5 * e.step  # type: ignore[type-var,operator]


def _burst_veto(out: list[tuple[list[Shot], _Eval | None]]) -> list[tuple[list[Shot], _Eval | None]]:
    """Dissolve unsupported groups embedded in a stream of similar ungrouped frames.

    Auto-exposure bursts wander in 1/3-stop increments and can mimic a small
    bracket. A real bracket isn't flanked by ungrouped frames at the same
    cadence and exposure level. Groups backed by metadata are exempt.
    """
    def burst_neighbour(q: int, shots: list[Shot], e: _Eval, before: bool) -> bool:
        if not 0 <= q < len(out):
            return False
        nshots, ne = out[q]
        if ne is not None:
            # Back-to-back brackets repeat one signature; a burst produces mismatched ones.
            if ne.meta_support or (ne.ranks == e.ranks and abs(ne.step - e.step) < 0.2):
                return False
        return _in_burst(shots, e, nshots[-1] if before else nshots[0], before)

    veto = [
        e is not None and not e.meta_support and (burst_neighbour(p - 1, shots, e, True) or burst_neighbour(p + 1, shots, e, False))
        for p, (shots, e) in enumerate(out)
    ]
    result: list[tuple[list[Shot], _Eval | None]] = []
    for (shots, e), v in zip(out, veto):
        if v:
            result.extend(([s], None) for s in shots)
        else:
            result.append((shots, e))
    return result


def _confidence(score: float) -> float:
    return round(1.0 / (1.0 + math.exp(-score / 2.0)), 3)


def _segment_run(run: list[Shot], ctx: _CameraContext, cfg: EngineConfig, prior: Signature | None, cache: _Cache) -> list[Group]:
    groups: list[Group] = []
    for kind, block, conf, evidence in _carve_focus(run, cfg):
        if kind == "focus":
            groups.append(Group("focus", block, score=0.0, confidence=conf, evidence=evidence))
            continue
        for shots, e in _segment_exposure(block, ctx, cfg, prior, cache):
            if e is None:
                groups.append(Group("single", shots, confidence=1.0))
            else:
                groups.append(Group("exposure", shots, score=round(e.score, 2), confidence=_confidence(e.score),
                                    evidence=list(e.evidence), warnings=list(e.warnings)))
    return groups


def _dominant_signature(groups: list[Group]) -> Signature | None:
    sigs: Counter[Signature] = Counter()
    total = 0
    for g in groups:
        if g.kind != "exposure" or g.size < 3:
            continue
        evs = [s.ev for s in g.shots]
        order = sorted(range(g.size), key=lambda i: evs[i])  # type: ignore[arg-type,return-value]
        ranks = [0] * g.size
        for r, i in enumerate(order):
            ranks[i] = r
        sigs[Signature(g.size, tuple(ranks), round((g.step_ev or 0) * 3) / 3)] += 1
        total += 1
    if not sigs:
        return None
    sig, count = sigs.most_common(1)[0]
    return sig if count >= 2 and count >= 0.4 * total else None


def _pick_representative(g: Group) -> int:
    """Index of the frame that stands for the group (what a culler should look at)."""
    if g.kind == "exposure" and (slots := g.slots) is not None:
        return min(range(g.size), key=lambda i: (abs(slots[i]), i))  # the metered frame
    if g.kind == "exposure":
        evs = [s.ev for s in g.shots]
        if all(e is not None for e in evs):
            ordered = sorted(evs)  # type: ignore[type-var]
            mid = ordered[(len(ordered) - 1) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
            return min(range(g.size), key=lambda i: (abs(evs[i] - mid), abs(g.shots[i].meta.exposure_bias or 0.0), i))  # type: ignore[operator]
    if g.kind in ("focus", "burst", "timelapse"):
        return g.size // 2
    return 0


def _demote_low_confidence(groups: list[Group], cfg: EngineConfig) -> list[Group]:
    out: list[Group] = []
    for g in groups:
        if g.kind == "exposure" and g.confidence < cfg.min_confidence:
            for s in g.shots:
                out.append(Group("single", [s], confidence=1.0, warnings=[f"weak bracket candidate ({g.confidence:.2f}) below threshold"]))
        else:
            out.append(g)
    return out


def _annotate(groups: list[Group], prior: Signature | None) -> None:
    for g in groups:
        if g.kind == "exposure" and prior is not None and g.size < prior.size and all(s.meta.exposure_mode == 2 for s in g.shots):
            g.warnings.append(f"possibly incomplete: this camera's brackets are usually {prior.size} frames")
        if g.kind == "single" and g.shots[0].meta.exposure_mode == 2:
            g.warnings.append("flagged auto bracket but no partner frames found")


def detect(shots: list[Shot], cfg: EngineConfig | None = None) -> list[Group]:
    """Partition shots into exposure brackets, focus stacks, bursts, timelapses and singles."""
    cfg = cfg or EngineConfig()
    by_cam: dict[tuple[str, str, str], list[Shot]] = defaultdict(list)
    for s in shots:
        by_cam[s.camera_key].append(s)
    result: list[Group] = []
    for cam_shots in by_cam.values():
        cam_shots.sort(key=_order_key)
        ctx = _CameraContext(reports_aeb=any(s.meta.exposure_mode == 2 for s in cam_shots))
        maker_groups, cam_shots = bracket_cycles(cam_shots, cfg.cycles, similarity)
        runs = split_runs(cam_shots, cfg)
        cache: _Cache = {}  # group evaluations keyed by (first shot, size); shots outlive the call
        groups = [g for run in runs for g in _segment_run(run, ctx, cfg, None, cache)]
        prior = _dominant_signature(groups) if cfg.use_prior else None
        if prior is not None:
            groups = [g for run in runs for g in _segment_run(run, ctx, cfg, prior, cache)]
        groups = sorted(maker_groups + groups, key=lambda g: _order_key(g.shots[0]))
        groups = _demote_low_confidence(groups, cfg)
        _annotate(groups, prior)
        if cfg.detect_timelapse:  # before bursts: regular at >= 1 s is a timer, not a drive
            groups = form_timelapses(groups, cfg.timelapse)
        if cfg.detect_bursts:
            groups = form_bursts(groups, cfg.bursts)
        result.extend(groups)
    result.sort(key=lambda g: _order_key(g.shots[0]))
    for g in result:
        g.representative = _pick_representative(g)
    return result
