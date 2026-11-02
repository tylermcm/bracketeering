"""Exposure brackets from the camera's own record of its bracket program.

Some bodies write, per frame, whether auto-bracketing was on and which slot of the program the frame
was (Nikon: ShootingMode "Exposure Bracketing" + ExposureBracketValue). When that is available it
beats inferring brackets from measured exposure: with Auto ISO the camera re-meters on every press,
so measured deltas drift (a 0 / -4/3 / -2/3 cycle can measure +0.85 / -0.64 / 0).

Rules (validated against a hand-checked real shoot, docs/reviews/2026-11-02-china-8-26.md):

* Slots run in the program's order, the metered slot first by default (0, -, +). A slot that
  repeats or comes earlier in the order than its predecessor starts a new cycle: the shooter
  re-pressed and the camera restarted, or earlier frames of the cycle were deleted.
* A cycle is split where the shooter recomposed partway: a pause longer than `recompose_gap`, or
  (with --pixels) previews that clearly differ. Re-presses of a second or two on one composition
  stay together; they're alignable for HDR.
* A lone frame left over is a single (flagged), never a bracket of one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .model import Group, Shot, frame_step


_MAX_DELETED = 4  # a cycle can survive this many deleted frames inside it


@dataclass(slots=True)
class CycleConfig:
    recompose_gap: float = 3.0  # s between frames (end of one exposure to start of the next)
    session_window: float = 600.0  # s: slots seen this close together form one program


def _slot(s: Shot) -> float | None:
    m = s.meta
    if m.bracket_kind == "exposure" and m.bracket_value is not None:
        return round(m.bracket_value * 6) / 6  # programs are in 1/3 and 1/2 stops
    return None


def _gap(a: Shot, b: Shot) -> float | None:
    if a.time is None or b.time is None:
        return None
    return b.time - a.time - (a.meta.exposure_time or 0.0)


def _fmt(v: float) -> str:
    thirds = round(v * 3)
    if abs(v * 3 - thirds) < 1e-6 and thirds % 3:
        sign = "-" if thirds < 0 else "+"
        return f"{sign}{abs(thirds)}/3"
    halves = round(v * 2)
    if abs(v * 2 - halves) < 1e-6 and halves % 2:
        return f"{'-' if halves < 0 else '+'}{abs(halves)}/2"
    return f"{v:+g}" if v else "0"


def has_slots(shots: list[Shot]) -> bool:
    return any(_slot(s) is not None for s in shots)


def _first_slot(shots: list[Shot], cfg: CycleConfig) -> float:
    """The slot a cycle starts with: most common slot among frames that clearly begin a cycle."""
    starts: Counter[float] = Counter()
    prev = None
    for s in shots:
        v = _slot(s)
        if v is not None and (prev is None or _slot(prev) is None or (g := _gap(prev, s)) is None or g > cfg.recompose_gap):
            starts[v] += 1
        prev = s
    if not starts:
        return 0.0
    best = max(starts.values())
    tied = [v for v, c in starts.items() if c == best]
    return 0.0 if 0.0 in tied else min(tied)


def bracket_cycles(shots: list[Shot], cfg: CycleConfig, similar=None) -> tuple[list[Group], list[Shot]]:
    """Group slot-bearing frames of one camera (capture order) into brackets.

    Returns (groups, remaining shots without slot information). `similar(a, b)` returns a
    preview similarity in -1..1, or None when unknown.
    """
    if not has_slots(shots):
        return [], shots
    first = _first_slot(shots, cfg)
    values = sorted({v for s in shots if (v := _slot(s)) is not None} - {first})

    def rank(v: float) -> int:
        return 0 if v == first else 1 + values.index(v)

    cycles: list[list[Shot]] = []
    rest: list[Shot] = []
    current: list[Shot] = []
    for s in shots:
        v = _slot(s)
        if v is None:
            if current:
                cycles.append(current)
                current = []
            rest.append(s)
            continue
        if current:
            prev = current[-1]
            pv = _slot(prev)
            restart = v in {_slot(x) for x in current} or rank(v) <= rank(pv)  # type: ignore[arg-type]
            fs = frame_step(prev, s)
            if restart or (fs is not None and fs > _MAX_DELETED + 1):  # many frames missing between
                cycles.append(current)
                current = []
        current.append(s)
    if current:
        cycles.append(current)

    groups: list[Group] = []
    for cyc in cycles:
        for piece in _split_recomposed(cyc, cfg, similar):
            groups.append(_group(piece, shots, cfg))
    return groups, rest


def _split_recomposed(cycle: list[Shot], cfg: CycleConfig, similar) -> list[list[Shot]]:
    from .pixels import DIFFERENT

    pieces: list[list[Shot]] = [[cycle[0]]]
    for a, b in zip(cycle, cycle[1:]):
        g = _gap(a, b)
        sim = similar(a, b) if similar else None
        if (g is not None and g > cfg.recompose_gap) or (sim is not None and sim < DIFFERENT):
            pieces.append([b])
        else:
            pieces[-1].append(b)
    return pieces


def _program_size(piece: list[Shot], all_shots: list[Shot], cfg: CycleConfig) -> int:
    t0 = piece[0].time
    vals = {
        v for s in all_shots
        if (v := _slot(s)) is not None and (t0 is None or s.time is None or abs(s.time - t0) <= cfg.session_window)
    }
    return max(len(vals), len(piece))


def _group(piece: list[Shot], all_shots: list[Shot], cfg: CycleConfig) -> Group:
    slots = [_slot(s) for s in piece]
    program = _program_size(piece, all_shots, cfg)
    if len(piece) == 1:
        return Group("single", piece, confidence=1.0, warnings=[
            f"auto-bracket frame (slot {_fmt(slots[0])}) without partners: rest of its cycle recomposed or deleted"])  # type: ignore[arg-type]
    evidence = [
        "camera auto-bracket (maker note): slots " + ", ".join(_fmt(v) for v in slots),  # type: ignore[arg-type]
    ]
    gaps = [g for a, b in zip(piece, piece[1:]) if (g := _gap(a, b)) is not None]
    if gaps and max(gaps) > 0.6:
        evidence.append(f"shot across re-presses (longest pause {max(gaps):.1f} s), one composition")
    warnings = []
    if len(piece) < program:
        warnings.append(f"partial bracket: {len(piece)} of {program} slots")
    return Group("exposure", piece, score=10.0, confidence=0.99, evidence=evidence, warnings=warnings)
