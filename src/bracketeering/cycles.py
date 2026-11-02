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
    recompose_gap: float = 3.0  # s between frames (end of one exposure to start of the next)...
    rhythm_factor: float = 3.0  # ...and this many times the cycle's typical gap, whichever is longer
    drive_gap: float = 0.7  # s: below this, frames were fired by the drive; never split on looks
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
    session_of: list[int] = []  # contiguous run of bracketing-on frames each cycle belongs to
    session = 0
    rest: list[Shot] = []
    current: list[Shot] = []
    for s in shots:
        v = _slot(s)
        if v is None:
            if current:
                cycles.append(current)
                session_of.append(session)
                current = []
            session += 1  # bracketing was off for a frame: the program may have changed
            rest.append(s)
            continue
        if current:
            prev = current[-1]
            pv = _slot(prev)
            restart = v in {_slot(x) for x in current} or rank(v) <= rank(pv)  # type: ignore[arg-type]
            fs = frame_step(prev, s)
            if restart or (fs is not None and fs > _MAX_DELETED + 1):  # many frames missing between
                cycles.append(current)
                session_of.append(session)
                current = []
        current.append(s)
    if current:
        cycles.append(current)
        session_of.append(session)

    groups: list[Group] = []
    for k, cyc in enumerate(cycles):
        neighbours = [c for c, sid in zip(cycles, session_of) if sid == session_of[k]]
        for piece in _split_recomposed(cyc, cfg, similar):
            g = _group(piece, cyc, neighbours, cfg)
            if g.kind == "exposure" and _moving(piece, similar):
                g.warnings.append("frames don't register (subject or camera moving): an HDR merge may ghost")
            groups.append(g)
    return groups, rest


def _split_recomposed(cycle: list[Shot], cfg: CycleConfig, similar) -> list[list[Shot]]:
    """Split a camera cycle where the shooter recomposed.

    - A pause far longer than the cycle's own rhythm. Long exposures and single-frame drive
      (aurora on a tripod: 30 s exposures, presses 20-60 s apart) have long gaps everywhere,
      so a pause only counts against the other gaps in the same cycle, and never below
      `recompose_gap`.
    - Frames that clearly don't register, but only across a real pause. At drive speed a
      mismatch means the subject or camera is moving (wildlife), not a new composition:
      the cycle stays whole and gets flagged.
    """
    from .pixels import DIFFERENT

    gaps = [_gap(a, b) for a, b in zip(cycle, cycle[1:])]
    pieces: list[list[Shot]] = [[cycle[0]]]
    for i, (a, b) in enumerate(zip(cycle, cycle[1:])):
        g = gaps[i]
        others = [x for j, x in enumerate(gaps) if j != i and x is not None]
        rhythm = sorted(others)[len(others) // 2] if others else 0.0
        long_pause = g is not None and g > max(cfg.recompose_gap, cfg.rhythm_factor * rhythm)
        sim = similar(a, b) if similar else None
        moved = sim is not None and sim < DIFFERENT and g is not None and g >= cfg.drive_gap
        if long_pause or moved:
            pieces.append([b])
        else:
            pieces[-1].append(b)
    return pieces


def _moving(piece: list[Shot], similar) -> bool:
    if not similar:
        return False
    from .pixels import DIFFERENT

    return any((s := similar(a, b)) is not None and s < DIFFERENT for a, b in zip(piece, piece[1:]))


def _program_size(cycle: list[Shot], session: list[list[Shot]]) -> int | None:
    """Frames in this camera cycle's bracket program, or None when it can't be known reliably.

    The camera's own setting wins (Nikon BracketProgram). Otherwise it is inferred from the whole
    cycle (not just a recompose-split fragment of it): programs are evenly spaced around the metered
    slot, so a cycle reaching k steps out has 2k+1 frames (k+1 if one-sided). The step is the
    cycle's own slot spacing; a cycle with a single spacing (0, -4/3) borrows it from the nearest
    cycle of the same bracketing session, and the session says symmetric vs one-sided.
    """
    programs = [s.meta.bracket_program for s in cycle if s.meta.bracket_program]
    if programs:
        return max(programs)
    slots = sorted({v for s in cycle if (v := _slot(s)) is not None})
    if len(slots) < 2:
        return None
    spacings = [b - a for a, b in zip(slots, slots[1:])]
    step = min(spacings)
    if len(slots) < 3:
        # two slots: 0 / -4/3 could be a 3F at 4/3 or two slots of a 5F at 2/3; ask the nearest
        # cycle of the same session that shows a spacing of its own
        idx = next((i for i, c in enumerate(session) if c is cycle), 0)
        for _, c in sorted(enumerate(session), key=lambda ic: abs(ic[0] - idx)):
            cs = sorted({v for s in c if (v := _slot(s)) is not None})
            if len(cs) >= 3:
                step = min(step, min(b - a for a, b in zip(cs, cs[1:])))
                break
    session_slots = [v for c in session for s in c if (v := _slot(s)) is not None]
    reach = round(max(abs(v) for v in slots) / step)
    if all(v >= 0 for v in session_slots) or all(v <= 0 for v in session_slots):
        return reach + 1
    return 2 * reach + 1


def _group(piece: list[Shot], cycle: list[Shot], session: list[list[Shot]], cfg: CycleConfig) -> Group:
    slots = [_slot(s) for s in piece]
    program = _program_size(cycle, session)
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
    if program is not None and len(piece) < program:
        warnings.append(f"partial bracket: {len(piece)} of {program} slots")
    return Group("exposure", piece, score=10.0, confidence=0.99, evidence=evidence, warnings=warnings)
