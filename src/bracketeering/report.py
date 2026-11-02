"""Serialise detection results to JSON / CSV / human-readable text."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from . import __version__
from .model import FileMeta, Group, Shot


def _fmt_time(t: float | None) -> str | None:
    if t is None:
        return None
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "")


def _fmt_shutter(t: float | None) -> str | None:
    if t is None:
        return None
    return f"1/{round(1 / t)}" if t < 0.3 else f"{t:g}s"


def _shot_dict(s: Shot) -> dict:
    m = s.meta
    return {
        "files": [str(p) for p in s.paths],
        "time": _fmt_time(m.capture_time),
        "shutter": _fmt_shutter(m.exposure_time),
        "aperture": m.f_number,
        "iso": m.iso,
        "exposure_bias": m.exposure_bias,
        "ev": round(s.ev, 3) if s.ev is not None else None,
    }


def group_id(index: int, g: Group) -> str:
    prefix = {"exposure": "B", "focus": "F"}.get(g.kind, "S")
    return f"{prefix}{index:05d}"


def to_dict(groups: list[Group], errors: list[FileMeta] | None = None) -> dict:
    brackets, stacks, singles = [], [], []
    for i, g in enumerate(groups, 1):
        if g.kind == "single":
            entry = {"id": group_id(i, g), **_shot_dict(g.shots[0])}
            if g.warnings:
                entry["warnings"] = g.warnings
            singles.append(entry)
            continue
        m = g.shots[0].meta
        entry = {
            "id": group_id(i, g),
            "kind": g.kind,
            "size": g.size,
            "confidence": g.confidence,
            "camera": " ".join(x for x in (m.make, m.model) if x) or None,
            "start": _fmt_time(g.shots[0].time),
            "ev_offsets": g.ev_offsets if g.kind == "exposure" else None,
            "step_ev": g.step_ev if g.kind == "exposure" else None,
            "evidence": g.evidence,
            "warnings": g.warnings,
            "frames": [_shot_dict(s) for s in g.shots],
        }
        (brackets if g.kind == "exposure" else stacks).append(entry)
    n_shots = sum(g.size for g in groups)
    return {
        "tool": "bracketeering",
        "version": __version__,
        "summary": {
            "shots": n_shots,
            "files": sum(len(s.files) for g in groups for s in g.shots),
            "exposure_brackets": len(brackets),
            "focus_stacks": len(stacks),
            "singles": len(singles),
            "unreadable": len(errors or []),
        },
        "brackets": brackets,
        "focus_stacks": stacks,
        "singles": singles,
        "errors": [{"file": str(e.path), "error": e.error} for e in (errors or [])],
    }


def to_json(groups: list[Group], errors: list[FileMeta] | None = None) -> str:
    return json.dumps(to_dict(groups, errors), indent=2)


def to_csv(groups: list[Group]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["group_id", "kind", "size", "confidence", "position", "file", "time", "shutter", "aperture", "iso", "ev_offset"])
    for i, g in enumerate(groups, 1):
        offsets = g.ev_offsets if g.kind == "exposure" else []
        for pos, s in enumerate(g.shots, 1):
            for p in s.paths:
                m = s.meta
                w.writerow([
                    group_id(i, g), g.kind, g.size, g.confidence, pos, str(p), _fmt_time(m.capture_time),
                    _fmt_shutter(m.exposure_time), m.f_number, m.iso, offsets[pos - 1] if offsets else "",
                ])
    return buf.getvalue()


def to_text(groups: list[Group], errors: list[FileMeta] | None = None, verbose: bool = False) -> str:
    lines = []
    for i, g in enumerate(groups, 1):
        if g.kind == "single":
            if verbose:
                lines.append(f"{group_id(i, g)}  single   {g.shots[0].name}" + (f"  ! {'; '.join(g.warnings)}" if g.warnings else ""))
            continue
        names = ", ".join(s.name for s in g.shots)
        if g.kind == "exposure":
            offs = " ".join(f"{o:+g}" for o in g.ev_offsets)
            head = f"{group_id(i, g)}  bracket  {g.size} frames  step {g.step_ev} EV  [{offs}]  conf {g.confidence:.2f}"
        else:
            head = f"{group_id(i, g)}  focus    {g.size} frames  conf {g.confidence:.2f}"
        lines.append(head)
        lines.append(f"         {names}")
        if verbose and g.evidence:
            lines.append(f"         evidence: {'; '.join(g.evidence)}")
        for w in g.warnings:
            lines.append(f"         ! {w}")
    d = to_dict(groups, errors)["summary"]
    lines.append("")
    lines.append(
        f"{d['shots']} shots ({d['files']} files): {d['exposure_brackets']} exposure brackets, "
        f"{d['focus_stacks']} focus stacks, {d['singles']} singles"
        + (f", {d['unreadable']} unreadable" if d["unreadable"] else "")
    )
    return "\n".join(lines)
