"""Serialise detection results to JSON / CSV / human-readable text.

JSON schema "bracketeering/2" (see SCHEMA.md) is the machine contract for
pipelines: every input file appears exactly once in `files`, keyed by path.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone

from . import __version__
from .model import FileMeta, Group, Shot

SCHEMA = "bracketeering/2"
_PREFIX = {"exposure": "B", "focus": "F", "burst": "U", "timelapse": "T", "single": "S"}


def _fmt_time(t: float | None, offset: float | None = None) -> str | None:
    """Camera-local time with its UTC offset when the file recorded one, otherwise camera-local time."""
    if t is None:
        return None
    if offset is not None:
        return datetime.fromtimestamp(t, tz=timezone(timedelta(seconds=offset))).isoformat(timespec="milliseconds")
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "")


def _fmt_shutter(t: float | None) -> str | None:
    if t is None:
        return None
    return f"1/{round(1 / t)}" if t < 0.3 else f"{t:g}s"


def group_id(index: int, g: Group) -> str:
    return f"{_PREFIX.get(g.kind, 'S')}{index:05d}"


def frame_role(g: Group, pos: int) -> str:
    if g.kind == "single":
        return "single"
    return "representative" if pos == g.representative else "member"


def _frame_dict(g: Group, pos: int, s: Shot) -> dict:
    m = s.meta
    d = {
        "files": [str(p) for p in s.paths],
        "role": frame_role(g, pos),
        "time": _fmt_time(m.capture_time, m.utc_offset),
        "shutter": _fmt_shutter(m.exposure_time),
        "aperture": m.f_number,
        "iso": m.iso,
        "exposure_bias": m.exposure_bias,
        "ev": round(s.ev, 3) if s.ev is not None else None,
    }
    if g.best_exposed is not None:
        d["best_exposed"] = pos == g.best_exposed
    if s.pixels is not None:
        d["exposure"] = s.pixels.to_dict()  # type: ignore[attr-defined]
    return d


def to_dict(groups: list[Group], errors: list[FileMeta] | None = None) -> dict:
    errs = {str(e.path): e.error for e in (errors or [])}
    out_groups, files = [], {}
    scene_sizes: dict[int, int] = {}
    for g in groups:
        if g.scene is not None:
            scene_sizes[g.scene] = scene_sizes.get(g.scene, 0) + 1
    for i, g in enumerate(groups, 1):
        gid = group_id(i, g)
        m = g.shots[0].meta
        entry = {
            "id": gid,
            "kind": g.kind,
            "size": g.size,
            "confidence": g.confidence,
            "camera": " ".join(x for x in (m.make, m.model) if x) or None,
            "start": _fmt_time(g.shots[0].time, m.utc_offset),
            "representative": str(g.shots[g.representative].meta.path),
            "best_exposed": str(g.shots[g.best_exposed].meta.path) if g.best_exposed is not None else None,
            "scene": g.scene,
            "ev_offsets": g.ev_offsets if g.kind == "exposure" else None,
            "step_ev": g.step_ev if g.kind == "exposure" else None,
            "evidence": g.evidence,
            "warnings": g.warnings,
            "frames": [_frame_dict(g, pos, s) for pos, s in enumerate(g.shots)],
        }
        out_groups.append(entry)
        for pos, s in enumerate(g.shots):
            for f in s.files:
                key = str(f.path)
                files[key] = {
                    "group": gid,
                    "kind": g.kind,
                    "role": frame_role(g, pos),
                    "position": pos + 1,
                    "primary": f is s.meta,  # False for a sidecar (e.g. the JPEG of a RAW+JPEG pair)
                    "scene": g.scene,
                    "best_exposed": (pos == g.best_exposed) if g.best_exposed is not None else None,
                    "error": errs.get(key),
                }
    count = {k: sum(g.kind == k for g in groups) for k in ("exposure", "focus", "burst", "timelapse", "single")}
    return {
        "schema": SCHEMA,
        "tool_version": __version__,
        "summary": {
            "shots": sum(g.size for g in groups),
            "files": len(files),
            "exposure_brackets": count["exposure"],
            "focus_stacks": count["focus"],
            "bursts": count["burst"],
            "timelapses": count["timelapse"],
            "singles": count["single"],
            "near_duplicate_scenes": sum(n > 1 for n in scene_sizes.values()),
            "unreadable": len(errs),
        },
        "groups": out_groups,
        "files": files,
        "errors": [{"file": k, "error": v} for k, v in errs.items()],
    }


def to_json(groups: list[Group], errors: list[FileMeta] | None = None) -> str:
    return json.dumps(to_dict(groups, errors), indent=2)


def to_csv(groups: list[Group]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["file", "group_id", "kind", "role", "position", "group_size", "confidence", "scene", "best_exposed",
                "time", "shutter", "aperture", "iso", "ev_offset", "exposure_flag", "highlights_clipped", "shadows_crushed"])
    for i, g in enumerate(groups, 1):
        offsets = g.ev_offsets if g.kind == "exposure" else []
        for pos, s in enumerate(g.shots):
            m, px = s.meta, s.pixels
            for p in s.paths:
                w.writerow([
                    str(p), group_id(i, g), g.kind, frame_role(g, pos), pos + 1, g.size, g.confidence,
                    g.scene if g.scene is not None else "",
                    "" if g.best_exposed is None else pos == g.best_exposed,
                    _fmt_time(m.capture_time, m.utc_offset), _fmt_shutter(m.exposure_time), m.f_number, m.iso,
                    offsets[pos] if offsets else "",
                    px.flag if px else "", round(px.highlights, 4) if px else "", round(px.shadows, 4) if px else "",  # type: ignore[attr-defined]
                ])
    return buf.getvalue()


def to_text(groups: list[Group], errors: list[FileMeta] | None = None, verbose: bool = False) -> str:
    lines = []
    for i, g in enumerate(groups, 1):
        scene = f"  scene {g.scene}" if g.scene is not None else ""
        if g.kind == "single":
            if verbose:
                lines.append(f"{group_id(i, g)}  single   {g.shots[0].name}{scene}" + (f"  ! {'; '.join(g.warnings)}" if g.warnings else ""))
            continue
        names = ", ".join(("*" if pos == g.representative else "") + s.name for pos, s in enumerate(g.shots))
        if g.kind == "exposure":
            offs = " ".join(f"{o:+g}" for o in g.ev_offsets)
            head = f"{group_id(i, g)}  bracket  {g.size} frames  step {g.step_ev} EV  [{offs}]  conf {g.confidence:.2f}"
        elif g.kind == "burst":
            head = f"{group_id(i, g)}  burst    {g.size} frames  conf {g.confidence:.2f}"
        elif g.kind == "timelapse":
            head = f"{group_id(i, g)}  timelapse {g.size} frames  conf {g.confidence:.2f}"
        else:
            head = f"{group_id(i, g)}  focus    {g.size} frames  conf {g.confidence:.2f}"
        lines.append(head + scene)
        lines.append(f"         {names}")
        if g.best_exposed is not None:
            lines.append(f"         best exposed: {g.shots[g.best_exposed].name}")
        if verbose and g.evidence:
            lines.append(f"         evidence: {'; '.join(g.evidence)}")
        for w in g.warnings:
            lines.append(f"         ! {w}")
    d = to_dict(groups, errors)["summary"]
    lines.append("")
    lines.append(
        f"{d['shots']} shots ({d['files']} files): {d['exposure_brackets']} exposure brackets, "
        f"{d['bursts']} bursts, {d['timelapses']} timelapses, {d['focus_stacks']} focus stacks, {d['singles']} singles"
        + (f", {d['near_duplicate_scenes']} near-duplicate scenes" if any(g.scene is not None for g in groups) else "")
        + (f", {d['unreadable']} unreadable" if d["unreadable"] else "")
        + "   (* = representative frame)"
    )
    return "\n".join(lines)
