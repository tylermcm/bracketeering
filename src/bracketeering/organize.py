"""Materialise detected groups as folders (copy / move / link)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .model import Group

MODES = ("copy", "move", "symlink", "hardlink")


@dataclass(slots=True)
class Action:
    src: Path
    dst: Path


def folder_name(g: Group, index: int) -> str:
    first = g.shots[0]
    stamp = (
        datetime.fromtimestamp(first.time, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        if first.time is not None else "undated"
    )
    stem = first.meta.path.stem
    if g.kind == "exposure":
        return f"{stamp}_{stem}_HDR{g.size}x{g.step_ev:g}EV"
    return f"{stamp}_{stem}_STACK{g.size}"


def plan(groups: list[Group], dest: Path, include_singles: bool = False, min_confidence: float = 0.0) -> list[Action]:
    actions: list[Action] = []
    used: set[str] = set()
    for i, g in enumerate(groups, 1):
        if g.kind == "single":
            if not include_singles:
                continue
            folder = dest / "_singles"
        else:
            if g.confidence < min_confidence:
                continue
            name = folder_name(g, i)
            while name in used:
                name += "_"
            used.add(name)
            folder = dest / name
        for s in g.shots:
            for p in s.paths:
                actions.append(Action(p, folder / p.name))
    return actions


def execute(actions: list[Action], mode: str) -> list[tuple[Action, str]]:
    """Run the plan. Never overwrites; returns (action, error) for failures."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    failures: list[tuple[Action, str]] = []
    for a in actions:
        try:
            a.dst.parent.mkdir(parents=True, exist_ok=True)
            if a.dst.exists() or a.dst.is_symlink():
                raise FileExistsError(f"{a.dst} already exists")
            if mode == "copy":
                shutil.copy2(a.src, a.dst)
            elif mode == "move":
                shutil.move(str(a.src), str(a.dst))
            elif mode == "symlink":
                os.symlink(a.src.resolve(), a.dst)
            else:
                os.link(a.src, a.dst)
        except OSError as exc:
            failures.append((a, str(exc)))
    return failures
