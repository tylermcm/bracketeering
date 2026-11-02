"""Command-line interface: `bracketeer scan` and `bracketeer organize`."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import __version__, analyze
from .engine import EngineConfig
from .metadata import BACKENDS
from .organize import MODES, execute, plan
from .report import to_csv, to_json, to_text


_DEFAULTS = EngineConfig()


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="+", type=Path, help="files or folders to scan")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    p.add_argument("--backend", choices=BACKENDS, default="auto",
                   help="metadata reader: auto (native + exiftool fallback), native, or exiftool (adds maker-note bracket tags)")
    p.add_argument("--workers", type=int, help="parallel metadata readers")
    p.add_argument("--max-gap", type=float, default=_DEFAULTS.max_gap,
                   help="max seconds between the end of one frame and the start of the next within a bracket (default %(default)s)")
    p.add_argument("--manual-gap", type=float, default=_DEFAULTS.manual_max_gap,
                   help="max gap for user-driven brackets shot in Manual mode (default %(default)s)")
    p.add_argument("--max-size", type=int, default=_DEFAULTS.max_size, help="largest bracket size (default %(default)s)")
    p.add_argument("--min-size", type=int, default=_DEFAULTS.min_size, help="smallest bracket size (default %(default)s)")
    p.add_argument("--no-focus", action="store_true", help="disable focus-stack detection")


def _config(a: argparse.Namespace) -> EngineConfig:
    return EngineConfig(max_gap=a.max_gap, manual_max_gap=a.manual_gap, max_size=a.max_size,
                        min_size=a.min_size, detect_focus=not a.no_focus)


def _run(a: argparse.Namespace):
    t0 = time.perf_counter()
    groups, errors = analyze(a.paths, _config(a), backend=a.backend, recursive=not a.no_recursive, workers=a.workers)
    n = sum(g.size for g in groups)
    print(f"analysed {n} shots in {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    return groups, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bracketeer", description="Detect and group camera exposure brackets and focus stacks.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("scan", help="detect brackets and print a report")
    _add_common(sp)
    sp.add_argument("--format", choices=("text", "json", "csv"), default="text")
    sp.add_argument("-o", "--output", type=Path, help="write the report to a file")
    sp.add_argument("-v", "--verbose", action="store_true", help="show singles and evidence")

    op = sub.add_parser("organize", help="copy/move/link each bracket into its own folder")
    _add_common(op)
    op.add_argument("--dest", type=Path, required=True, help="destination root")
    op.add_argument("--mode", choices=MODES, default="copy")
    op.add_argument("--include-singles", action="store_true", help="also place singles in <dest>/_singles")
    op.add_argument("--min-confidence", type=float, default=0.5)
    op.add_argument("--dry-run", action="store_true", help="print the plan without touching files")

    a = ap.parse_args(argv)
    try:
        groups, errors = _run(a)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if a.cmd == "scan":
        out = {"json": lambda: to_json(groups, errors), "csv": lambda: to_csv(groups)}.get(
            a.format, lambda: to_text(groups, errors, verbose=a.verbose))()
        if a.output:
            a.output.write_text(out + ("\n" if not out.endswith("\n") else ""), encoding="utf-8")
        else:
            try:
                print(out)
                sys.stdout.flush()
            except BrokenPipeError:  # e.g. piped into `head`
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0

    actions = plan(groups, a.dest, include_singles=a.include_singles, min_confidence=a.min_confidence)
    if a.dry_run:
        for act in actions:
            print(f"{a.mode}: {act.src} -> {act.dst}")
        print(f"{len(actions)} files would be processed", file=sys.stderr)
        return 0
    failures = execute(actions, a.mode)
    for act, err in failures:
        print(f"failed: {act.src}: {err}", file=sys.stderr)
    print(f"{len(actions) - len(failures)}/{len(actions)} files processed", file=sys.stderr)
    return 1 if failures else 0
