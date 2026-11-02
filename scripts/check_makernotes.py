#!/usr/bin/env python3
"""Check the native maker-note reader against exiftool on real files.

    python scripts/check_makernotes.py PATH [PATH ...]

For every image, compares what the native reader extracts (bracketing on/off, bracket slot,
drive mode, shutter count) with exiftool's reading of the same tags, and prints mismatches plus a
summary. Needs exiftool on PATH.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bracketeering.metadata import exiftool, read_native  # noqa: E402
from bracketeering.scan import discover  # noqa: E402


def main() -> int:
    if not exiftool.available():
        print("exiftool not found: sudo apt install libimage-exiftool-perl", file=sys.stderr)
        return 2
    files = discover([Path(p) for p in sys.argv[1:] or ["."]])
    ext = {str(m.path): m for m in exiftool.read_exiftool(files)}
    stats: Counter[str] = Counter()
    shown = 0
    for f in files:
        n, e = read_native(f), ext.get(str(f))
        if e is None or e.error:
            stats["exiftool failed"] += 1
            continue
        stats["files"] += 1
        checks = {
            "bracket_kind": (n.bracket_kind, e.bracket_kind),
            "bracket_value": (None if n.bracket_value is None else round(n.bracket_value, 2),
                              None if e.bracket_value is None else round(e.bracket_value, 2)),
            "drive_mode": (n.drive_mode, e.drive_mode),
            "shutter_count": (n.shutter_count, e.shutter_count),
        }
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        for k in checks:
            stats[f"{k} {'ok' if k not in bad else 'MISMATCH'}"] += 1
        if bad and shown < 25:
            shown += 1
            print(f"{f.name}: " + "; ".join(f"{k} native={v[0]!r} exiftool={v[1]!r}" for k, v in bad.items()))
    print()
    for k, v in sorted(stats.items()):
        print(f"{k:28} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
