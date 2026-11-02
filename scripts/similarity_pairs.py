#!/usr/bin/env python3
"""Print preview similarity for pairs of frames, to calibrate the scene / recompose thresholds.

    python scripts/similarity_pairs.py A.NEF B.NEF [C.NEF D.NEF ...]

Each pair prints its similarity (-1..1). The engine treats >= 0.6 as the same scene
(--scene-similarity) and < 0.35 as recomposed / a different subject.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bracketeering.pixels import SAME_SCENE, DIFFERENT, analyse_file, thumb_similarity  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        print(__doc__)
        return 2
    for a, b in zip(args[::2], args[1::2]):
        sa, sb = analyse_file(Path(a)), analyse_file(Path(b))
        if sa is None or sb is None or sa.thumb is None or sb.thumb is None:
            print(f"{a}  {b}  (no preview)")
            continue
        sim = thumb_similarity(sa.thumb, sb.thumb)
        verdict = "same scene" if sim >= SAME_SCENE else "different" if sim < DIFFERENT else "in between"
        hd = (sa.dhash ^ sb.dhash).bit_count()
        print(f"{Path(a).name:16} {Path(b).name:16} similarity {sim:+.3f}  ({verdict}; dHash distance {hd})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
