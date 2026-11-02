"""File discovery and RAW+JPEG sidecar pairing."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .model import IMAGE_EXTENSIONS, FileMeta, Shot

SIDECAR_TIME_TOLERANCE = 1.5  # seconds


def discover(roots: Iterable[Path], recursive: bool = True) -> list[Path]:
    """Find image files under the given roots (files are accepted as-is)."""
    found: list[Path] = []
    stack = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.append(root)
        else:
            stack.append(root)
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.name.startswith("."):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if recursive:
                            stack.append(Path(entry.path))
                    elif os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                        found.append(Path(entry.path))
        except OSError:
            continue
    found.sort()
    return found


def pair_sidecars(metas: list[FileMeta]) -> list[Shot]:
    """Merge files sharing a directory and stem (IMG_1.CR3 + IMG_1.JPG) into one Shot.

    Files are only merged when their capture times agree, which protects
    against unrelated files that happen to share a name.
    """
    buckets: dict[tuple[str, str], list[FileMeta]] = defaultdict(list)
    for m in metas:
        buckets[(str(m.path.parent), m.path.stem.lower())].append(m)
    shots: list[Shot] = []
    for files in buckets.values():
        clusters: list[list[FileMeta]] = []
        for f in sorted(files, key=lambda f: (not f.is_raw, f.path.name)):
            for c in clusters:
                ref = c[0].capture_time
                if ref is None or f.capture_time is None or abs(ref - f.capture_time) <= SIDECAR_TIME_TOLERANCE:
                    c.append(f)
                    break
            else:
                clusters.append([f])
        shots.extend(Shot(c) for c in clusters)
    return shots
