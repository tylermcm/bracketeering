"""File discovery and RAW+JPEG sidecar pairing."""

from __future__ import annotations

import fnmatch
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .model import IMAGE_EXTENSIONS, FileMeta, Shot

SIDECAR_TIME_TOLERANCE = 1.5  # seconds


def discover(roots: Iterable[Path], recursive: bool = True, exclude: Iterable[str] = ()) -> list[Path]:
    """Find image files under the given roots (files are accepted as-is).

    `exclude` holds glob patterns matched against folder names (e.g. "_winners", "*export*").
    """
    patterns = list(exclude)
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
                        if recursive and not any(fnmatch.fnmatch(entry.name, p) for p in patterns):
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
    return merge_copies(shots)


def _identity(s: Shot) -> tuple | None:
    """A key that only two files of the *same capture* share."""
    m = s.meta
    cam = s.camera_key
    if not any(cam):
        return None
    if m.shutter_count is not None:
        return (cam, "count", m.shutter_count)
    if m.capture_time is not None and m.time_precise and m.file_number is not None:
        return (cam, "time", round(m.capture_time, 3), m.file_number, m.exposure_time, m.f_number, m.iso)
    return None


def merge_copies(shots: list[Shot]) -> list[Shot]:
    """Merge copies of one capture found in different folders (e.g. a "_picks" folder) into one shot.

    Without this, a copy and its original look like two frames 0 s apart: a fake burst. The file
    in the shallowest folder becomes the primary; the copies stay listed under the same shot.
    """
    by_key: dict[tuple, list[Shot]] = defaultdict(list)
    out: list[Shot] = []
    for s in shots:
        key = _identity(s)
        if key is None:
            out.append(s)
        else:
            by_key[key].append(s)
    for same in by_key.values():
        if len(same) == 1:
            out.append(same[0])
            continue
        same.sort(key=lambda s: (len(s.meta.path.parts), str(s.meta.path)))
        files = [f for s in same for f in s.files]
        out.append(Shot(files))  # primary: the RAW in the shallowest folder (see Shot.__post_init__)
    return out
