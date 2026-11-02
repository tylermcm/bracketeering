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


def _identities(s: Shot) -> list[tuple]:
    """Keys that only files of the *same capture* share.

    Copies keep the body's shutter count; edited exports (other folder, name suffix or format)
    usually lose the maker notes but keep the capture time, frame number and exposure.
    """
    m = s.meta
    cam = s.camera_key
    if not any(cam):
        return []
    keys: list[tuple] = []
    if m.shutter_count is not None:
        keys.append((cam, "count", m.shutter_count))
    if m.capture_time is not None and m.file_number is not None:
        t = round(m.capture_time, 2) if m.time_precise else int(m.capture_time)
        keys.append((cam, "time", t, m.file_number, m.exposure_time, m.f_number, m.iso))
        if m.time_precise:  # an export without sub-seconds still matches its source's whole second
            keys.append((cam, "time", int(m.capture_time), m.file_number, m.exposure_time, m.f_number, m.iso))
    return keys


def merge_copies(shots: list[Shot]) -> list[Shot]:
    """Merge copies and edited exports of one capture (e.g. a "_picks" or "Edits" folder) into one shot.

    Without this, a copy and its original look like two frames 0 s apart: a fake burst. Shots sharing
    any identity key are merged; the RAW in the shallowest folder becomes the primary file.
    """
    parent = list(range(len(shots)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_with: dict[tuple, int] = {}
    for i, s in enumerate(shots):
        for key in _identities(s):
            j = first_with.setdefault(key, i)
            if j != i:
                parent[find(i)] = find(j)
    members: dict[int, list[Shot]] = defaultdict(list)
    for i, s in enumerate(shots):
        members[find(i)].append(s)
    out: list[Shot] = []
    for same in members.values():
        if len(same) == 1:
            out.append(same[0])
            continue
        same.sort(key=lambda s: (len(s.meta.path.parts), str(s.meta.path)))
        out.append(Shot([f for s in same for f in s.files]))  # primary: see Shot.__post_init__
    return out
