"""Metadata extraction front-end."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..model import FileMeta
from . import exiftool
from .native import read_native

BACKENDS = ("auto", "native", "exiftool")


def read_many(paths: list[Path], backend: str = "auto", workers: int | None = None) -> list[FileMeta]:
    """Read metadata for many files, preserving input order.

    auto      native reader (fast) with exiftool fallback for files it cannot parse
    native    built-in reader only
    exiftool  exiftool for everything (adds maker-note bracket tags; slower)
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "exiftool":
        if not exiftool.available():
            raise RuntimeError("exiftool not found on PATH")
        return exiftool.read_exiftool(paths, workers=workers)
    workers = workers or min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        metas = list(pool.map(read_native, paths, chunksize=16))
    if backend == "auto" and exiftool.available():
        failed = [i for i, m in enumerate(metas) if m.error or m.exposure_time is None]
        if failed:
            for i, m in zip(failed, exiftool.read_exiftool([paths[i] for i in failed])):
                if not m.error or metas[i].error:
                    metas[i] = m  # exiftool's reading, or its (more specific) error
    return metas


__all__ = ["BACKENDS", "read_many", "read_native"]
