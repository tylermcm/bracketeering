"""bracketeering: fast, accurate camera bracket detection and grouping."""

from __future__ import annotations

__version__ = "0.1.0"

from pathlib import Path  # noqa: E402

from .engine import EngineConfig, detect  # noqa: E402
from .model import FileMeta, Group, Shot  # noqa: E402


def analyze(
    paths: list[str | Path],
    config: EngineConfig | None = None,
    backend: str = "auto",
    recursive: bool = True,
    workers: int | None = None,
) -> tuple[list[Group], list[FileMeta]]:
    """Scan files/directories and return (groups, unreadable_files)."""
    from .metadata import read_many
    from .scan import discover, pair_sidecars

    files = discover([Path(p) for p in paths], recursive=recursive)
    metas = read_many(files, backend=backend, workers=workers)
    errors = [m for m in metas if m.error]
    shots = pair_sidecars(metas)
    return detect(shots, config), errors


__all__ = ["EngineConfig", "FileMeta", "Group", "Shot", "__version__", "analyze", "detect"]
