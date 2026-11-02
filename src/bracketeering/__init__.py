"""bracketeering: fast, accurate camera bracket detection and grouping."""

from __future__ import annotations

__version__ = "0.3.2"

from pathlib import Path  # noqa: E402

from .engine import EngineConfig, detect  # noqa: E402
from .model import FileMeta, Group, Shot  # noqa: E402
from .pixels import PixelConfig  # noqa: E402


def analyze(
    paths: list[str | Path],
    config: EngineConfig | None = None,
    backend: str = "auto",
    recursive: bool = True,
    workers: int | None = None,
    pixels: bool = False,
    pixel_config: "PixelConfig | None" = None,
    exclude: list[str] | tuple[str, ...] = (),
) -> tuple[list[Group], list[FileMeta]]:
    """Scan files/directories and return (groups, unreadable_files).

    pixels=True also decodes embedded previews (needs Pillow) for exposure
    stats, each bracket's best-exposed frame, and near-duplicate scenes.
    """
    from .metadata import read_many
    from .scan import discover, pair_sidecars

    files = discover([Path(p) for p in paths], recursive=recursive, exclude=exclude)
    metas = read_many(files, backend=backend, workers=workers)
    errors = [m for m in metas if m.error]
    shots = pair_sidecars(metas)
    if pixels:
        from .pixels import PixelConfig, analyse_shots, finish_groups

        pcfg = pixel_config or PixelConfig()
        analyse_shots(shots, pcfg, workers=workers)
        groups = detect(shots, config)
        finish_groups(groups, pcfg)
    else:
        groups = detect(shots, config)
    return groups, errors


__all__ = ["EngineConfig", "FileMeta", "Group", "PixelConfig", "Shot", "__version__", "analyze", "detect"]
