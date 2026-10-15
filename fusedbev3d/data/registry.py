"""Dataset registry: turn a config into a concrete dataset instance."""

from __future__ import annotations

from pathlib import Path

from fusedbev3d.config import Config, DataConfig, GridConfig
from fusedbev3d.data.synth import SyntheticDrivingDataset

__all__ = ["build_dataset", "available_datasets", "cache_path"]

#: Datasets addressable by :func:`build_dataset`.
DATASETS: tuple[str, ...] = ("synth", "nuscenes-mini")


def available_datasets() -> tuple[str, ...]:
    """Return the names accepted by :func:`build_dataset`."""
    return DATASETS


def cache_path(split: str, cfg: DataConfig, name: str | None = None) -> Path:
    """Return the deterministic cache path for ``(dataset, split)``."""
    name = name or cfg.name
    return Path(cfg.cache_dir) / f"{name}_{split}.pt"


def build_dataset(
    name: str | None = None,
    split: str = "train",
    cfg: Config | DataConfig | None = None,
    grid: GridConfig | None = None,
    num_scenes: int | None = None,
    use_cache: bool = True,
) -> SyntheticDrivingDataset:
    """Build a dataset by name.

    Args:
        name: ``"synth"`` or ``"nuscenes-mini"``; defaults to ``cfg.name``.
        split: ``"train"`` or ``"val"``.
        cfg: A :class:`~fusedbev3d.config.Config` (its ``data`` group is used) or a
            bare :class:`~fusedbev3d.config.DataConfig`.  Defaults to ``DataConfig()``.
        grid: Optional grid override.
        num_scenes: Optional scene-count override.
        use_cache: Load ``data/cache/<name>_<split>.pt`` when it exists.

    Returns:
        A dataset exposing the shared sample dictionary.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    data_cfg: DataConfig
    if cfg is None:
        data_cfg = DataConfig()
    elif isinstance(cfg, Config):
        data_cfg = cfg.data
    else:
        data_cfg = cfg
    name = name or data_cfg.name
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {DATASETS}")
    if grid is None and isinstance(cfg, Config):
        grid = cfg.grid
    cache = cache_path(split, data_cfg, name) if use_cache else None

    if name == "synth":
        return SyntheticDrivingDataset(
            split=split, num_scenes=num_scenes, grid=grid, cfg=data_cfg, cache=cache
        )

    from fusedbev3d.data.nuscenes_adapter import NuScenesMiniDataset

    return NuScenesMiniDataset(
        split=split, num_scenes=num_scenes, grid=grid, cfg=data_cfg, cache=cache
    )
