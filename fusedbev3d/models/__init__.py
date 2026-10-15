"""Model components: lidar pillars, camera LSS lifting, fusion backbone and heads."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "voxelize",
    "PillarEncoder",
    "CameraEncoder",
    "frustum_grid",
    "bev_pool",
    "BEVFusion",
    "CenterHead",
    "OccupancyHead",
    "FusedBEV3D",
]

_LAZY: dict[str, tuple[str, str]] = {
    "voxelize": ("fusedbev3d.models.pillar_encoder", "voxelize"),
    "PillarEncoder": ("fusedbev3d.models.pillar_encoder", "PillarEncoder"),
    "CameraEncoder": ("fusedbev3d.models.camera_encoder", "CameraEncoder"),
    "frustum_grid": ("fusedbev3d.models.camera_encoder", "frustum_grid"),
    "bev_pool": ("fusedbev3d.models.camera_encoder", "bev_pool"),
    "BEVFusion": ("fusedbev3d.models.fusion", "BEVFusion"),
    "CenterHead": ("fusedbev3d.models.heads", "CenterHead"),
    "OccupancyHead": ("fusedbev3d.models.heads", "OccupancyHead"),
    "FusedBEV3D": ("fusedbev3d.models.fusedbev3d", "FusedBEV3D"),
}


def __getattr__(name: str) -> Any:
    """Lazily re-export model symbols without risking an import cycle."""
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
