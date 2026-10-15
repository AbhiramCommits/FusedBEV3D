"""Model components: lidar pillars, camera LSS lifting, fusion backbone and heads."""

from __future__ import annotations

__all__ = [
    "voxelize",
    "PillarEncoder",
    "CameraEncoder",
    "BEVFusion",
    "CenterHead",
    "OccupancyHead",
    "FusedBEV3D",
]


def __getattr__(name: str):
    """Lazily re-export model classes so torch is only imported when needed."""
    from fusedbev3d.models import (  # noqa: PLC0415
        camera_encoder,
        fusedbev3d,
        fusion,
        heads,
        pillar_encoder,
    )

    mapping = {
        "voxelize": pillar_encoder.voxelize,
        "PillarEncoder": pillar_encoder.PillarEncoder,
        "CameraEncoder": camera_encoder.CameraEncoder,
        "frustum_grid": camera_encoder.frustum_grid,
        "bev_pool": camera_encoder.bev_pool,
        "BEVFusion": fusion.BEVFusion,
        "CenterHead": heads.CenterHead,
        "OccupancyHead": heads.OccupancyHead,
        "FusedBEV3D": fusedbev3d.FusedBEV3D,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
