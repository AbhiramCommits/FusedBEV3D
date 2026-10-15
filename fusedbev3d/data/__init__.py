"""Dataset layer: procedural multi-modal generator, nuScenes adapter and registry."""

from __future__ import annotations

__all__ = ["SyntheticDrivingDataset", "NuScenesMiniDataset", "build_dataset", "collate_fn"]


def __getattr__(name: str):
    """Lazily re-export dataset classes so `import fusedbev3d.data` is torch-optional."""
    if name == "SyntheticDrivingDataset":
        from fusedbev3d.data.synth import SyntheticDrivingDataset

        return SyntheticDrivingDataset
    if name == "collate_fn":
        from fusedbev3d.data.synth import collate_fn

        return collate_fn
    if name == "NuScenesMiniDataset":
        from fusedbev3d.data.nuscenes_adapter import NuScenesMiniDataset

        return NuScenesMiniDataset
    if name == "build_dataset":
        from fusedbev3d.data.registry import build_dataset

        return build_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
