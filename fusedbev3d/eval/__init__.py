"""Evaluation: nuScenes-style 3D detection, CLEAR-MOT tracking and occupancy IoU."""

from __future__ import annotations

__all__ = ["rotated_iou_bev", "evaluate_detection", "print_table", "OccupancyEvaluator"]


def __getattr__(name: str):
    """Lazily re-export the evaluation helpers."""
    if name in {"rotated_iou_bev", "evaluate_detection", "print_table"}:
        from fusedbev3d.eval import detection

        return getattr(detection, name)
    if name in {"OccupancyEvaluator", "evaluate_occupancy"}:
        from fusedbev3d.eval import occupancy

        return getattr(occupancy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
