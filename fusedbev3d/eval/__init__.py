"""Evaluation: nuScenes-style 3D detection, CLEAR-MOT tracking and occupancy IoU."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "rotated_iou_bev",
    "evaluate_detection",
    "print_table",
    "evaluate_occupancy",
    "evaluate_tracking",
]

_LAZY: dict[str, tuple[str, str]] = {
    "rotated_iou_bev": ("fusedbev3d.eval.detection", "rotated_iou_bev"),
    "evaluate_detection": ("fusedbev3d.eval.detection", "evaluate_detection"),
    "print_table": ("fusedbev3d.eval.detection", "print_table"),
    "evaluate_occupancy": ("fusedbev3d.eval.occupancy", "evaluate_occupancy"),
    "evaluate_tracking": ("fusedbev3d.eval.tracking", "evaluate_tracking"),
}


def __getattr__(name: str) -> Any:
    """Lazily re-export the evaluation helpers."""
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
