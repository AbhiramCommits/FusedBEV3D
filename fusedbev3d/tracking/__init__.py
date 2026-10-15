"""Multi-object tracking: constant-velocity Kalman filter and a greedy 3D tracker."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["KalmanBoxTracker", "Greedy3DTracker"]

_LAZY: dict[str, tuple[str, str]] = {
    "KalmanBoxTracker": ("fusedbev3d.tracking.kalman", "KalmanBoxTracker"),
    "Greedy3DTracker": ("fusedbev3d.tracking.tracker", "Greedy3DTracker"),
}


def __getattr__(name: str) -> Any:
    """Lazily re-export the tracker classes."""
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
