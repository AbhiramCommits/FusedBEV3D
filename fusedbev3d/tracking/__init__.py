"""Multi-object tracking: constant-velocity Kalman filter and a greedy 3D tracker."""

from __future__ import annotations

__all__ = ["KalmanBoxTracker", "Greedy3DTracker"]


def __getattr__(name: str):
    """Lazily re-export tracker classes."""
    if name == "KalmanBoxTracker":
        from fusedbev3d.tracking.kalman import KalmanBoxTracker

        return KalmanBoxTracker
    if name == "Greedy3DTracker":
        from fusedbev3d.tracking.tracker import Greedy3DTracker

        return Greedy3DTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
