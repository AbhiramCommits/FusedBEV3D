"""Greedy / Hungarian 3D multi-object tracker with track birth-death management."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fusedbev3d.tracking.kalman import KalmanBoxTracker, wrap_angle

__all__ = ["Greedy3DTracker", "build_cost_matrix"]

try:  # SciPy is a declared dependency but the import is kept defensive
    from scipy.optimize import linear_sum_assignment

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def build_cost_matrix(
    tracks: Sequence[KalmanBoxTracker],
    detections: np.ndarray,
    det_classes: np.ndarray,
    dist_gate: float,
) -> np.ndarray:
    """Centre-distance cost matrix, gated by class and distance.

    Args:
        tracks: Currently alive tracks (their *predicted* boxes are used).
        detections: ``(N, 9)`` detections ``x y z l w h yaw vx vy``.
        det_classes: ``(N,)`` detection class ids.
        dist_gate: Maximum centre distance in metres; larger distances are forbidden.

    Returns:
        ``(T, N)`` cost matrix with large values where association is not allowed.
    """
    t = len(tracks)
    n = detections.shape[0]
    cost = np.full((t, n), 1e6, dtype=np.float64)
    if t == 0 or n == 0:
        return cost
    for i, track in enumerate(tracks):
        pred = track.as_box()
        same_class = det_classes == track.class_id
        if not same_class.any():
            continue
        dist = np.linalg.norm(detections[:, :2] - pred[None, :2], axis=1)
        allowed = same_class & (dist <= dist_gate)
        cost[i, allowed] = dist[allowed]
    return cost


class Greedy3DTracker:
    """Online 3D tracker with selectable greedy or Hungarian association.

    Args:
        max_age: Frames a track may go unmatched before it is deleted.
        min_hits: Consecutive hits required before a track is reported.
        dist_gate: Association gate in metres.
        method: ``"hungarian"`` (SciPy's ``linear_sum_assignment``) or ``"greedy"``.
        dt: Frame period in seconds.
        class_aware: When true, only detections of the track's class can match.
        birth_threshold: Minimum score for an unmatched detection to spawn a new
            track.  ``None`` (or ``0``) lets every fed detection spawn a track.
        report_threshold: Minimum smoothed track score for a track to be reported.
        two_stage: Associate detections to tracks updated in the previous frame
            first, and only then to coasting tracks.  This prevents a stale track
            from stealing a detection and swaps the two identities back and forth.
    """

    def __init__(
        self,
        max_age: int = 2,
        min_hits: int = 1,
        dist_gate: float = 4.0,
        method: str = "hungarian",
        dt: float = 0.5,
        class_aware: bool = True,
        birth_threshold: float | None = None,
        report_threshold: float = 0.0,
        two_stage: bool = True,
    ) -> None:
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.dist_gate = float(dist_gate)
        self.method = method
        self.dt = float(dt)
        self.class_aware = bool(class_aware)
        self.birth_threshold = float(birth_threshold) if birth_threshold is not None else 0.0
        self.report_threshold = float(report_threshold)
        self.two_stage = bool(two_stage)
        self.tracks: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self._next_id = 1

    def reset(self) -> None:
        """Drop all tracks and restart id numbering (used between sequences)."""
        self.tracks = []
        self.frame_count = 0
        self._next_id = 1

    # -- association ------------------------------------------------------------------

    def _associate(self, cost: np.ndarray) -> list[tuple[int, int]]:
        """Return ``(track_index, detection_index)`` pairs from a cost matrix."""
        t, n = cost.shape
        if t == 0 or n == 0:
            return []
        gate = cost < 1e5
        pairs: list[tuple[int, int]] = []
        if self.method == "hungarian" and HAVE_SCIPY:
            rows, cols = linear_sum_assignment(cost)
            pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=False) if gate[r, c]]
        else:
            used_t: set[int] = set()
            used_d: set[int] = set()
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
            for r, c in order:
                r, c = int(r), int(c)
                if r in used_t or c in used_d or not gate[r, c]:
                    continue
                used_t.add(r)
                used_d.add(c)
                pairs.append((r, c))
        return pairs

    # -- public API -------------------------------------------------------------------

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Advance the tracker by one frame.

        Args:
            boxes: ``(N, 9)`` detections ``x y z l w h yaw vx vy``.
            scores: ``(N,)`` detection scores.
            classes: ``(N,)`` detection class ids.

        Returns:
            The list of currently *confirmed* tracks, each a dict with ``box``,
            ``track_id``, ``class_id``, ``score``, ``age`` and ``hits``.
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 9)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)
        self.frame_count += 1

        for track in self.tracks:
            track.predict()

        groups: list[list[int]]
        if self.two_stage:
            fresh = [i for i, t in enumerate(self.tracks) if t.time_since_update == 1]
            stale = [i for i, t in enumerate(self.tracks) if t.time_since_update > 1]
            groups = [fresh, stale] if self.method != "greedy" else [fresh + stale]
        else:
            groups = [list(range(len(self.tracks)))]

        remaining = list(range(boxes.shape[0]))
        matched_d: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for group in groups:
            if not group or not remaining:
                continue
            subset = [self.tracks[i] for i in group]
            det_cls = classes[remaining] if self.class_aware else np.zeros(len(remaining), dtype=np.int64)
            cost = build_cost_matrix(subset, boxes[remaining], det_cls, self.dist_gate)
            for local_t, local_d in self._associate(cost):
                pairs.append((group[local_t], remaining[local_d]))
                matched_d.add(remaining[local_d])
            remaining = [d for d in remaining if d not in matched_d]

        for ti, di in pairs:
            self.tracks[ti].update(boxes[di], float(scores[di]))

        for di in range(boxes.shape[0]):
            if di in matched_d or float(scores[di]) < self.birth_threshold:
                continue
            track = KalmanBoxTracker(
                boxes[di], int(classes[di]), track_id=self._next_id, dt=self.dt
            )
            track.score = float(scores[di])
            self._next_id += 1
            self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results: list[dict[str, Any]] = []
        for t in self.tracks:
            if t.time_since_update > 0 or t.score < self.report_threshold:
                continue
            if t.hits < self.min_hits and self.frame_count > self.min_hits:
                continue
            box = t.as_box()
            box[6] = wrap_angle(box[6])
            results.append(
                {
                    "box": box,
                    "track_id": t.id,
                    "class_id": t.class_id,
                    "score": t.score,
                    "age": t.age,
                    "hits": t.hits,
                }
            )
        return results

    def predict_all(self) -> np.ndarray:
        """Return ``(T, 9)`` current track boxes without advancing the filter."""
        if not self.tracks:
            return np.zeros((0, 9))
        return np.stack([t.as_box() for t in self.tracks])
