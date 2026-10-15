"""Constant-velocity Kalman filter for 3D boxes.

State is ``(x, y, z, yaw, vx, vy)``; the measurement is ``(x, y, z, yaw)``.  The
motion model is constant velocity in the plane with a static height and heading
(the heading is instead corrected by the measurement), which is exactly the
``constant-velocity + slight yaw-rate`` motion the synthetic generator produces.
Box dimensions are not part of the filter state; they are smoothed with an
exponential moving average, which is far more stable than filtering log-dims.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["KalmanBoxTracker", "wrap_angle"]

STATE_DIM = 6
MEAS_DIM = 4


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class KalmanBoxTracker:
    """A single tracked object with a constant-velocity Kalman filter.

    Args:
        box: ``(9,)`` detection ``x y z l w h yaw vx vy``.
        class_id: Integer class id of the object.
        track_id: Unique integer id assigned to this track.
        dt: Frame period in seconds.
        process_std: Standard deviation of the constant-velocity process noise.
        meas_std: ``(4,)`` measurement noise std for ``x, y, z, yaw``.
        size_alpha: EMA factor for box dimensions (1.0 disables smoothing).
    """

    def __init__(
        self,
        box: np.ndarray,
        class_id: int,
        track_id: int,
        dt: float = 0.5,
        process_std: float = 0.6,
        meas_std: tuple[float, float, float, float] = (0.25, 0.25, 0.3, 0.15),
        size_alpha: float = 0.35,
    ) -> None:
        self.dt = float(dt)
        self.id = int(track_id)
        self.class_id = int(class_id)
        self.size_alpha = float(size_alpha)
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.score = 1.0

        box = np.asarray(box, dtype=np.float64).reshape(-1)
        self.x = np.array([box[0], box[1], box[2], wrap_angle(box[6]), box[7], box[8]], dtype=np.float64)
        self.P = np.eye(STATE_DIM) * np.array([1.0, 1.0, 1.0, 0.1, 4.0, 4.0])
        self.size = np.array([box[3], box[4], box[5]], dtype=np.float64)
        self.f = self._motion_matrix()
        self.q = np.diag([0.25, 0.25, 0.06, 0.02, process_std**2, process_std**2])
        self.h = np.zeros((MEAS_DIM, STATE_DIM))
        self.h[0, 0] = self.h[1, 1] = self.h[2, 2] = self.h[3, 3] = 1.0
        self.r = np.diag(np.square(np.asarray(meas_std, dtype=np.float64)))

    def _motion_matrix(self) -> np.ndarray:
        """Return the ``dt``-dependent 6x6 constant-velocity transition matrix."""
        f = np.eye(STATE_DIM)
        f[0, 4] = f[1, 5] = self.dt
        return f

    def predict(self) -> np.ndarray:
        """Advance the state one frame and return the predicted ``(9,)`` box."""
        self.x = self.f @ self.x
        self.P = self.f @ self.P @ self.f.T + self.q
        self.age += 1
        self.time_since_update += 1
        if self.hit_streak > 0:
            self.hit_streak += 1
        return self.as_box()

    def update(self, box: np.ndarray, score: float = 1.0) -> None:
        """Correct the state with a matched detection.

        Args:
            box: ``(9,)`` detection ``x y z l w h yaw vx vy``.
            score: Detection confidence (kept for bookkeeping/score smoothing).
        """
        box = np.asarray(box, dtype=np.float64).reshape(-1)
        z = np.array([box[0], box[1], box[2], wrap_angle(box[6])])
        y = z - self.h @ self.x
        y[3] = wrap_angle(y[3])
        s = self.h @ self.P @ self.h.T + self.r
        k = self.P @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.x[3] = wrap_angle(self.x[3])
        self.P = (np.eye(STATE_DIM) - k @ self.h) @ self.P
        alpha = self.size_alpha
        self.size = (1 - alpha) * self.size + alpha * np.array([box[3], box[4], box[5]])
        self.score = 0.7 * self.score + 0.3 * float(score)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def as_box(self) -> np.ndarray:
        """Return the current state as ``(9,)`` ``x y z l w h yaw vx vy``."""
        return np.array(
            [self.x[0], self.x[1], self.x[2], self.size[0], self.size[1], self.size[2], self.x[3], self.x[4], self.x[5]],
            dtype=np.float64,
        )

    @property
    def confirmed(self) -> bool:
        """Whether this track has ever been observed in consecutive frames."""
        return self.hit_streak > 0 and self.time_since_update == 0

    def info(self) -> dict[str, Any]:
        """Return a short diagnostic dictionary for logging."""
        return {
            "id": self.id,
            "class_id": self.class_id,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "score": self.score,
        }
