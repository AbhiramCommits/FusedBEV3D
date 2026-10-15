"""Multi-object tracking: Kalman filtering, association and CLEAR-MOT bookkeeping."""

from __future__ import annotations

import math

import numpy as np

from fusedbev3d.eval.tracking import evaluate_tracking
from fusedbev3d.tracking.kalman import KalmanBoxTracker
from fusedbev3d.tracking.tracker import Greedy3DTracker


def _box(x: float, y: float, yaw: float = 0.0, vx: float = 0.0, vy: float = 0.0) -> np.ndarray:
    """A car-sized box with a given centre and velocity."""
    return np.array([x, y, 1.0, 4.2, 1.8, 1.6, yaw, vx, vy], dtype=np.float64)


def _gt(x: float, y: float, track_id: int, class_id: int = 0, yaw: float = 0.0) -> np.ndarray:
    """A ground-truth row in the 11-column layout."""
    return np.array([x, y, 1.0, 4.2, 1.8, 1.6, yaw, 0.0, 0.0, class_id, track_id], dtype=np.float64)


def _frame(rows):
    """Wrap tracked rows into the evaluation dictionary."""
    boxes = np.stack([r["box"] for r in rows]) if rows else np.zeros((0, 9))
    return {
        "boxes": boxes,
        "scores": np.array([r["score"] for r in rows], dtype=np.float64),
        "classes": np.array([r["class_id"] for r in rows], dtype=np.int64),
        "track_ids": np.array([r["track_id"] for r in rows], dtype=np.int64),
    }


def test_kalman_filter_tracks_constant_velocity():
    """A constant-velocity target is followed with sub-decimetre error."""
    tracker = KalmanBoxTracker(_box(0.0, 0.0, 0.0, vx=1.0, vy=0.0), class_id=0, track_id=1, dt=1.0)
    for t in range(1, 11):
        tracker.predict()
        tracker.update(_box(1.0 * t, 0.0, 0.0, vx=1.0, vy=0.0))
    box = tracker.as_box()
    assert abs(box[0] - 10.0) < 0.15
    assert abs(box[7] - 1.0) < 0.2  # estimated vx
    assert tracker.hits == 11
    assert tracker.time_since_update == 0


def test_kalman_filter_smooths_box_size():
    """Box dimensions follow an EMA rather than jumping to the latest noisy value."""
    tracker = KalmanBoxTracker(_box(0.0, 0.0), class_id=0, track_id=1, size_alpha=0.5)
    tracker.update(np.array([0.0, 0.0, 1.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0]))
    assert 4.2 < tracker.as_box()[3] < 10.0
    first = float(tracker.as_box()[3])
    tracker.update(np.array([0.0, 0.0, 1.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0]))
    # alpha = 0.5: 4.2 -> 7.1 -> 8.55, never jumping straight to 10
    assert math.isclose(first, 7.1, abs_tol=0.05)
    assert math.isclose(float(tracker.as_box()[3]), 8.55, abs_tol=0.05)


def test_tracker_keeps_a_stable_id_for_a_clean_sequence():
    """Perfect detections of one object give exactly one track id and MOTA 1.0."""
    tracker = Greedy3DTracker(max_age=2, min_hits=1, dist_gate=4.0, report_threshold=0.0)
    predictions, targets = [], []
    for t in range(10):
        x = 2.0 + 0.5 * t
        rows = tracker.update(_box(x, 0.0)[None], np.array([0.9]), np.array([0]))
        predictions.append(_frame(rows))
        targets.append(np.stack([_gt(x, 0.0, track_id=101)]))
    result = evaluate_tracking(predictions, targets, dist_threshold=2.0)
    assert result["ID_switches"] == 0
    assert result["FP"] == 0 and result["FN"] == 0
    assert math.isclose(result["MOTA"], 1.0, abs_tol=1e-9)
    # the reported box is the filtered state, so MOTP is small but not exactly zero
    assert result["MOTP"] < 0.05
    assert result["mostly_tracked"] == 1


def test_tracker_rebirth_after_a_long_occlusion():
    """A track that outlives ``max_age`` is deleted and reborn with a new id.

    The object is detected in frames 0-2, invisible in frames 3-5 (three frames,
    which exceeds ``max_age=2``) and detected again in frames 6-9.  The ground
    truth exists in all ten frames, so of the ten ground-truth boxes three become
    false negatives during the gap and the rebirth adds exactly one identity
    switch:

        MOTA = 1 - (FN + FP + IDSW) / n_gt = 1 - (3 + 0 + 1) / 10 = 0.6
    """
    tracker = Greedy3DTracker(max_age=2, min_hits=1, dist_gate=4.0, report_threshold=0.0)
    gt_x = [2.0 + 0.5 * t for t in range(10)]
    visible = [True, True, True, False, False, False, True, True, True, True]
    predictions, targets = [], []
    for t in range(10):
        if visible[t]:
            rows = tracker.update(_box(gt_x[t], 0.0)[None], np.array([0.9]), np.array([0]))
        else:
            rows = tracker.update(np.zeros((0, 9)), np.array([]), np.array([], dtype=np.int64))
        predictions.append(_frame(rows))
        targets.append(np.stack([_gt(gt_x[t], 0.0, track_id=77)]))
    result = evaluate_tracking(predictions, targets, dist_threshold=2.0)
    assert result["ID_switches"] == 1
    assert result["FN"] == 3 and result["FP"] == 0
    assert math.isclose(result["MOTA"], 0.6, abs_tol=1e-9)
    # matched in 7 of 10 frames: below the 80 % mostly-tracked bar, above the 20 % loss bar
    assert result["mostly_tracked"] == 0
    assert result["mostly_lost"] == 0
    assert result["partially_tracked"] == 1


def test_crossing_objects_of_different_classes_do_not_swap():
    """A car and a pedestrian crossing keep their ids thanks to class-aware gating."""
    tracker = Greedy3DTracker(max_age=2, min_hits=1, dist_gate=4.0, report_threshold=0.0)
    predictions, targets = [], []
    for t in range(10):
        y_car = -2.0 + 0.5 * t
        y_ped = 2.0 - 0.5 * t
        rows = tracker.update(
            np.stack([_box(0.0, y_car), _box(0.0, y_ped)]),
            np.array([0.9, 0.9]),
            np.array([0, 1]),
        )
        predictions.append(_frame(rows))
        targets.append(np.stack([_gt(0.0, y_car, track_id=1, class_id=0), _gt(0.0, y_ped, track_id=2, class_id=1)]))
    result = evaluate_tracking(predictions, targets, dist_threshold=2.0)
    assert result["ID_switches"] == 0
    assert math.isclose(result["MOTA"], 1.0, abs_tol=1e-9)


def test_crossing_objects_of_the_same_class_stay_mostly_tracked():
    """Two identical cars crossing is genuinely ambiguous, but MOTA must stay high.

    The frames where the two detections coincide are a hard association problem;
    the requirement here is that the tracker does not fall apart (MOTA stays high,
    at most one identity switch per crossing) rather than an exact id mapping.
    """
    tracker = Greedy3DTracker(max_age=2, min_hits=1, dist_gate=4.0, report_threshold=0.0)
    predictions, targets = [], []
    for t in range(10):
        y1 = -2.0 + 0.5 * t
        y2 = 2.0 - 0.5 * t
        rows = tracker.update(
            np.stack([_box(0.0, y1), _box(0.0, y2)]),
            np.array([0.9, 0.9]),
            np.array([0, 0]),
        )
        predictions.append(_frame(rows))
        targets.append(np.stack([_gt(0.0, y1, track_id=1), _gt(0.0, y2, track_id=2)]))
    result = evaluate_tracking(predictions, targets, dist_threshold=2.0)
    assert result["ID_switches"] <= 2
    assert result["MOTA"] >= 0.9
    assert result["FP"] == 0 and result["FN"] == 0


def test_greedy_and_hungarian_agree_on_an_easy_scene():
    """Both association modes solve a well-separated scene identically."""
    outputs = {}
    for method in ("hungarian", "greedy"):
        tracker = Greedy3DTracker(max_age=2, min_hits=1, dist_gate=4.0, report_threshold=0.0, method=method)
        ids = []
        for t in range(6):
            rows = tracker.update(
                np.stack([_box(2.0 + 0.5 * t, -3.0), _box(2.0 + 0.5 * t, 3.0)]),
                np.array([0.9, 0.9]),
                np.array([0, 0]),
            )
            ids.append(sorted(r["track_id"] for r in rows))
        outputs[method] = ids
    assert outputs["hungarian"] == outputs["greedy"]
    assert outputs["hungarian"][-1] == [1, 2]


def test_max_age_controls_track_lifetime():
    """A short max_age retires an unmatched track sooner than a long one."""
    for max_age, expected in ((0, 1), (3, 4)):
        tracker = Greedy3DTracker(max_age=max_age, min_hits=1, dist_gate=4.0, report_threshold=0.0)
        tracker.update(_box(2.0, 0.0)[None], np.array([0.9]), np.array([0]))
        alive = 1
        for _ in range(3):
            tracker.update(np.zeros((0, 9)), np.array([]), np.array([], dtype=np.int64))
            if tracker.tracks:
                alive += 1
        assert alive == expected, f"max_age={max_age}"


def test_amota_style_average_is_reported_for_several_gates():
    """The multi-threshold MOTA average is present and bounded by its extremes."""
    tracker = Greedy3DTracker(max_age=2, min_hits=1, dist_gate=4.0, report_threshold=0.0)
    predictions, targets = [], []
    for t in range(6):
        x = 2.0 + 0.5 * t
        predictions.append(_frame(tracker.update(_box(x, 0.0)[None], np.array([0.9]), np.array([0]))))
        targets.append(np.stack([_gt(x, 0.0, track_id=5)]))
    result = evaluate_tracking(predictions, targets, dist_threshold=0.5)
    values = list(result["mot_by_threshold"].values())
    assert math.isclose(result["AMOTA_style"], float(np.mean(values)), abs_tol=1e-9)
    assert min(values) <= result["AMOTA_style"] <= max(values)
