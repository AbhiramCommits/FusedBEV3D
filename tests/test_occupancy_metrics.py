"""Occupancy metrics: per-class IoU against hand-built volumes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fusedbev3d.eval.occupancy import evaluate_occupancy, occupancy_confusion


def test_perfect_prediction_gives_iou_one():
    """A prediction identical to the label scores mIoU 1.0."""
    vol = np.array([[0, 1], [2, 1]], dtype=np.int64)[None]
    result = evaluate_occupancy(vol, vol)
    assert math.isclose(result["mIoU"], 1.0, abs_tol=1e-9)
    assert math.isclose(result["occupied_iou"], 1.0, abs_tol=1e-9)
    assert math.isclose(result["accuracy"], 1.0, abs_tol=1e-9)
    assert math.isclose(result["occupied_precision"], 1.0, abs_tol=1e-9)


def test_hand_computed_ious():
    """A 2x2 volume with one mistaken voxel yields the hand-computed per-class IoU.

    ``target = [[0, 0], [1, 2]]`` and ``pred = [[0, 1], [1, 2]]``: the single
    mistake is voxel ``(0, 1)``, which is truly free but predicted occupied.  That
    gives free one TP and one FN (IoU ``1 / 2``) and occupied one TP plus one FP
    (IoU ``1 / 2``); unknown is perfect (IoU ``1``).  Hence
    ``mIoU = (0.5 + 0.5 + 1) / 3 = 2/3`` and ``mIoU_observed = 0.5``.
    """
    target = np.array([[0, 0], [1, 2]], dtype=np.int64)[None]
    pred = np.array([[0, 1], [1, 2]], dtype=np.int64)[None]
    result = evaluate_occupancy(pred, target)
    assert math.isclose(result["per_class_iou"]["free"], 0.5, abs_tol=1e-9)
    assert math.isclose(result["per_class_iou"]["occupied"], 0.5, abs_tol=1e-9)
    assert math.isclose(result["per_class_iou"]["unknown"], 1.0, abs_tol=1e-9)
    assert math.isclose(result["mIoU"], 2.0 / 3.0, abs_tol=1e-9)
    assert math.isclose(result["mIoU_observed"], 0.5, abs_tol=1e-9)


def test_all_occupied_predicted_gives_known_numbers():
    """Predicting every voxel occupied is a well-defined, poor result."""
    target = np.array([[0, 0], [1, 2]], dtype=np.int64)[None]
    pred = np.ones_like(target)
    result = evaluate_occupancy(pred, target)
    # occupied: 1 TP and 3 FPs (every voxel is predicted occupied) -> IoU 1/4
    assert math.isclose(result["occupied_iou"], 0.25, abs_tol=1e-9)
    assert math.isclose(result["occupied_precision"], 0.25, abs_tol=1e-9)
    assert math.isclose(result["occupied_recall"], 1.0, abs_tol=1e-9)
    # free and unknown are never predicted -> IoU 0
    assert result["per_class_iou"]["free"] == 0.0
    assert result["per_class_iou"]["unknown"] == 0.0
    assert math.isclose(result["mIoU"], 0.25 / 3.0, abs_tol=1e-9)


def test_class_absent_from_both_is_zero_not_error():
    """A class with no ground truth and no prediction scores IoU 0 without dividing by zero."""
    target = np.zeros((1, 2, 2), dtype=np.int64)
    pred = np.zeros((1, 2, 2), dtype=np.int64)
    result = evaluate_occupancy(pred, target)
    assert result["per_class_iou"]["occupied"] == 0.0
    assert result["per_class_iou"]["unknown"] == 0.0
    assert result["per_class_iou"]["free"] == 1.0
    assert math.isfinite(result["mIoU"])


def test_confusion_matrix_layout_is_prediction_by_ground_truth():
    """The confusion matrix is indexed ``[predicted, ground truth]``."""
    target = np.array([0, 1, 2, 2])
    pred = np.array([0, 1, 1, 2])
    conf = occupancy_confusion(pred, target)
    assert conf.shape == (3, 3)
    assert conf[0, 0] == 1  # free predicted free
    assert conf[1, 1] == 1  # occupied predicted occupied
    assert conf[1, 2] == 1  # occupied predicted, but actually unknown
    assert conf[2, 2] == 1
    assert conf.sum() == 4


def test_evaluate_occupancy_reports_voxel_counts_and_fractions():
    """Voxel counts and class fractions are consistent with the inputs."""
    target = np.array([[0, 0, 1, 2]], dtype=np.int64)[None]
    result = evaluate_occupancy(target, target)
    assert result["num_voxels"] == 4
    assert math.isclose(sum(result["class_fraction"].values()), 1.0, abs_tol=1e-9)
    assert result["per_class"]["occupied"]["gt_voxels"] == 1


def test_multi_batch_volumes_are_flattened():
    """IoU aggregates over the whole batch rather than averaging per-sample IoUs."""
    target = np.array([[[0]], [[1]]], dtype=np.int64)
    pred = np.array([[[0]], [[0]]], dtype=np.int64)
    result = evaluate_occupancy(pred, target)
    assert result["num_voxels"] == 2
    assert math.isclose(result["per_class_iou"]["free"], 0.5, abs_tol=1e-9)
    assert math.isclose(result["per_class_iou"]["occupied"], 0.0, abs_tol=1e-9)


@pytest.mark.parametrize("shape", [(1, 2, 2), (3, 2, 2)])
def test_output_shapes(shape):
    """The evaluator accepts the 3D label tensors used by the dataset."""
    rng = np.random.default_rng(0)
    target = rng.integers(0, 3, size=shape)
    result = evaluate_occupancy(target, target)
    assert math.isclose(result["mIoU"], 1.0, abs_tol=1e-9)
