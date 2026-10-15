"""Detection metrics: AP bookkeeping and the NDS-style aggregate."""

from __future__ import annotations

import math

import numpy as np
import torch

from fusedbev3d.eval.detection import (
    average_precision,
    evaluate_detection,
    match_frame,
    precision_recall_curves,
)

GT_A = np.array([3.0, 2.0, 1.0, 4.0, 1.8, 1.6, 0.2, 1.0, 0.0, 0, 11.0])
GT_B = np.array([-4.0, -3.0, 0.9, 0.7, 0.7, 1.75, 1.1, 0.0, 0.0, 1, 12.0])
FAR = np.array([60.0, 60.0, 1.0, 4.0, 1.8, 1.6, 0.0, 0.0, 0.0, 0, 0.0])


def _frame(boxes, scores, classes):
    """Build one prediction dict from python lists."""
    return {
        "boxes": torch.tensor(np.asarray(boxes, dtype=np.float64).reshape(-1, 9)),
        "scores": torch.tensor(np.asarray(scores, dtype=np.float64)),
        "classes": torch.tensor(np.asarray(classes, dtype=np.int64)),
    }


def test_perfect_predictions_give_map_one():
    """Exact predictions give AP 1.0 for the class and mAP/NDS 1.0 overall."""
    preds = [_frame(np.concatenate([GT_A[:9][None], GT_B[:9][None]]), [0.9, 0.8], [0, 1])]
    targets = [torch.tensor(np.stack([GT_A, GT_B]))]
    result = evaluate_detection(preds, targets, ("car", "pedestrian", "cyclist"))
    assert math.isclose(result["mAP"], 1.0, abs_tol=1e-9)
    assert math.isclose(result["NDS"], 1.0, abs_tol=1e-9)
    assert result["per_class"]["car"]["AP"] == 1.0
    assert result["per_class"]["pedestrian"]["AP"] == 1.0
    assert math.isclose(result["per_class"]["car"]["ATE"], 0.0, abs_tol=1e-9)


def test_no_predictions_give_zero_map():
    """Without any prediction every class scores AP 0.0."""
    preds = [_frame(np.zeros((0, 9)), [], [])]
    targets = [torch.tensor(GT_A[None, :])]
    result = evaluate_detection(preds, targets)
    assert result["mAP"] == 0.0
    assert result["per_class"]["car"]["AP"] == 0.0
    assert result["per_class"]["car"]["num_gt"] == 1


def test_hand_computed_average_precision():
    """A crafted TP/FP ordering reproduces the hand-computed AP.

    Two ground truths, three predictions in descending score order:
    TP, FP, TP.  Cumulative precision is ``1, 0.5, 2/3`` at recall
    ``0.5, 0.5, 1.0``.  Right-max smoothing lifts the middle point to ``2/3``,
    so ``AP = 0.5 * 1 + 0.5 * 2/3 = 5/6``.
    """
    tp = np.array([1.0, 0.0, 1.0])
    fp = np.array([0.0, 1.0, 0.0])
    assert math.isclose(average_precision(tp, fp, 2), 5.0 / 6.0, abs_tol=1e-9)
    assert average_precision(tp, fp, 0) != average_precision(tp, fp, 0)  # nan when no GT


def test_hand_computed_ap_through_the_full_evaluator():
    """The same 5/6 case, driven end-to-end through ``evaluate_detection``."""
    gt1 = GT_A.copy()
    gt2 = GT_A.copy()
    gt2[0] = 9.0
    gt2[10] = 11.0
    targets = [torch.tensor(np.stack([gt1, gt2]))]
    hit2 = gt2[:9].copy()
    hit2[0] += 0.2
    preds = [_frame(np.stack([gt1[:9], FAR[:9], hit2]), [0.9, 0.8, 0.7], [0, 0, 0])]
    result = evaluate_detection(preds, targets, dist_thresholds=(2.0,))
    assert result["per_class"]["car"]["num_gt"] == 2
    assert result["per_class"]["car"]["num_tp"] == 2
    assert math.isclose(result["per_class"]["car"]["AP"], 5.0 / 6.0, abs_tol=1e-6)
    assert math.isclose(result["mAP"], 5.0 / 6.0, abs_tol=1e-6)


def test_matching_respects_class_and_distance_gate():
    """Matching never crosses classes and never exceeds the distance gate."""
    pred_boxes = np.array([GT_A[:9], GT_B[:9]])
    pred_cls = np.array([1, 0])  # deliberately swapped classes
    gt = np.stack([GT_A, GT_B])
    pairs, fp, fn = match_frame(pred_boxes, np.array([0.9, 0.8]), pred_cls, gt, 1.0)
    assert len(pairs) == 0, "swapped classes must not match"
    assert sorted(fp) == [0, 1]
    assert sorted(fn) == [0, 1]

    pairs, fp, fn = match_frame(pred_boxes, np.array([0.9, 0.8]), np.array([0, 1]), gt, 1.0)
    assert {p[0] for p in pairs} == {0, 1}
    assert fp == [] and fn == []


def test_distance_gate_excludes_far_predictions():
    """A prediction beyond the gate is an FP rather than a TP."""
    preds = [_frame(FAR[:9][None], [0.95], [0])]
    targets = [torch.tensor(GT_A[None, :])]
    tight = evaluate_detection(preds, targets, dist_thresholds=(2.0,), primary_threshold=2.0)
    assert tight["per_class"]["car"]["AP"] == 0.0
    assert tight["mean_tp_bev_iou"] == 0.0


def test_duplicate_predictions_cost_precision():
    """Extra predictions of the same object are false positives."""
    dup = np.stack([GT_A[:9], GT_A[:9] + np.array([0.1, 0.0, 0, 0, 0, 0, 0, 0, 0])])
    preds = [_frame(dup, [0.9, 0.85], [0, 0])]
    targets = [torch.tensor(GT_A[None, :])]
    result = evaluate_detection(preds, targets, dist_thresholds=(2.0,), primary_threshold=2.0)
    assert math.isclose(result["per_class"]["car"]["AP"], 1.0, abs_tol=1e-9)
    assert result["num_predictions"] == 2
    assert result["per_class"]["car"]["num_tp"] == 1


def test_nds_is_bounded_and_monotone_in_map():
    """The NDS-style aggregate stays in [0, 1] and grows with mAP."""
    perfect = evaluate_detection(
        [_frame(GT_A[:9][None], [0.9], [0])], [torch.tensor(GT_A[None, :])]
    )
    empty = evaluate_detection(
        [_frame(np.zeros((0, 9)), [], [])], [torch.tensor(GT_A[None, :])]
    )
    assert 0.0 <= empty["NDS"] <= 1.0
    assert empty["NDS"] < perfect["NDS"] <= 1.0


def test_precision_recall_curves_endpoints():
    """PR curves start at full precision and are monotone in recall."""
    preds = [_frame(np.stack([GT_A[:9], FAR[:9]]), [0.9, 0.8], [0, 0])]
    targets = [torch.tensor(GT_A[None, :])]
    curves = precision_recall_curves(preds, targets, dist_threshold=2.0)
    car = curves["car"]
    assert car["recall"][0] == 0.0 and car["precision"][0] == 1.0
    assert all(b >= a - 1e-9 for a, b in zip(car["recall"], car["recall"][1:], strict=False))
    assert math.isclose(car["ap"], 1.0, abs_tol=1e-9)


def test_per_class_ap_by_threshold_is_reported_for_every_gate():
    """AP is reported for each of the four nuScenes distance thresholds."""
    preds = [_frame(GT_A[:9][None], [0.9], [0])]
    targets = [torch.tensor(GT_A[None, :])]
    result = evaluate_detection(preds, targets)
    by_thr = result["per_class"]["car"]["AP_by_threshold"]
    assert sorted(by_thr) == [0.5, 1.0, 2.0, 4.0]
    assert all(v == 1.0 for v in by_thr.values())
