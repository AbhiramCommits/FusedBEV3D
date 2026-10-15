"""CLEAR-MOT tracking metrics computed from scratch, plus a multi-threshold variant."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["evaluate_tracking", "print_tracking_table", "DEFAULT_THRESHOLDS"]

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


def _associate_frame(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    dist_threshold: float,
) -> list[tuple[int, int, float]]:
    """Hungarian (or greedy fallback) association for one frame.

    Args:
        pred_boxes: ``(P, 9)`` predicted boxes.
        pred_classes: ``(P,)`` predicted class ids.
        gt_boxes: ``(G, 11)`` ground-truth boxes (class in column 9).
        dist_threshold: Centre-distance gate in metres.

    Returns:
        A list of ``(pred_idx, gt_idx, distance)`` triples.
    """
    p, g = pred_boxes.shape[0], gt_boxes.shape[0]
    if p == 0 or g == 0:
        return []
    dist = np.linalg.norm(pred_boxes[:, None, :2] - gt_boxes[None, :, :2], axis=-1)
    same = pred_classes[:, None] == gt_boxes[None, :, 9].astype(np.int64)
    cost = np.where(same & (dist <= dist_threshold), dist, 1e6)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c), float(dist[r, c])) for r, c in zip(rows, cols, strict=False) if cost[r, c] < 1e5]
    except ImportError:  # pragma: no cover
        pairs: list[tuple[int, int, float]] = []
        used_p: set[int] = set()
        used_g: set[int] = set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            r, c = int(r), int(c)
            if cost[r, c] >= 1e5 or r in used_p or c in used_g:
                continue
            used_p.add(r)
            used_g.add(c)
            pairs.append((r, c, float(dist[r, c])))
        return pairs


def evaluate_tracking(
    predictions: Sequence[dict[str, Any]],
    targets: Sequence[np.ndarray],
    dist_threshold: float = 2.0,
    extra_thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Compute CLEAR-MOT metrics plus a multi-threshold MOTA average.

    Args:
        predictions: Per-frame dicts with ``boxes (P,9)``, ``scores (P,)``,
            ``classes (P,)`` and ``track_ids (P,)``.
        targets: Per-frame ``(G, 11)`` ground truth, track id in column 10.
        dist_threshold: Primary association gate in metres.
        extra_thresholds: Thresholds used for the AMOTA-style average.

    Returns:
        A metrics dict with MOTA, MOTP, ID switches, FP, FN, mostly-tracked /
        mostly-lost counts, per-threshold MOTA and the AMOTA-style average.
    """
    per_threshold: dict[float, dict[str, Any]] = {}
    all_thresholds = sorted({float(dist_threshold), *[float(t) for t in extra_thresholds]})
    for thr in all_thresholds:
        per_threshold[thr] = _clear_mot(predictions, targets, thr)

    primary = per_threshold[float(dist_threshold)]
    result = dict(primary)
    result["dist_threshold"] = float(dist_threshold)
    result["mot_by_threshold"] = {t: per_threshold[t]["MOTA"] for t in all_thresholds}
    result["AMOTA_style"] = float(np.mean([per_threshold[t]["MOTA"] for t in all_thresholds]))
    result["amota_thresholds"] = all_thresholds
    return result


def _clear_mot(
    predictions: Sequence[dict[str, Any]],
    targets: Sequence[np.ndarray],
    dist_threshold: float,
) -> dict[str, Any]:
    """Core CLEAR-MOT accumulation for a single distance threshold."""
    id_switches = 0
    false_pos = 0
    false_neg = 0
    matches = 0
    motp_sum = 0.0
    gt_count = 0
    last_track_of_gt: dict[int, int] = {}
    gt_frames: dict[int, int] = {}
    gt_matched: dict[int, int] = {}
    n_pred = 0

    for pred, gt in zip(predictions, targets, strict=False):
        boxes = np.asarray(pred["boxes"], dtype=np.float64).reshape(-1, 9)
        classes = np.asarray(pred["classes"], dtype=np.int64).reshape(-1)
        track_ids = np.asarray(pred.get("track_ids", np.arange(boxes.shape[0])), dtype=np.int64).reshape(-1)
        gt = np.asarray(gt, dtype=np.float64).reshape(-1, 11)
        n_pred += boxes.shape[0]
        gt_count += gt.shape[0]
        for row in gt:
            gid = int(row[10])
            gt_frames[gid] = gt_frames.get(gid, 0) + 1
        pairs = _associate_frame(boxes, classes, gt, dist_threshold)
        matched_p = {p[0] for p in pairs}
        matched_g = {p[1] for p in pairs}
        for pi, gi, dist in pairs:
            matches += 1
            motp_sum += dist
            gid = int(gt[gi, 10])
            gt_matched[gid] = gt_matched.get(gid, 0) + 1
            tid = int(track_ids[pi]) if pi < track_ids.shape[0] else pi
            # CLEAR MOT counts an ID switch when a ground-truth object's assigned
            # track id changes relative to its last association (even across a gap).
            if gid in last_track_of_gt and last_track_of_gt[gid] != tid:
                id_switches += 1
            last_track_of_gt[gid] = tid
        false_pos += boxes.shape[0] - len(matched_p)
        false_neg += gt.shape[0] - len(matched_g)

    mota = 1.0 - (false_neg + false_pos + id_switches) / max(gt_count, 1)
    motp = motp_sum / max(matches, 1)
    mt = sum(1 for gid, n in gt_frames.items() if gt_matched.get(gid, 0) >= 0.8 * n)
    ml = sum(1 for gid, n in gt_frames.items() if gt_matched.get(gid, 0) <= 0.2 * n)
    return {
        "MOTA": float(mota),
        "MOTP": float(motp),
        "ID_switches": int(id_switches),
        "FP": int(false_pos),
        "FN": int(false_neg),
        "TP": int(matches),
        "num_gt": int(gt_count),
        "num_predictions": int(n_pred),
        "num_gt_tracks": len(gt_frames),
        "mostly_tracked": int(mt),
        "mostly_lost": int(ml),
        "partially_tracked": int(len(gt_frames) - mt - ml),
        "recall": float(matches / max(gt_count, 1)),
        "precision": float(matches / max(n_pred, 1)),
    }


def print_tracking_table(result: dict[str, Any], title: str = "tracking (CLEAR MOT)") -> str:
    """Render a tracking metrics dictionary and print it.

    Args:
        result: Output of :func:`evaluate_tracking`.
        title: Header line.

    Returns:
        The rendered table (also printed).
    """
    lines = [f"== {title} =="]
    lines.append(f"gate            : {result['dist_threshold']} m")
    lines.append(
        f"MOTA            : {result['MOTA']:.4f}   MOTP: {result['MOTP']:.4f} m   "
        f"ID switches: {result['ID_switches']}"
    )
    lines.append(
        f"TP/FP/FN        : {result['TP']}/{result['FP']}/{result['FN']}   "
        f"recall {result['recall']:.4f}  precision {result['precision']:.4f}"
    )
    lines.append(
        f"GT tracks       : {result['num_gt_tracks']}   MT {result['mostly_tracked']}  "
        f"PT {result['partially_tracked']}  ML {result['mostly_lost']}"
    )
    mot = ", ".join(f"{t:.1f}:{v:.4f}" for t, v in result["mot_by_threshold"].items())
    lines.append(f"MOTA by gate    : {mot}")
    lines.append(f"AMOTA-style avg : {result['AMOTA_style']:.4f}")
    table = "\n".join(lines)
    print(table)
    return table
