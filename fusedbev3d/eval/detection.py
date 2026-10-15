"""nuScenes-style 3D detection metrics, implemented from scratch.

Everything here - the rotated BEV IoU (Sutherland-Hodgman polygon clipping), the
centre-distance matching, the interpolated AP, the TP error breakdown
(ATE/ASE/AOE/AVE) and the NDS aggregate - is computed from raw boxes.  No metric
library is used.

Conventions
-----------
A *box* can be given either as the FusedBEV3D 11-column layout
``x y z l w h yaw vx vy class track_id`` or as a compact 5-column
``x y l w yaw``; both are accepted wherever a box tensor is expected.
Distances are Euclidean centre distances in metres (nuScenes matching rule).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

__all__ = [
    "boxes_to_bev",
    "box_corners",
    "polygon_area",
    "polygon_clip",
    "rotated_iou_bev",
    "iou_3d",
    "match_frame",
    "average_precision",
    "evaluate_detection",
    "print_table",
]

DIST_THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
PRIMARY_THRESHOLD = 2.0


# --------------------------------------------------------------------------------------
# Polygon / IoU helpers
# --------------------------------------------------------------------------------------


def boxes_to_bev(boxes: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return ``(N, 5)`` ``x y l w yaw`` from either supported box layout."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 5))
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] >= 7:
        return arr[:, [0, 1, 3, 4, 6]]
    if arr.shape[-1] == 5:
        return arr
    raise ValueError(f"unsupported box layout with {arr.shape[-1]} columns")


def box_corners(boxes: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return ``(N, 4, 2)`` counter-clockwise BEV corners of ``(N, 5)`` boxes."""
    bev = boxes_to_bev(boxes)
    n = bev.shape[0]
    if n == 0:
        return np.zeros((0, 4, 2))
    x, y, half_l, half_w, yaw = bev[:, 0], bev[:, 1], bev[:, 2] / 2, bev[:, 3] / 2, bev[:, 4]
    local = np.stack(
        [
            np.stack([-half_l, -half_w], axis=1),
            np.stack([half_l, -half_w], axis=1),
            np.stack([half_l, half_w], axis=1),
            np.stack([-half_l, half_w], axis=1),
        ],
        axis=1,
    )
    c, s = np.cos(yaw)[:, None, None], np.sin(yaw)[:, None, None]
    rot = np.concatenate(
        [np.concatenate([c, -s], axis=2), np.concatenate([s, c], axis=2)], axis=1
    )
    world = np.einsum("nij,nkj->nki", rot, local)
    return world + np.stack([x, y], axis=1)[:, None, :]


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of an arbitrary simple polygon ``(M, 2)``."""
    poly = np.asarray(poly, dtype=np.float64)
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _clip_edge(poly: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Clip a convex polygon against the half-plane left of edge ``a -> b``."""
    if poly.shape[0] == 0:
        return poly
    ex, ey = b[0] - a[0], b[1] - a[1]
    side = ex * (poly[:, 1] - a[1]) - ey * (poly[:, 0] - a[0])
    inside = side >= -1e-12
    if inside.all():
        return poly
    if not inside.any():
        return np.zeros((0, 2))
    out: list[np.ndarray] = []
    n = poly.shape[0]
    for i in range(n):
        cur, prev = poly[i], poly[i - 1]
        cur_in, prev_in = bool(inside[i]), bool(inside[i - 1])
        if cur_in != prev_in:
            t = side[i - 1] / (side[i - 1] - side[i])
            out.append(prev + t * (cur - prev))
        if cur_in:
            out.append(cur)
    return np.asarray(out, dtype=np.float64)


def polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of convex polygon ``subject`` by convex ``clip``.

    Args:
        subject: ``(M, 2)`` counter-clockwise convex polygon.
        clip: ``(K, 2)`` counter-clockwise convex polygon.

    Returns:
        The ``(J, 2)`` intersection polygon (empty when disjoint).
    """
    out = np.asarray(subject, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    for i in range(clip.shape[0]):
        if out.shape[0] == 0:
            break
        out = _clip_edge(out, clip[i], clip[(i + 1) % clip.shape[0]])
    return out


def rotated_iou_bev(boxes_a: torch.Tensor | np.ndarray, boxes_b: torch.Tensor | np.ndarray) -> np.ndarray:
    """Pairwise rotated BEV IoU between two box sets.

    Args:
        boxes_a: ``(N, 5)`` or ``(N, >=7)`` boxes.
        boxes_b: ``(M, 5)`` or ``(M, >=7)`` boxes.

    Returns:
        ``(N, M)`` float64 IoU matrix.
    """
    ca = box_corners(boxes_a)
    cb = box_corners(boxes_b)
    if ca.shape[0] == 0 or cb.shape[0] == 0:
        return np.zeros((ca.shape[0], cb.shape[0]))
    area_a = np.array([polygon_area(p) for p in ca])
    area_b = np.array([polygon_area(p) for p in cb])
    iou = np.zeros((ca.shape[0], cb.shape[0]), dtype=np.float64)
    for i in range(ca.shape[0]):
        for j in range(cb.shape[0]):
            inter = polygon_area(polygon_clip(ca[i], cb[j]))
            union = area_a[i] + area_b[j] - inter
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou


def iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Exact 3D IoU for two z-aligned prisms given in the 11-column layout.

    Because both boxes are vertical prisms the BEV intersection is constant along
    the overlapping height range, so ``V_inter = A_bev_inter * dz_overlap``.
    """
    a = boxes_to_bev(box_a[None, :])[0]
    b = boxes_to_bev(box_b[None, :])[0]
    inter_area = polygon_area(polygon_clip(box_corners(a[None, :])[0], box_corners(b[None, :])[0]))
    if inter_area <= 0:
        return 0.0
    za0, za1 = box_a[2] - box_a[5] / 2, box_a[2] + box_a[5] / 2
    zb0, zb1 = box_b[2] - box_b[5] / 2, box_b[2] + box_b[5] / 2
    dz = max(0.0, min(za1, zb1) - max(za0, zb0))
    inter = inter_area * dz
    vol_a = a[2] * a[3] * box_a[5]
    vol_b = b[2] * b[3] * box_b[5]
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------


def match_frame(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    dist_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy score-ordered centre-distance matching for a single frame.

    Predictions are visited in descending score order; each is matched to the
    closest still-unmatched ground-truth box of the same class within
    ``dist_threshold`` metres (the nuScenes rule).

    Args:
        pred_boxes: ``(P, >=5)`` predicted boxes.
        pred_scores: ``(P,)`` confidence scores.
        pred_classes: ``(P,)`` integer class ids.
        gt_boxes: ``(G, 11)`` ground-truth boxes.
        dist_threshold: Maximum centre distance in metres for a valid match.

    Returns:
        ``(matches, fp_indices, unmatched_gt)`` where ``matches`` is a list of
        ``(pred_idx, gt_idx, distance)``, ``fp_indices`` are unmatched prediction
        indices and ``unmatched_gt`` are unmatched ground-truth indices.
    """
    p = pred_boxes.shape[0]
    g = gt_boxes.shape[0]
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    if p == 0:
        return matches, [], list(range(g))
    if g > 0:
        d = np.linalg.norm(pred_boxes[:, None, :2] - gt_boxes[None, :, :2], axis=-1)
    else:
        d = np.zeros((p, 0))
    for i in np.argsort(-pred_scores, kind="stable"):
        if g == 0:
            continue
        candidates = [
            j
            for j in range(g)
            if j not in used_gt and pred_classes[i] == int(gt_boxes[j, 9]) and d[i, j] <= dist_threshold
        ]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: d[i, k])
        used_gt.add(j)
        matches.append((int(i), int(j), float(d[i, j])))
    fp = [i for i in range(p) if i not in {m[0] for m in matches}]
    return matches, fp, [j for j in range(g) if j not in used_gt]


def average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    """Area under the (right-max smoothed) precision-recall curve.

    Args:
        tp: ``(P,)`` 1/0 true-positive flags in *descending score* order.
        fp: ``(P,)`` 1/0 false-positive flags in the same order.
        num_gt: Total number of ground-truth boxes for this class.

    Returns:
        AP in ``[0, 1]``; ``nan`` when ``num_gt == 0``.
    """
    if num_gt == 0:
        return float("nan")
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    if tp.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    for i in range(precision.size - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    recall = np.concatenate([[0.0], recall])
    ap = float(np.sum(np.diff(recall) * precision))
    return float(np.clip(ap, 0.0, 1.0))


# --------------------------------------------------------------------------------------
# Full evaluation
# --------------------------------------------------------------------------------------


def _wrap_angle(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def evaluate_detection(
    predictions: Sequence[dict[str, Any]],
    targets: Sequence[torch.Tensor | np.ndarray],
    class_names: Sequence[str] = ("car", "pedestrian", "cyclist"),
    dist_thresholds: Iterable[float] = DIST_THRESHOLDS,
    primary_threshold: float = PRIMARY_THRESHOLD,
) -> dict[str, Any]:
    """Compute nuScenes-style detection metrics over a whole split.

    Args:
        predictions: Per-frame dicts with ``boxes (P,9)``, ``scores (P,)`` and
            ``classes (P,)``; boxes are ``x y z l w h yaw vx vy``.
        targets: Per-frame ``(G, 11)`` ground-truth box tensors (or arrays).
        class_names: Class names, indexed by class id.
        dist_thresholds: Matching thresholds in metres.
        primary_threshold: Threshold whose per-class AP defines ``mAP``/``NDS``.

    Returns:
        A metrics dictionary with ``mAP``, ``NDS``, ``mATE``/``mASE``/``mAOE``/
        ``mAVE``, per-class AP at every threshold, the average TP BEV IoU and
        prediction/ground-truth counts.
    """
    thresholds = list(dist_thresholds)
    num_classes = len(class_names)
    n_frames = len(predictions)
    primary_idx = (
        thresholds.index(primary_threshold) if primary_threshold in thresholds else len(thresholds) // 2
    )

    gt_boxes = [np.asarray(t, dtype=np.float64).reshape(-1, 11) for t in targets]
    gt_counts = np.zeros(num_classes, dtype=np.int64)
    for g in gt_boxes:
        if g.shape[0]:
            gt_counts += np.bincount(g[:, 9].astype(np.int64), minlength=num_classes)

    ap = np.full((num_classes, len(thresholds)), np.nan)
    tp_errors = {k: np.full(num_classes, np.nan) for k in ("ATE", "ASE", "AOE", "AVE")}
    tp_iou_sum = np.zeros(num_classes)
    tp_iou_count = np.zeros(num_classes)
    n_pred = 0

    for ti, thr in enumerate(thresholds):
        flat: list[tuple[float, int, np.ndarray, int, np.ndarray, int]] = []
        for f in range(n_frames):
            pred = predictions[f]
            boxes = np.asarray(pred["boxes"], dtype=np.float64).reshape(-1, 9)
            scores = np.asarray(pred["scores"], dtype=np.float64).reshape(-1)
            classes = np.asarray(pred["classes"], dtype=np.int64).reshape(-1)
            if ti == 0:
                n_pred += int(boxes.shape[0])
            for c in range(num_classes):
                keep = classes == c
                pc, ps = boxes[keep], scores[keep]
                matched: dict[int, int] = {}
                if pc.shape[0]:
                    matches, _, _ = match_frame(pc, ps, np.full(pc.shape[0], c), gt_boxes[f], thr)
                    matched = {m[0]: m[1] for m in matches}
                for i in range(pc.shape[0]):
                    flat.append(
                        (
                            float(ps[i]),
                            1 if i in matched else 0,
                            pc[i].copy(),
                            matched.get(i, -1),
                            gt_boxes[f],
                            c,
                        )
                    )

        for c in range(num_classes):
            ranked = sorted([e for e in flat if e[5] == c], key=lambda r: -r[0])
            flags = np.array([r[1] for r in ranked], dtype=np.float64)
            ap[c, ti] = average_precision(flags, 1.0 - flags, int(gt_counts[c]))
            if ti == primary_idx:
                acc: dict[str, list[float]] = {k: [] for k in tp_errors}
                for r in ranked:
                    if r[1] != 1 or r[3] < 0:
                        continue
                    pbox, gbox = r[2], r[4][r[3]]
                    dist = float(np.linalg.norm(pbox[:2] - gbox[:2]))
                    acc["ATE"].append(min(1.0, dist / 2.0))
                    acc["ASE"].append(1.0 - iou_3d(pbox, gbox))
                    acc["AOE"].append(min(1.0, abs(_wrap_angle(pbox[6] - gbox[6])) / (math.pi / 2.0)))
                    acc["AVE"].append(min(1.0, float(np.linalg.norm(pbox[7:9] - gbox[7:9])) / 2.0))
                    tp_iou_sum[c] += float(rotated_iou_bev(pbox[None, :], gbox[None, :])[0, 0])
                    tp_iou_count[c] += 1
                for k in tp_errors:
                    if acc[k]:
                        tp_errors[k][c] = float(np.mean(acc[k]))

    per_class_ap = ap[:, primary_idx]
    m_ap = float(np.nanmean(per_class_ap)) if np.isfinite(per_class_ap).any() else 0.0
    m_ate = 1.0 - float(np.nanmean(tp_errors["ATE"])) if np.isfinite(tp_errors["ATE"]).any() else 0.0
    m_ase = 1.0 - float(np.nanmean(tp_errors["ASE"])) if np.isfinite(tp_errors["ASE"]).any() else 0.0
    m_aoe = 1.0 - float(np.nanmean(tp_errors["AOE"])) if np.isfinite(tp_errors["AOE"]).any() else 0.0
    m_ave = 1.0 - float(np.nanmean(tp_errors["AVE"])) if np.isfinite(tp_errors["AVE"]).any() else 0.0
    # nuScenes-style NDS with the attribute-error term (AAE) omitted: weights are
    # 5 for mAP and 1 for each of the four TP accuracies, normalised by 9.
    nds = (5.0 * m_ap + m_ate + m_ase + m_aoe + m_ave) / 9.0

    result: dict[str, Any] = {
        "mAP": m_ap,
        "NDS": float(nds),
        "mATE": m_ate,
        "mASE": m_ase,
        "mAOE": m_aoe,
        "mAVE": m_ave,
        "primary_threshold": float(primary_threshold),
        "dist_thresholds": thresholds,
        "num_frames": n_frames,
        "num_predictions": int(n_pred),
        "num_gt": int(gt_counts.sum()),
        "gt_per_class": {n: int(gt_counts[i]) for i, n in enumerate(class_names)},
        "mean_tp_bev_iou": float(tp_iou_sum.sum() / max(tp_iou_count.sum(), 1)),
        "per_class": {},
    }
    for c, name in enumerate(class_names):
        result["per_class"][name] = {
            "AP": float(per_class_ap[c]) if np.isfinite(per_class_ap[c]) else 0.0,
            "AP_by_threshold": {
                float(t): (float(ap[c, i]) if np.isfinite(ap[c, i]) else 0.0)
                for i, t in enumerate(thresholds)
            },
            "num_gt": int(gt_counts[c]),
            "num_tp": int(tp_iou_count[c]),
            "ATE": float(tp_errors["ATE"][c]) if np.isfinite(tp_errors["ATE"][c]) else float("nan"),
            "ASE": float(tp_errors["ASE"][c]) if np.isfinite(tp_errors["ASE"][c]) else float("nan"),
            "AOE": float(tp_errors["AOE"][c]) if np.isfinite(tp_errors["AOE"][c]) else float("nan"),
            "AVE": float(tp_errors["AVE"][c]) if np.isfinite(tp_errors["AVE"][c]) else float("nan"),
        }
    return result


def print_table(result: dict[str, Any], title: str = "3D detection (nuScenes-style)") -> str:
    """Render a metrics dictionary as a fixed-width table and print it.

    Args:
        result: Output of :func:`evaluate_detection`.
        title: Header line.

    Returns:
        The rendered table (also printed to stdout).
    """
    lines = [f"== {title} =="]
    header = f"{'class':<12}{'AP':>8}{'AP@0.5':>9}{'AP@1.0':>9}{'AP@2.0':>9}{'AP@4.0':>9}{'ATE':>8}{'AOE':>8}{'nGT':>7}{'nTP':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in result["per_class"].items():
        apb = m["AP_by_threshold"]
        lines.append(
            f"{name:<12}{m['AP']:>8.4f}"
            f"{apb.get(0.5, float('nan')):>9.4f}{apb.get(1.0, float('nan')):>9.4f}"
            f"{apb.get(2.0, float('nan')):>9.4f}{apb.get(4.0, float('nan')):>9.4f}"
            f"{m['ATE']:>8.4f}{m['AOE']:>8.4f}{m['num_gt']:>7}{m['num_tp']:>6}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"mAP={result['mAP']:.4f}  NDS={result['NDS']:.4f}  (threshold {result['primary_threshold']} m)"
        f"  mATE={result['mATE']:.4f}  mASE={result['mASE']:.4f}"
        f"  mAOE={result['mAOE']:.4f}  mAVE={result['mAVE']:.4f}"
    )
    lines.append(
        f"frames={result['num_frames']}  gt={result['num_gt']}  preds={result['num_predictions']}"
        f"  mean TP BEV IoU={result['mean_tp_bev_iou']:.4f}"
    )
    table = "\n".join(lines)
    print(table)
    return table
