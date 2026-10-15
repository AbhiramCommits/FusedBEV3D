"""Occupancy evaluation: per-class IoU, mIoU and occupied-voxel precision/recall."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["occupancy_confusion", "evaluate_occupancy", "print_occupancy_table"]

CLASS_NAMES: tuple[str, ...] = ("free", "occupied", "unknown")
FREE, OCCUPIED, UNKNOWN = 0, 1, 2


def occupancy_confusion(
    prediction: np.ndarray, target: np.ndarray, num_classes: int = 3
) -> np.ndarray:
    """Return the ``(num_classes, num_classes)`` confusion matrix ``[pred, gt]``.

    Args:
        prediction: Integer label array of any shape.
        target: Integer label array of the same shape.
        num_classes: Number of classes.

    Returns:
        Counts indexed ``[predicted, ground_truth]``.
    """
    pred = np.asarray(prediction, dtype=np.int64).ravel()
    tgt = np.asarray(target, dtype=np.int64).ravel()
    idx = pred * num_classes + tgt
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def evaluate_occupancy(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int = 3,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """Compute per-class IoU, mIoU and occupied-voxel precision/recall.

    The IoU of class ``c`` is ``TP_c / (TP_c + FP_c + FN_c)`` over the whole
    evaluated volume.  ``unknown`` is scored as a real class here (the generator
    labels it from lidar visibility), and ``mIoU_observed`` restricts the average
    to the ``free`` and ``occupied`` classes, which is the more common protocol.

    Args:
        prediction: ``(N, Z, H, W)`` predicted labels in ``{0, 1, 2}``.
        target: ``(N, Z, H, W)`` ground-truth labels.
        num_classes: Number of classes.
        class_names: Class names in label order.

    Returns:
        A metrics dictionary with ``mIoU``, ``mIoU_observed``, per-class IoU and
        precision/recall, ``occupied_precision``/``occupied_recall``,
        per-class accuracy and the confusion matrix.
    """
    conf = occupancy_confusion(prediction, target, num_classes)
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-12)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    total = conf.sum()
    per_class = {}
    for c, name in enumerate(class_names):
        per_class[name] = {
            "iou": float(iou[c]),
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "gt_voxels": int(conf[:, c].sum()),
            "pred_voxels": int(conf[c, :].sum()),
        }
    result: dict[str, Any] = {
        "mIoU": float(np.mean(iou)),
        "mIoU_observed": float(np.mean(iou[:2])),
        "per_class_iou": {name: float(iou[c]) for c, name in enumerate(class_names)},
        "per_class": per_class,
        "occupied_precision": float(precision[OCCUPIED]),
        "occupied_recall": float(recall[OCCUPIED]),
        "occupied_iou": float(iou[OCCUPIED]),
        "accuracy": float(tp.sum() / max(total, 1)),
        "num_voxels": int(total),
        "class_fraction": {
            name: float(conf[:, c].sum() / max(total, 1)) for c, name in enumerate(class_names)
        },
        "confusion": conf.tolist(),
        "class_names": list(class_names),
    }
    return result


def print_occupancy_table(result: dict[str, Any], title: str = "occupancy") -> str:
    """Render an occupancy metrics dictionary and print it.

    Args:
        result: Output of :func:`evaluate_occupancy`.
        title: Header line.

    Returns:
        The rendered table (also printed).
    """
    lines = [f"== {title} =="]
    header = f"{'class':<10}{'IoU':>9}{'prec':>9}{'recall':>9}{'gt_vox':>12}{'pred_vox':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in result["per_class"].items():
        lines.append(
            f"{name:<10}{m['iou']:>9.4f}{m['precision']:>9.4f}{m['recall']:>9.4f}"
            f"{m['gt_voxels']:>12}{m['pred_voxels']:>12}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"mIoU={result['mIoU']:.4f}  mIoU(free+occupied)={result['mIoU_observed']:.4f}  "
        f"occupied precision={result['occupied_precision']:.4f} "
        f"recall={result['occupied_recall']:.4f}  overall accuracy={result['accuracy']:.4f}"
    )
    table = "\n".join(lines)
    print(table)
    return table
