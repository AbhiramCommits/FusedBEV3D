"""Shared inference helpers used by ``scripts/eval.py``, ``scripts/track.py`` and the ablation.

Keeping the prediction loops in the package (rather than in the CLI scripts) means
the headline detection numbers, the tracking numbers and every ablation row are
produced by exactly the same code path.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fusedbev3d.config import Config
from fusedbev3d.data.registry import build_dataset
from fusedbev3d.data.synth import collate_fn
from fusedbev3d.eval.detection import evaluate_detection
from fusedbev3d.eval.occupancy import evaluate_occupancy
from fusedbev3d.eval.tracking import evaluate_tracking
from fusedbev3d.models.fusedbev3d import normalize_modalities
from fusedbev3d.tracking.tracker import Greedy3DTracker

__all__ = ["make_loader", "predict_split", "track_split", "evaluate_model"]

DEFAULT_TRACKER: dict[str, Any] = {
    "max_age": 2,
    "min_hits": 1,
    "dist_gate": 4.0,
    "method": "hungarian",
    "report_threshold": 0.4,
}


def make_loader(cfg: Config, split: str, scenes: int | None = None, batch_size: int | None = None) -> DataLoader:
    """Build a deterministic, ordered evaluation loader for ``split``."""
    ds = build_dataset(cfg.data.name, split, cfg, num_scenes=scenes)
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    cfg: Config,
    loader: DataLoader,
    modalities: Sequence[str] | str | None = None,
    score_threshold: float = 0.05,
    topk: int = 100,
    device: torch.device | str = "cpu",
    tracker: Greedy3DTracker | None = None,
    track_score_threshold: float = 0.1,
) -> dict[str, Any]:
    """Run the detector (optionally with a tracker) over an ordered split.

    Args:
        model: A loaded :class:`~fusedbev3d.models.fusedbev3d.FusedBEV3D`.
        cfg: Configuration of the checkpoint.
        loader: Ordered dataloader (use :func:`make_loader`).
        modalities: Optional modality override for inference-time dropout.
        score_threshold: Minimum decoded score for a reported detection.
        topk: Heatmap peaks decoded per frame.
        device: Torch device.
        tracker: Optional tracker; when given, ``track_ids`` are added to the
            per-frame predictions and the tracker is reset on every new scene.
        track_score_threshold: Detections below this score are not fed to the tracker.

    Returns:
        A dictionary with ``predictions``, ``targets``, ``occupancy_pred``,
        ``occupancy_true``, ``ms_per_frame`` and the effective ``modalities``.
    """
    device = torch.device(device)
    model.eval()
    mods = normalize_modalities(modalities) if modalities is not None else None
    predictions: list[dict[str, Any]] = []
    targets: list[np.ndarray] = []
    occ_pred: list[np.ndarray] = []
    occ_true: list[np.ndarray] = []
    current_scene = -1
    frames = 0
    start = time.time()
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch, modalities=mods, decode=True, topk=topk)
        det = out["detections"]
        keep = det["keep"] & (det["scores"] >= score_threshold)
        for i in range(det["boxes"].shape[0]):
            mask = keep[i]
            if tracker is not None:
                scene = int(batch["scene_id"][i])
                if scene != current_scene:
                    tracker.reset()
                    current_scene = scene
                tm = mask & (det["scores"][i] >= track_score_threshold)
                tracked = tracker.update(
                    det["boxes"][i][tm].cpu().numpy(),
                    det["scores"][i][tm].cpu().numpy(),
                    det["classes"][i][tm].cpu().numpy(),
                )
                predictions.append(
                    {
                        "boxes": np.stack([t["box"] for t in tracked]) if tracked else np.zeros((0, 9)),
                        "scores": np.array([t["score"] for t in tracked], dtype=np.float64),
                        "classes": np.array([t["class_id"] for t in tracked], dtype=np.int64),
                        "track_ids": np.array([t["track_id"] for t in tracked], dtype=np.int64),
                    }
                )
            else:
                predictions.append(
                    {
                        "boxes": det["boxes"][i][mask].cpu().numpy(),
                        "scores": det["scores"][i][mask].cpu().numpy(),
                        "classes": det["classes"][i][mask].cpu().numpy(),
                    }
                )
            targets.append(batch["boxes"][i].cpu().numpy())
            frames += 1
        occ_pred.append(model.occupancy_head.predict(out["occupancy"]).cpu().numpy())
        occ_true.append(batch["occupancy"].cpu().numpy())
    ms = (time.time() - start) * 1000.0 / max(frames, 1)
    return {
        "predictions": predictions,
        "targets": targets,
        "occupancy_pred": np.concatenate(occ_pred, axis=0),
        "occupancy_true": np.concatenate(occ_true, axis=0),
        "ms_per_frame": ms,
        "modalities": list(mods) if mods else list(model.modalities),
    }


def track_split(
    model: torch.nn.Module,
    cfg: Config,
    loader: DataLoader,
    tracker_kwargs: dict[str, Any] | None = None,
    score_threshold: float = 0.1,
    topk: int = 100,
    device: torch.device | str = "cpu",
    eval_gate: float = 2.0,
    modalities: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Track a split and return CLEAR-MOT metrics plus latency.

    Args:
        model: Loaded model.
        cfg: Configuration of the checkpoint.
        loader: Ordered dataloader.
        tracker_kwargs: Tracker overrides.
        score_threshold: Detection threshold fed to the tracker.
        topk: Heatmap peaks decoded per frame.
        device: Torch device.
        eval_gate: CLEAR-MOT matching gate in metres.
        modalities: Optional modality override, forwarded to the detector.

    Returns:
        The :func:`fusedbev3d.eval.tracking.evaluate_tracking` dictionary plus
        ``ms_per_frame``.
    """
    tracker = Greedy3DTracker(**(tracker_kwargs or DEFAULT_TRACKER))
    out = predict_split(
        model,
        cfg,
        loader,
        modalities=modalities,
        score_threshold=score_threshold,
        topk=topk,
        device=device,
        tracker=tracker,
        track_score_threshold=score_threshold,
    )
    metrics = evaluate_tracking(out["predictions"], out["targets"], dist_threshold=eval_gate)
    metrics["ms_per_frame"] = out["ms_per_frame"]
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    cfg: Config,
    split: str = "val",
    scenes: int | None = None,
    device: torch.device | str = "cpu",
    score_threshold: float = 0.05,
    track_score_threshold: float = 0.1,
    tracker_kwargs: dict[str, Any] | None = None,
    modalities: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Evaluate detection, occupancy and tracking for one model variant.

    Args:
        model: Loaded model.
        cfg: Configuration of the checkpoint.
        split: Split to evaluate.
        scenes: Optional scene-count override.
        device: Torch device.
        score_threshold: Detection reporting threshold for the detection metrics.
        track_score_threshold: Detection threshold fed into the tracker.
        tracker_kwargs: Tracker overrides.
        modalities: Optional inference-time modality override.

    Returns:
        ``{"detection", "occupancy", "tracking", "ms_per_frame", "modalities"}``.
    """
    loader = make_loader(cfg, split, scenes)
    det_out = predict_split(
        model,
        cfg,
        loader,
        modalities=modalities,
        score_threshold=score_threshold,
        topk=100,
        device=device,
    )
    detection = evaluate_detection(
        det_out["predictions"], det_out["targets"], cfg.model.class_names
    )
    occupancy = evaluate_occupancy(det_out["occupancy_pred"], det_out["occupancy_true"])
    track_out = track_split(
        model,
        cfg,
        loader,
        tracker_kwargs=tracker_kwargs,
        score_threshold=track_score_threshold,
        device=device,
        eval_gate=2.0,
        modalities=modalities,
    )
    return {
        "detection": detection,
        "occupancy": occupancy,
        "tracking": track_out,
        "predictions": det_out["predictions"],
        "targets": det_out["targets"],
        "ms_per_frame": det_out["ms_per_frame"],
        "tracking_ms_per_frame": track_out["ms_per_frame"],
        "modalities": det_out["modalities"],
    }
