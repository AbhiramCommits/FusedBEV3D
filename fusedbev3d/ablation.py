"""Modality ablation: camera-only vs lidar-only vs fused, plus inference-time dropout.

:func:`run_ablation` trains every variant under an identical seed, schedule and step
budget (the *only* difference is which encoders produce non-zero BEV features),
evaluates detection / occupancy / tracking for each, and finally reloads the fused
checkpoint and re-evaluates it with a single modality zeroed at inference time.
That last block is the robustness check: it isolates the effect of losing a sensor
without a second training run.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fusedbev3d.config import Config
from fusedbev3d.eval.detection import precision_recall_curves
from fusedbev3d.inference import evaluate_model
from fusedbev3d.train import load_checkpoint, pick_device, train

__all__ = ["VARIANTS", "run_ablation", "row_from_results", "write_tables"]

#: Ordered variant name -> enabled modalities.
VARIANTS: dict[str, tuple[str, ...]] = {
    "camera_only": ("camera",),
    "lidar_only": ("lidar",),
    "fused": ("camera", "lidar"),
}

#: Inference-time modality-dropout rows derived from the fused checkpoint.
DROPOUT_ROWS: dict[str, tuple[str, ...]] = {
    "fused (drop camera at inference)": ("lidar",),
    "fused (drop lidar at inference)": ("camera",),
}


def _row(
    variant: str,
    metrics: dict[str, Any],
    params: int,
    wall_clock_s: float,
    ms_per_frame: float,
    modalities: Sequence[str],
    kind: str,
) -> dict[str, Any]:
    """Flatten one variant's metrics into an ablation table row."""
    det = metrics["detection"]
    occ = metrics["occupancy"]
    trk = metrics["tracking"]
    ap = {name: det["per_class"][name]["AP"] for name in det["per_class"]}
    return {
        "variant": variant,
        "kind": kind,
        "modalities": "+".join(modalities) if modalities else "-",
        "mAP": det["mAP"],
        "NDS": det["NDS"],
        "ATE": det["mATE"],
        "AOE": det["mAOE"],
        "AP_car": ap.get("car", 0.0),
        "AP_pedestrian": ap.get("pedestrian", 0.0),
        "AP_cyclist": ap.get("cyclist", 0.0),
        "occupancy_mIoU": occ["mIoU"],
        "occupancy_mIoU_observed": occ["mIoU_observed"],
        "occupancy_occupied_IoU": occ["occupied_iou"],
        "occupied_precision": occ["occupied_precision"],
        "occupied_recall": occ["occupied_recall"],
        "MOTA": trk["MOTA"],
        "MOTP": trk["MOTP"],
        "ID_switches": trk["ID_switches"],
        "mostly_tracked": trk["mostly_tracked"],
        "mostly_lost": trk["mostly_lost"],
        "FP": trk["FP"],
        "FN": trk["FN"],
        "AMOTA_style": trk["AMOTA_style"],
        "params": params,
        "train_wall_clock_s": wall_clock_s,
        "inference_ms_per_frame": ms_per_frame,
    }


def row_from_results(
    variant: str,
    metrics: dict[str, Any],
    params: int,
    wall_clock_s: float,
    kind: str = "trained",
) -> dict[str, Any]:
    """Public helper turning :func:`evaluate_model` output into an ablation row."""
    return _row(
        variant,
        metrics,
        params,
        wall_clock_s,
        metrics.get("ms_per_frame", float("nan")),
        metrics.get("modalities", ()),
        kind,
    )


def summary_from_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Reconstruct the training summary of an existing run from its checkpoint.

    This is what makes ``run_ablation(..., reuse=True)`` possible: the trained
    weights, the parameter count and the wall-clock time are all stored in the
    checkpoint, so the whole ablation table can be recomputed without retraining.

    Args:
        checkpoint: Path to a ``ckpt.pt`` written by ``train``.

    Returns:
        A summary dictionary with the same keys :func:`fusedbev3d.train.train` returns.
    """
    import torch

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    history = list(payload.get("history", []))
    extra = payload.get("extra", {}) or {}
    steps = max((int(h.get("step", 0)) for h in history), default=0)
    train_losses = [h["train_loss"] for h in history if "train_loss" in h]
    val_losses = [h["val_loss"] for h in history if "val_loss" in h]
    return {
        "name": Path(checkpoint).parent.name,
        "out": str(Path(checkpoint).parent),
        "checkpoint": str(checkpoint),
        "params": int(extra.get("params", 0)),
        "epochs": int(payload.get("epoch", 0)),
        "steps": steps,
        "wall_clock_s": float(extra.get("wall_clock_s", 0.0)),
        "history": history,
        "final_train_loss": train_losses[-1] if train_losses else float("nan"),
        "val_loss": val_losses[-1] if val_losses else float("nan"),
        "modalities": list(payload.get("modal", [])),
        "device": str(extra.get("device", "cpu")),
        "distributed": False,
        "world_size": 1,
    }


def run_ablation(
    cfg: Config,
    num_scenes: int | None = None,
    variants: Sequence[str] | None = None,
    out_root: str | Path = "runs/ablation",
    dropout_eval: bool = True,
    verbose: bool = True,
    reuse: bool = False,
) -> dict[str, Any]:
    """Train and evaluate every modality variant plus the inference-time dropout rows.

    Args:
        cfg: Base configuration.  ``epochs``, ``lr``, ``batch_size``, ``seed`` and
            every data setting are shared by all variants; only ``modalities``
            (and the run name) change.
        num_scenes: Optional training-scene override (applies to every variant).
        variants: Subset of :data:`VARIANTS` keys to run, in order.
        out_root: Directory holding one run folder per variant.
        dropout_eval: Also evaluate the fused checkpoint with one modality zeroed.
        verbose: Print progress.
        reuse: When true, skip training for variants that already have
            ``<out_root>/<variant>/ckpt.pt`` and re-evaluate from those weights.

    Returns:
        ``{"rows", "trained", "dropout", "pr_curves", "checkpoints", "config",
        "grid", "hardware"}``.
    """
    keys = list(variants or VARIANTS.keys())
    device = pick_device(cfg.train.device)
    cfg.train.device = str(device)
    out_root = Path(out_root)
    rows: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    pr_curves: dict[str, Any] = {}
    checkpoints: dict[str, str] = {}
    occupancy_confusion: dict[str, Any] = {}

    for key in keys:
        if key not in VARIANTS:
            raise KeyError(f"unknown variant {key!r}; known: {sorted(VARIANTS)}")
        mods = VARIANTS[key]
        run_cfg = copy.deepcopy(cfg)
        run_cfg.train.modalities = tuple(mods)
        run_cfg.train.name = key
        run_cfg.train.out = str(out_root / key)
        existing = out_root / key / "ckpt.pt"
        if reuse and existing.exists():
            if verbose:
                print(f"\n===== variant {key} : reusing {existing} =====", flush=True)
            summary = summary_from_checkpoint(existing)
        else:
            if verbose:
                print(f"\n===== variant {key} : modalities={mods} =====", flush=True)
            summary = train(run_cfg, num_scenes=num_scenes, verbose=verbose)
            (out_root / key).mkdir(parents=True, exist_ok=True)
            (out_root / key / "summary.json").write_text(
                json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2, default=str)
            )
        model, payload = load_checkpoint(summary["checkpoint"], device=device)
        ckpt_cfg: Config = payload["config_obj"]
        metrics = evaluate_model(
            model,
            ckpt_cfg,
            split="val",
            scenes=None,
            device=device,
            tracker_kwargs=None,
        )
        row = row_from_results(key, metrics, summary["params"], summary["wall_clock_s"])
        rows.append(row)
        trained[key] = {"summary": summary, "metrics": metrics}
        checkpoints[key] = summary["checkpoint"]
        pr_curves[key] = precision_recall_curves(
            metrics["predictions"], metrics["targets"], ckpt_cfg.model.class_names
        )
        occupancy_confusion[key] = metrics["occupancy"]["confusion"]
        if verbose:
            print(
                f"--- {key}: mAP={row['mAP']:.4f} NDS={row['NDS']:.4f} "
                f"occ_mIoU={row['occupancy_mIoU']:.4f} MOTA={row['MOTA']:.4f} "
                f"IDSW={row['ID_switches']}",
                flush=True,
            )

    if dropout_eval and "fused" in checkpoints:
        model, payload = load_checkpoint(checkpoints["fused"], device=device)
        ckpt_cfg = payload["config_obj"]
        for label, mods in DROPOUT_ROWS.items():
            if verbose:
                print(f"\n===== dropout row {label} : keep {mods} =====", flush=True)
            metrics = evaluate_model(
                model,
                ckpt_cfg,
                split="val",
                scenes=None,
                device=device,
                modalities=mods,
            )
            rows.append(row_from_results(label, metrics, int(payload["extra"].get("params", 0)),
                                         0.0, kind="inference_dropout"))
            if verbose:
                last = rows[-1]
                print(
                    f"--- {label}: mAP={last['mAP']:.4f} NDS={last['NDS']:.4f} "
                    f"occ_mIoU={last['occupancy_mIoU']:.4f} MOTA={last['MOTA']:.4f}",
                    flush=True,
                )

    result = {
        "rows": rows,
        "trained": trained,
        "pr_curves": pr_curves,
        "checkpoints": checkpoints,
        "occupancy_confusion": occupancy_confusion,
        "variants": keys,
        "config": cfg.to_dict(),
        "grid": cfg.grid.describe(),
        "hardware": hardware_info(),
    }
    return result


def hardware_info() -> dict[str, Any]:
    """Return a small description of the machine that produced the numbers."""
    import os
    import platform

    import torch

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu" if not torch.cuda.is_available() else "cuda",
    }


def write_tables(result: dict[str, Any], csv_path: str | Path, md_path: str | Path) -> tuple[Path, Path]:
    """Write the ablation rows to CSV and a markdown table.

    Args:
        result: Output of :func:`run_ablation`.
        csv_path: Destination CSV path.
        md_path: Destination markdown path.

    Returns:
        ``(csv_path, md_path)``.
    """
    import csv

    rows = result["rows"]
    csv_path = Path(csv_path)
    md_path = Path(md_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [k for k in rows[0] if k != "modalities"] if rows else []
    columns.insert(1, "modalities")

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})

    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def save_result(result: dict[str, Any], path: str | Path) -> Path:
    """Dump the whole ablation result (minus non-serialisable objects) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": result["rows"],
        "config": result["config"],
        "grid": result["grid"],
        "hardware": result["hardware"],
        "variants": result["variants"],
        "checkpoints": result["checkpoints"],
        "occupancy_confusion": result["occupancy_confusion"],
        "trained": {
            k: {
                "summary": {kk: vv for kk, vv in v["summary"].items() if kk != "history"},
                "detection": v["metrics"]["detection"],
                "occupancy": v["metrics"]["occupancy"],
                "tracking": v["metrics"]["tracking"],
            }
            for k, v in result.get("trained", {}).items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
