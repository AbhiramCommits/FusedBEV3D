#!/usr/bin/env python3
"""Evaluate a trained FusedBEV3D checkpoint on a split.

Examples::

    python scripts/eval.py --ckpt runs/fused/ckpt.pt
    python scripts/eval.py --ckpt runs/fused/ckpt.pt --out results/detection_fused.json

Prints per-class AP at the nuScenes distance thresholds, mAP, the ATE/ASE/AOE/AVE
TP-error breakdown and the NDS-style aggregate, and writes
``results/detection_<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusedbev3d.config import Config  # noqa: E402
from fusedbev3d.data.registry import build_dataset  # noqa: E402
from fusedbev3d.data.synth import collate_fn  # noqa: E402
from fusedbev3d.eval.detection import evaluate_detection, print_table  # noqa: E402
from fusedbev3d.models.fusedbev3d import normalize_modalities  # noqa: E402
from fusedbev3d.train import load_checkpoint  # noqa: E402

PREDICT_SCORE_THRESHOLD = 0.05
DECODE_TOPK = 100


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint produced by scripts/train.py")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--scenes", type=int, default=None, help="override the number of scenes")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--modalities", default=None, help="override the checkpoint modalities")
    p.add_argument("--score-threshold", type=float, default=PREDICT_SCORE_THRESHOLD)
    p.add_argument("--topk", type=int, default=DECODE_TOPK)
    p.add_argument("--out", default=None, help="detection JSON output path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", action="store_true", help="also print the raw metrics JSON")
    return p.parse_args(argv)


@torch.no_grad()
def predict_split(
    model,
    cfg: Config,
    split: str,
    scenes: int | None,
    batch_size: int,
    score_threshold: float,
    topk: int,
    device: torch.device,
) -> tuple[list[dict], list[torch.Tensor], float]:
    """Run the detector over a split and collect per-frame predictions.

    Args:
        model: Loaded :class:`~fusedbev3d.models.fusedbev3d.FusedBEV3D`.
        cfg: Configuration stored in the checkpoint.
        split: ``"train"`` or ``"val"``.
        scenes: Optional scene-count override.
        batch_size: Evaluation batch size.
        score_threshold: Minimum decoded score for a detection to be kept.
        topk: Number of heatmap peaks decoded per frame.
        device: Torch device.

    Returns:
        ``(predictions, targets, ms_per_frame)``.
    """
    ds = build_dataset(cfg.data.name, split, cfg, num_scenes=scenes)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    model.eval()
    predictions: list[dict] = []
    targets: list[torch.Tensor] = []
    frames = 0
    start = time.time()
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch, decode=True, topk=topk)
        det = out["detections"]
        keep = det["keep"] & (det["scores"] >= score_threshold)
        for i in range(det["boxes"].shape[0]):
            m = keep[i]
            predictions.append(
                {
                    "boxes": det["boxes"][i][m].cpu(),
                    "scores": det["scores"][i][m].cpu(),
                    "classes": det["classes"][i][m].cpu(),
                }
            )
            targets.append(batch["boxes"][i].cpu())
        frames += det["boxes"].shape[0]
    ms = (time.time() - start) * 1000.0 / max(frames, 1)
    return predictions, targets, ms


def main(argv: list[str] | None = None) -> int:
    """Entry point: evaluate detection and write ``results/detection_<name>.json``."""
    args = parse_args(argv)
    model, payload = load_checkpoint(args.ckpt, device=args.device)
    cfg: Config = payload["config_obj"]
    if args.modalities:
        cfg.train.modalities = normalize_modalities(args.modalities)
    name = cfg.train.name
    device = torch.device(args.device)
    model.to(device)
    batch_size = args.batch_size or cfg.train.batch_size

    preds, targets, ms = predict_split(
        model, cfg, args.split, args.scenes, batch_size, args.score_threshold, args.topk, device
    )
    det = evaluate_detection(preds, targets, cfg.model.class_names)
    det.update(
        {
            "ms_per_frame": ms,
            "split": args.split,
            "modalities": list(cfg.train.modalities),
            "checkpoint": str(args.ckpt),
        }
    )
    out = Path(args.out) if args.out else Path("results") / f"detection_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(det, indent=2))
    print_table(det, title=f"3D detection - {name} [{'+'.join(cfg.train.modalities)}] on {args.split}")
    print(f"wrote {out}")
    if args.json:
        print(json.dumps(det, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
