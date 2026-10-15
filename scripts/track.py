#!/usr/bin/env python3
"""Run the detector over validation sequences in temporal order, track, evaluate.

Example::

    python scripts/track.py --ckpt runs/fused/ckpt.pt

Uses :class:`~fusedbev3d.tracking.tracker.Greedy3DTracker` (Kalman-filtered,
Hungarian association by default; ``--assoc greedy`` switches to greedy) and
reports CLEAR-MOT metrics (MOTA, MOTP, ID switches, FP/FN, MT/ML).  Results are
written to ``results/tracking_<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusedbev3d.config import Config  # noqa: E402
from fusedbev3d.data.registry import build_dataset  # noqa: E402
from fusedbev3d.data.synth import collate_fn  # noqa: E402
from fusedbev3d.eval.tracking import evaluate_tracking, print_tracking_table  # noqa: E402
from fusedbev3d.models.fusedbev3d import normalize_modalities  # noqa: E402
from fusedbev3d.tracking.tracker import Greedy3DTracker  # noqa: E402
from fusedbev3d.train import load_checkpoint  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for tracking."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint produced by scripts/train.py")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--scenes", type=int, default=None)
    p.add_argument("--score-threshold", type=float, default=0.1)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--assoc", default="hungarian", choices=["hungarian", "greedy"])
    p.add_argument("--dist-gate", type=float, default=4.0, help="association gate (m)")
    p.add_argument("--eval-gate", type=float, default=2.0, help="CLEAR-MOT matching gate (m)")
    p.add_argument("--max-age", type=int, default=2)
    p.add_argument("--min-hits", type=int, default=1)
    p.add_argument("--report-threshold", type=float, default=0.4)
    p.add_argument("--modalities", default=None)
    p.add_argument("--out", default=None, help="tracking JSON output path")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


@torch.no_grad()
def run_tracking(
    model,
    cfg: Config,
    split: str,
    scenes: int | None,
    score_threshold: float,
    topk: int,
    tracker_kwargs: dict,
    device: torch.device,
) -> tuple[list[dict], list[np.ndarray], float]:
    """Track a whole split frame by frame, resetting the tracker per sequence.

    Args:
        model: Loaded FusedBEV3D model.
        cfg: Configuration from the checkpoint.
        split: Split name.
        scenes: Optional scene-count override.
        score_threshold: Minimum detection score fed to the tracker.
        topk: Heatmap peaks decoded per frame.
        tracker_kwargs: Constructor kwargs for :class:`Greedy3DTracker`.
        device: Torch device.

    Returns:
        ``(predictions, targets, ms_per_frame)`` where predictions carry
        ``track_ids``.
    """
    ds = build_dataset(cfg.data.name, split, cfg, num_scenes=scenes)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    model.eval()
    tracker = Greedy3DTracker(**tracker_kwargs)
    predictions: list[dict] = []
    targets: list[np.ndarray] = []
    current_scene = -1
    frames = 0
    start = time.time()
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch, decode=True, topk=topk)
        det = out["detections"]
        keep = det["keep"] & (det["scores"] >= score_threshold)
        for i in range(det["boxes"].shape[0]):
            scene = int(batch["scene_id"][i])
            if scene != current_scene:
                tracker.reset()
                current_scene = scene
            mask = keep[i]
            tracked = tracker.update(
                det["boxes"][i][mask].cpu().numpy(),
                det["scores"][i][mask].cpu().numpy(),
                det["classes"][i][mask].cpu().numpy(),
            )
            predictions.append(
                {
                    "boxes": np.stack([t["box"] for t in tracked]) if tracked else np.zeros((0, 9)),
                    "scores": np.array([t["score"] for t in tracked], dtype=np.float64),
                    "classes": np.array([t["class_id"] for t in tracked], dtype=np.int64),
                    "track_ids": np.array([t["track_id"] for t in tracked], dtype=np.int64),
                    "frame_id": int(batch["frame_id"][i]),
                }
            )
            targets.append(batch["boxes"][i].cpu().numpy())
            frames += 1
    ms = (time.time() - start) * 1000.0 / max(frames, 1)
    return predictions, targets, ms


def main(argv: list[str] | None = None) -> int:
    """Entry point: track the split, evaluate CLEAR MOT and write the JSON."""
    args = parse_args(argv)
    model, payload = load_checkpoint(args.ckpt, device=args.device)
    cfg: Config = payload["config_obj"]
    if args.modalities:
        cfg.train.modalities = normalize_modalities(args.modalities)
    name = cfg.train.name
    device = torch.device(args.device)
    model.to(device)

    tracker_kwargs = {
        "max_age": args.max_age,
        "min_hits": args.min_hits,
        "dist_gate": args.dist_gate,
        "report_threshold": args.report_threshold,
        "method": args.assoc,
    }
    preds, targets, ms = run_tracking(
        model, cfg, args.split, args.scenes, args.score_threshold, args.topk, tracker_kwargs, device
    )
    result = evaluate_tracking(preds, targets, dist_threshold=args.eval_gate)
    result.update(
        {
            "split": args.split,
            "modalities": list(cfg.train.modalities),
            "checkpoint": str(args.ckpt),
            "score_threshold": args.score_threshold,
            "association": args.assoc,
            "association_gate": args.dist_gate,
            "eval_gate": args.eval_gate,
            "ms_per_frame": ms,
            "tracker": tracker_kwargs,
        }
    )
    out = Path(args.out) if args.out else Path("results") / f"tracking_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print_tracking_table(result, title=f"tracking - {name} [{'+'.join(cfg.train.modalities)}] on {args.split}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
