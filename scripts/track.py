#!/usr/bin/env python3
"""Run the detector over validation sequences in temporal order, track, evaluate.

Example::

    python scripts/track.py --ckpt runs/fused/ckpt.pt

Uses :class:`~fusedbev3d.tracking.tracker.Greedy3DTracker` (Kalman-filtered, two-stage
Hungarian association by default; ``--assoc greedy`` switches to greedy) and reports
CLEAR-MOT metrics (MOTA, MOTP, ID switches, FP/FN, MT/ML) plus a multi-gate
AMOTA-style average.  Results are written to ``results/tracking_<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusedbev3d.config import Config  # noqa: E402
from fusedbev3d.eval.tracking import print_tracking_table  # noqa: E402
from fusedbev3d.inference import make_loader, track_split  # noqa: E402
from fusedbev3d.models.fusedbev3d import normalize_modalities  # noqa: E402
from fusedbev3d.train import load_checkpoint  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for tracking."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint produced by scripts/train.py")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--scenes", type=int, default=None)
    p.add_argument("--score-threshold", type=float, default=0.1, help="detections below this are not tracked")
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
        "method": args.assoc,
        "report_threshold": args.report_threshold,
    }
    loader = make_loader(cfg, args.split, args.scenes, cfg.train.batch_size)
    result = track_split(
        model,
        cfg,
        loader,
        tracker_kwargs=tracker_kwargs,
        score_threshold=args.score_threshold,
        topk=args.topk,
        device=device,
        eval_gate=args.eval_gate,
        modalities=cfg.train.modalities,
    )
    result.update(
        {
            "split": args.split,
            "modalities": list(cfg.train.modalities),
            "checkpoint": str(args.ckpt),
            "score_threshold": args.score_threshold,
            "association": args.assoc,
            "association_gate": args.dist_gate,
            "eval_gate": args.eval_gate,
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
