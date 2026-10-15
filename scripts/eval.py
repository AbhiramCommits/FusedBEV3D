#!/usr/bin/env python3
"""Evaluate a trained FusedBEV3D checkpoint on a split.

Examples::

    python scripts/eval.py --ckpt runs/fused/ckpt.pt
    python scripts/eval.py --ckpt runs/fused/ckpt.pt --out results/detection_fused.json

Prints per-class AP at the nuScenes distance thresholds, mAP, the ATE/ASE/AOE/AVE
TP-error breakdown and the NDS-style aggregate, plus the occupancy per-class IoU,
and writes ``results/detection_<name>.json`` and ``results/occupancy_<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusedbev3d.config import Config  # noqa: E402
from fusedbev3d.eval.detection import evaluate_detection, print_table  # noqa: E402
from fusedbev3d.eval.occupancy import evaluate_occupancy, print_occupancy_table  # noqa: E402
from fusedbev3d.inference import make_loader, predict_split  # noqa: E402
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
    p.add_argument("--occupancy-out", default=None, help="occupancy JSON output path")
    p.add_argument("--no-occupancy", action="store_true", help="skip the occupancy evaluation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", action="store_true", help="also print the raw metrics JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: evaluate detection + occupancy and write the result JSONs."""
    args = parse_args(argv)
    model, payload = load_checkpoint(args.ckpt, device=args.device)
    cfg: Config = payload["config_obj"]
    if args.modalities:
        cfg.train.modalities = normalize_modalities(args.modalities)
    name = cfg.train.name
    device = torch.device(args.device)
    model.to(device)

    loader = make_loader(cfg, args.split, args.scenes, args.batch_size or cfg.train.batch_size)
    out = predict_split(
        model,
        cfg,
        loader,
        modalities=cfg.train.modalities,
        score_threshold=args.score_threshold,
        topk=args.topk,
        device=device,
    )
    det = evaluate_detection(out["predictions"], out["targets"], cfg.model.class_names)
    det.update(
        {
            "ms_per_frame": out["ms_per_frame"],
            "split": args.split,
            "modalities": list(cfg.train.modalities),
            "checkpoint": str(args.ckpt),
        }
    )
    out_path = Path(args.out) if args.out else Path("results") / f"detection_{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(det, indent=2))
    print_table(det, title=f"3D detection - {name} [{'+'.join(cfg.train.modalities)}] on {args.split}")
    print(f"wrote {out_path}")

    payload_out: dict[str, object] = {"detection": det}
    if not args.no_occupancy:
        occ = evaluate_occupancy(out["occupancy_pred"], out["occupancy_true"])
        occ.update(
            {
                "split": args.split,
                "modalities": list(cfg.train.modalities),
                "checkpoint": str(args.ckpt),
                "ms_per_frame": out["ms_per_frame"],
            }
        )
        occ_path = Path(args.occupancy_out) if args.occupancy_out else Path("results") / f"occupancy_{name}.json"
        occ_path.parent.mkdir(parents=True, exist_ok=True)
        occ_path.write_text(json.dumps(occ, indent=2))
        print_occupancy_table(occ, title=f"occupancy - {name} [{'+'.join(cfg.train.modalities)}] on {args.split}")
        print(f"wrote {occ_path}")
        payload_out["occupancy"] = occ
    if args.json:
        print(json.dumps(payload_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
