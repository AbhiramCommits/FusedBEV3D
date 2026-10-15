#!/usr/bin/env python3
"""Train a FusedBEV3D model.

Examples::

    # CPU / single GPU
    python scripts/train.py --modalities camera,lidar --epochs 12 --out runs/fused

    # multi-GPU DDP with AMP
    torchrun --nproc_per_node=4 scripts/train.py --dataset synth --epochs 12 --out runs/fused

Writes ``runs/<name>/ckpt.pt`` and a per-epoch ``metrics.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusedbev3d.config import load_config  # noqa: E402
from fusedbev3d.data.registry import DATASETS  # noqa: E402
from fusedbev3d.train import train  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None, choices=list(DATASETS))
    p.add_argument("--modalities", default="camera,lidar", help="camera,lidar | camera | lidar")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--scenes", type=int, default=None, help="training scenes (default: config)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="auto|cpu|cuda")
    p.add_argument("--amp", action="store_true", default=None, help="force AMP on (CUDA only)")
    p.add_argument("--config", default=None, help="YAML/JSON config override file")
    p.add_argument("--out", default=None, help="run directory (default runs/<name>)")
    p.add_argument("--name", default=None, help="run name")
    p.add_argument("--max-steps", type=int, default=None, help="global step budget override")
    p.add_argument("--summary", default=None, help="optional path for a JSON run summary")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: build the config, train, and report the run summary."""
    args = parse_args(argv)
    overrides: dict[str, object] = {}
    if args.dataset:
        overrides["data.name"] = args.dataset
    if args.modalities:
        overrides["train.modalities"] = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["train.batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["train.lr"] = args.lr
    if args.seed is not None:
        overrides["train.seed"] = args.seed
        overrides["data.seed"] = args.seed
    if args.device is not None:
        overrides["train.device"] = args.device
    if args.amp:
        overrides["train.amp"] = True
    if args.name is not None:
        overrides["train.name"] = args.name
    if args.out is not None:
        overrides["train.out"] = args.out
    if args.max_steps is not None:
        overrides["train.max_steps"] = args.max_steps
    cfg = load_config(args.config, **overrides)

    summary = train(cfg, num_scenes=args.scenes)
    print("== run summary ==")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    if args.summary:
        p = Path(args.summary)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
