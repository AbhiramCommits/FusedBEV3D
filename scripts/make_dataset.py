#!/usr/bin/env python3
"""Materialise a FusedBEV3D split into ``data/cache/`` and print its statistics.

Examples::

    python scripts/make_dataset.py --split train --scenes 40
    python scripts/make_dataset.py --split val --scenes 10
    python scripts/make_dataset.py --dataset nuscenes-mini --dataroot data/nuscenes

The command prints the number of frames, the box count per class, the mean number
of lidar points per frame and the free/occupied/unknown occupancy balance, then
writes the split to ``data/cache/<dataset>_<split>.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusedbev3d.config import load_config  # noqa: E402
from fusedbev3d.data.registry import DATASETS, cache_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the dataset builder."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="synth", choices=list(DATASETS))
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--scenes", type=int, default=None, help="number of scenes (default: config)")
    p.add_argument("--frames", type=int, default=None, help="frames per scene (synth only)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None, help="optional YAML/JSON config override file")
    p.add_argument("--out", default=None, help="cache output path (default data/cache/<ds>_<split>.pt)")
    p.add_argument("--no-cache", action="store_true", help="generate without writing a cache file")
    p.add_argument("--stats-json", default=None, help="also write the statistics to this JSON file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: build the split, print statistics, write the cache."""
    args = parse_args(argv)
    overrides: dict[str, object] = {}
    if args.seed is not None:
        overrides["data.seed"] = args.seed
    if args.frames is not None:
        overrides["data.frames_per_scene"] = args.frames
    cfg = load_config(args.config, **overrides) if args.config else load_config(**overrides)
    cfg.data.name = args.dataset

    scenes = args.scenes
    if scenes is None:
        scenes = cfg.data.val_scenes if args.split == "val" else cfg.data.num_scenes

    print(f"[make_dataset] dataset={args.dataset} split={args.split} scenes={scenes} "
          f"frames/scene={cfg.data.frames_per_scene} seed={cfg.data.seed}")
    print(f"[make_dataset] grid: {cfg.grid.describe()}")

    if args.dataset == "synth":
        from fusedbev3d.data.synth import SyntheticDrivingDataset

        ds = SyntheticDrivingDataset(split=args.split, num_scenes=scenes, grid=cfg.grid, cfg=cfg.data)
        ds.build(verbose=True)
    else:
        from fusedbev3d.data.nuscenes_adapter import NuScenesMiniDataset

        ds = NuScenesMiniDataset(split=args.split, num_scenes=scenes, grid=cfg.grid, cfg=cfg.data)

    stats = ds.statistics()
    out = Path(args.out) if args.out else cache_path(args.split, cfg.data, args.dataset)
    if not args.no_cache:
        ds.save_cache(out)
        print(f"[make_dataset] wrote cache -> {out} ({out.stat().st_size / 1e6:.1f} MB)")

    print(f"[make_dataset] frames           : {stats['frames']} ({stats['scenes']} scenes)")
    print(f"[make_dataset] boxes total      : {stats['boxes_total']}")
    for name, count in stats["boxes_per_class"].items():
        print(f"[make_dataset]   {name:<11}: {count}")
    print(f"[make_dataset] points/frame     : {stats['points_per_frame']:.1f}")
    occ = stats["occupancy_counts"]
    frac = stats["occupancy_fraction"]
    print(
        "[make_dataset] occupancy voxels : "
        f"free={occ['free']} ({frac['free']:.3f}) "
        f"occupied={occ['occupied']} ({frac['occupied']:.4f}) "
        f"unknown={occ['unknown']} ({frac['unknown']:.3f})"
    )

    if args.stats_json:
        p = Path(args.stats_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"config": cfg.to_dict(), "stats": stats, "cache": str(out)}, indent=2))
        print(f"[make_dataset] wrote stats -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
