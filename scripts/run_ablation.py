#!/usr/bin/env python3
"""Run the camera-only / lidar-only / fused ablation end to end and write artifacts.

Example::

    python scripts/run_ablation.py --epochs 12

Trains the three variants under identical seeds and schedules, evaluates detection,
occupancy and tracking for each, adds two inference-time modality-dropout rows
(the fused checkpoint evaluated with one modality zeroed), then writes:

* ``results/ablation.csv`` and ``results/ablation.md``   - the results table
* ``results/ablation_full.json``                          - every raw metric
* ``results/plots/ablation_map.png``                      - mAP / NDS / occupancy / MOTA bars
* ``results/plots/pr_curves.png``                         - per-class precision-recall curves
* ``results/plots/occupancy_confusion.png``               - occupancy confusion matrix
* ``results/qualitative/*.png``                           - BEV overlays of real val frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from fusedbev3d.ablation import VARIANTS, run_ablation, save_result, write_tables  # noqa: E402
from fusedbev3d.config import Config, load_config  # noqa: E402
from fusedbev3d.data.registry import DATASETS  # noqa: E402
from fusedbev3d.inference import make_loader  # noqa: E402
from fusedbev3d.train import load_checkpoint  # noqa: E402
from fusedbev3d.viz import render_qualitative_frame  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the ablation."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None, choices=list(DATASETS))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--scenes", type=int, default=None, help="training scenes per variant")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--variants", default=",".join(VARIANTS), help="comma separated variant subset")
    p.add_argument("--out-root", default="runs/ablation", help="directory for the per-variant runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--no-dropout", action="store_true", help="skip the inference-time dropout rows")
    p.add_argument("--reuse", action="store_true",
                   help="re-evaluate existing <out-root>/<variant>/ckpt.pt instead of retraining")
    p.add_argument("--no-qualitative", action="store_true")
    p.add_argument("--config", default=None, help="YAML/JSON config override file")
    return p.parse_args(argv)


def plot_ablation(result: dict, path: Path) -> Path:
    """Bar chart of mAP / NDS / occupancy mIoU / MOTA per variant."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = result["rows"]
    labels = [r["variant"] for r in rows]
    metrics = ["mAP", "NDS", "occupancy_mIoU", "MOTA"]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [r[m] for r in rows], width, label=m)
    for i, m in enumerate(metrics):
        for xi, r in zip(x + (i - 1.5) * width, rows, strict=False):
            ax.text(xi, r[m] + 0.01, f"{r[m]:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.12)
    ax.set_title("FusedBEV3D modality ablation (val split)")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pr_curves(result: dict, path: Path) -> Path:
    """Per-class precision-recall curves for every trained variant."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    curves = result.get("pr_curves", {})
    variants = list(curves)
    if not variants:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "no PR curve data", ha="center")
    else:
        fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 4.8), sharey=True)
        if len(variants) == 1:
            axes = [axes]
        for ax, name in zip(axes, variants, strict=False):
            for cls, data in curves[name].items():
                ax.plot(data["recall"], data["precision"], label=f"{cls} (AP={data['ap']:.3f})")
            ax.set_title(name)
            ax.set_xlabel("recall")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("precision")
    fig.suptitle("Per-class precision-recall at a 2.0 m centre-distance threshold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_occupancy_confusion(result: dict, path: Path) -> Path:
    """Normalised occupancy confusion matrix of the fused variant (or the first one)."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    conf = result.get("occupancy_confusion", {})
    key = "fused" if "fused" in conf else next(iter(conf), None)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    if key is None:
        ax.text(0.5, 0.5, "no confusion data", ha="center")
    else:
        cm = np.asarray(conf[key], dtype=np.float64)
        cm = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)
        im = ax.imshow(cm, cmap="magma", vmin=0, vmax=1)
        names = ["free", "occupied", "unknown"]
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), names)
        ax.set_xlabel("ground truth")
        ax.set_ylabel("prediction")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"occupancy confusion (column-normalised) - {key}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def render_qualitative(cfg: Config, ckpt: str, out_dir: Path, n_frames: int = 3) -> list[str]:
    """Render a few real validation frames with GT and predicted boxes plus occupancy."""
    from fusedbev3d.train import pick_device

    model, payload = load_checkpoint(ckpt, device=pick_device(cfg.train.device))
    ckpt_cfg: Config = payload["config_obj"]
    loader = make_loader(ckpt_cfg, "val", scenes=1, batch_size=n_frames)
    batch = next(iter(loader))
    paths: list[str] = []
    for i in range(min(n_frames, batch["images"].shape[0])):
        p = out_dir / f"val_{batch['tokens'][i].replace('/', '_')}.png"
        render_qualitative_frame(
            model,
            batch,
            i,
            p,
            ckpt_cfg.grid,
            title=f"{batch['tokens'][i]} - fused (camera+lidar)",
        )
        paths.append(str(p))
    return paths


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the ablation, write tables, plots and qualitative frames."""
    args = parse_args(argv)
    overrides: dict[str, object] = {"train.epochs": args.epochs}
    if args.dataset:
        overrides["data.name"] = args.dataset
    if args.batch_size is not None:
        overrides["train.batch_size"] = args.batch_size
    if args.seed is not None:
        overrides["train.seed"] = args.seed
        overrides["data.seed"] = args.seed
    if args.device is not None:
        overrides["train.device"] = args.device
    cfg = load_config(args.config, **overrides)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    result = run_ablation(
        cfg,
        num_scenes=args.scenes,
        variants=variants,
        out_root=args.out_root,
        dropout_eval=not args.no_dropout,
        reuse=args.reuse,
    )

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    csv_path, md_path = write_tables(result, results_dir / "ablation.csv", results_dir / "ablation.md")
    save_result(result, results_dir / "ablation_full.json")
    plot_ablation(result, plots_dir / "ablation_map.png")
    plot_pr_curves(result, plots_dir / "pr_curves.png")
    plot_occupancy_confusion(result, plots_dir / "occupancy_confusion.png")

    qualitative: list[str] = []
    fused_ckpt = result["checkpoints"].get("fused")
    if fused_ckpt and not args.no_qualitative:
        qualitative = render_qualitative(cfg, fused_ckpt, results_dir / "qualitative", n_frames=3)
        (results_dir / "ablation_full.json").write_text(
            json.dumps({**json.loads((results_dir / "ablation_full.json").read_text()),
                        "qualitative": qualitative}, indent=2, default=str)
        )

    print("\n== ablation table ==")
    print(md_path.read_text())
    print(f"wrote {csv_path}, {md_path}, {results_dir / 'ablation_full.json'}")
    print(f"wrote {plots_dir / 'ablation_map.png'}, {plots_dir / 'pr_curves.png'}, "
          f"{plots_dir / 'occupancy_confusion.png'}")
    if qualitative:
        print("quantitative frames:", *qualitative, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
