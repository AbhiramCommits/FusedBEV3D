"""Training loop with DDP, AMP, cosine schedule with warmup and checkpointing.

Runs single-process on CPU out of the box and scales to multi-GPU ``torchrun``
launches with no code changes: distributed state is read from the standard
environment variables, the datasets are wrapped in a ``DistributedSampler``, the
model is wrapped in ``DistributedDataParallel`` and only rank 0 logs and writes
checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from fusedbev3d.config import Config
from fusedbev3d.data.registry import build_dataset
from fusedbev3d.data.synth import collate_fn
from fusedbev3d.models.fusedbev3d import FusedBEV3D, normalize_modalities
from fusedbev3d.targets import targets_from_batch

__all__ = ["train", "set_seed", "pick_device", "build_dataloaders", "batch_to_device", "load_checkpoint"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (deterministic, CPU-friendly)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(spec: str = "auto") -> torch.device:
    """Resolve a device specification into a :class:`torch.device`.

    Args:
        spec: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:<i>"``.

    Returns:
        The resolved device (``cuda`` only when actually available).
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)


def _dist_env() -> tuple[int, int, int]:
    """Read rank/world-size/local-rank from the environment (``torchrun`` compatible)."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local


def init_distributed(device: torch.device) -> tuple[bool, int, int]:
    """Initialise the process group when launched under ``torchrun``.

    Args:
        device: The device this process will train on.

    Returns:
        ``(distributed, rank, world_size)``.
    """
    rank, world, local = _dist_env()
    if world <= 1 or not dist.is_available():
        return False, 0, 1
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    if device.type == "cuda":
        torch.cuda.set_device(local)
    return True, rank, world


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Recursively move every tensor in a collated batch onto ``device``."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
            out[key] = [v.to(device, non_blocking=True) for v in value]
        else:
            out[key] = value
    return out


def build_dataloaders(
    cfg: Config,
    rank: int = 0,
    world_size: int = 1,
    num_scenes: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build the train/val dataloaders (with distributed samplers when needed).

    Args:
        cfg: Full configuration.
        rank: Process rank.
        world_size: Number of processes.
        num_scenes: Optional override of the training scene count.

    Returns:
        ``(train_loader, val_loader)``.
    """
    train_ds = build_dataset(cfg.data.name, "train", cfg, num_scenes=num_scenes)
    val_ds = build_dataset(cfg.data.name, "val", cfg)
    distributed = world_size > 1
    gen = torch.Generator()
    gen.manual_seed(cfg.train.seed)
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.train.seed
        )
        if distributed
        else None
    )
    val_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        if distributed
        else None
    )
    common = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": False,
    }
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        generator=gen if train_sampler is None else None,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **common)
    return train_loader, val_loader


def _lr_lambda(warmup_steps: int, total_steps: int):
    """Return a callable implementing linear warmup + cosine decay."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return fn


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    """Mean total loss over a loader (used for the validation curve)."""
    model.eval()
    net = model.module if hasattr(model, "module") else model
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = batch_to_device(batch, device)
        targets = targets_from_batch(batch, cfg.grid, cfg.model.num_classes)
        outputs = model(batch)
        losses = net.loss(outputs, targets, cfg.train.loss_weights)
        total += float(losses["total"].detach())
        count += 1
    model.train()
    return total / max(count, 1)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    cfg: Config,
    epoch: int,
    history: Iterable[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a self-contained checkpoint that :func:`load_checkpoint` can restore."""
    path.parent.mkdir(parents=True, exist_ok=True)
    net = model.module if hasattr(model, "module") else model
    payload = {
        "model": net.state_dict(),
        "config": cfg.to_dict(),
        "epoch": epoch,
        "history": list(history),
        "image_hw": net.image_hw,
        "modal": list(net.modalities),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[FusedBEV3D, dict[str, Any]]:
    """Rebuild a :class:`FusedBEV3D` from a checkpoint.

    Args:
        path: Checkpoint path written by :func:`save_checkpoint`.
        device: Device to load the weights onto.

    Returns:
        ``(model, payload)`` where ``payload`` holds the config, epoch and history.
    """
    from fusedbev3d.config import Config as _Config  # local import to avoid a cycle
    from fusedbev3d.config import DataConfig, GridConfig, ModelConfig, TrainConfig

    device = torch.device(device)
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    raw = payload["config"]
    cfg = _Config(
        grid=GridConfig(**{**raw["grid"], "x_range": tuple(raw["grid"]["x_range"]),
                           "y_range": tuple(raw["grid"]["y_range"]),
                           "z_range": tuple(raw["grid"]["z_range"])}),
        model=ModelConfig(**{**raw["model"], "class_names": tuple(raw["model"]["class_names"])}),
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
    )
    model = FusedBEV3D(
        cfg.model, cfg.grid, tuple(payload["image_hw"]), cfg.train.modalities
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    payload["config_obj"] = cfg
    return model, payload


def train(cfg: Config, num_scenes: int | None = None, verbose: bool = True) -> dict[str, Any]:
    """Train a FusedBEV3D model and write a checkpoint plus a JSONL metric log.

    Args:
        cfg: Full configuration (data/model/train groups).
        num_scenes: Optional override of the training scene count.
        verbose: Print per-epoch progress on rank 0.

    Returns:
        A summary dictionary with the epoch history, wall-clock time, parameter
        count and the checkpoint path.
    """
    device = pick_device(cfg.train.device)
    distributed, rank, world = init_distributed(device)
    set_seed(cfg.train.seed + rank)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // max(world, 1)))

    train_loader, val_loader = build_dataloaders(cfg, rank, world, num_scenes)
    modalities = normalize_modalities(cfg.train.modalities)
    model = FusedBEV3D(
        cfg.model, cfg.grid, (cfg.data.image_height, cfg.data.image_width), modalities
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
    warmup = max(int(cfg.train.warmup_frac * total_steps), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(warmup, total_steps))
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    out_dir = cfg.train.resolved_out()
    log_path = out_dir / "metrics.jsonl"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")

    history: list[dict[str, Any]] = []
    step = 0
    start = time.time()
    model.train()
    stop = False
    for epoch in range(cfg.train.epochs):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        running, seen = 0.0, 0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            targets = targets_from_batch(batch, cfg.grid, cfg.model.num_classes)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(batch)
                losses = model.module.loss(outputs, targets, cfg.train.loss_weights) if distributed else model.loss(outputs, targets, cfg.train.loss_weights)
            if scaler is not None:
                scaler.scale(losses["total"]).backward()
                if cfg.train.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                if cfg.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
            scheduler.step()
            step += 1
            running += float(losses["total"].detach())
            seen += 1
            if rank == 0 and verbose and cfg.train.log_every and step % cfg.train.log_every == 0:
                print(
                    f"  epoch {epoch + 1}/{cfg.train.epochs} step {step}/{total_steps} "
                    f"loss {running / seen:.4f} lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if step >= total_steps:
                stop = True
                break
        epoch_loss = running / max(seen, 1)
        record = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss": epoch_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_s": time.time() - start,
        }
        if rank == 0 and cfg.train.eval_every and (epoch + 1) % cfg.train.eval_every == 0:
            record["val_loss"] = evaluate_loss(model, val_loader, cfg, device)
        history.append(record)
        if rank == 0:
            with log_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            if verbose:
                print(
                    f"epoch {epoch + 1}/{cfg.train.epochs} train_loss={epoch_loss:.4f} "
                    f"lr={record['lr']:.2e} ({record['elapsed_s']:.1f}s)",
                    flush=True,
                )
        if stop:
            break

    wall = time.time() - start
    ckpt = out_dir / "ckpt.pt"
    if rank == 0:
        val_loss = evaluate_loss(model, val_loader, cfg, device)
        train_losses = [h["train_loss"] for h in history if "train_loss" in h]
        final_train_loss = train_losses[-1] if train_losses else float("nan")
        history.append({"epoch": cfg.train.epochs, "val_loss": val_loss, "final": True})
        save_checkpoint(
            ckpt,
            model,
            cfg,
            epoch=len(history),
            history=history,
            extra={"params": params, "wall_clock_s": wall, "device": str(device)},
        )
        with log_path.open("a") as fh:
            fh.write(json.dumps({"final": True, "val_loss": val_loss, "params": params, "wall_clock_s": wall}) + "\n")
        if verbose:
            print(f"checkpoint -> {ckpt}  params={params}  wall={wall:.1f}s  val_loss={val_loss:.4f}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    train_losses_all = [h["train_loss"] for h in history if "train_loss" in h]
    return {
        "name": cfg.train.name,
        "out": str(out_dir),
        "checkpoint": str(ckpt),
        "params": params,
        "epochs": cfg.train.epochs,
        "steps": step,
        "wall_clock_s": wall,
        "history": history,
        "final_train_loss": train_losses_all[-1] if train_losses_all else float("nan"),
        "val_loss": history[-1].get("val_loss", float("nan")),
        "modalities": list(modalities),
        "device": str(device),
        "distributed": distributed,
        "world_size": world,
    }
