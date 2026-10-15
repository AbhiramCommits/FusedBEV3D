"""End-to-end training smoke test: loss decreases and checkpoints round-trip."""

from __future__ import annotations

import copy

import torch

from fusedbev3d.config import Config
from fusedbev3d.models.fusedbev3d import FusedBEV3D
from fusedbev3d.targets import targets_from_batch
from fusedbev3d.train import load_checkpoint, save_checkpoint, set_seed


def _fresh_model(cfg: Config, seed: int = 0) -> FusedBEV3D:
    """Build a deterministically seeded model in training mode."""
    set_seed(seed)
    net = FusedBEV3D(cfg.model, cfg.grid, (cfg.data.image_height, cfg.data.image_width))
    net.train()
    return net


def test_two_training_steps_decrease_the_total_loss(batch, cfg):
    """Two AdamW steps on one real batch must lower the total loss."""
    model = _fresh_model(cfg)
    targets = targets_from_batch(batch, cfg.grid, cfg.model.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = model.loss(out, targets)["total"]
        losses.append(float(loss.detach()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    assert losses[1] < losses[0], f"loss did not decrease: {losses}"


def test_loss_terms_are_all_finite_and_named(batch, cfg):
    """Every documented loss term exists and is finite."""
    model = _fresh_model(cfg)
    targets = targets_from_batch(batch, cfg.grid, cfg.model.num_classes)
    losses = model.loss(model(batch), targets)
    expected = {"heatmap", "offset", "z", "dims", "rot", "vel", "occupancy", "total"}
    assert expected.issubset(losses)
    for name, value in losses.items():
        assert torch.isfinite(value).all(), name


def test_checkpoint_round_trip_reproduces_identical_outputs(batch, cfg, tmp_path):
    """Saving and reloading a checkpoint reproduces bit-identical predictions."""
    model = _fresh_model(cfg, seed=3)
    model.eval()
    path = save_checkpoint(tmp_path / "ckpt.pt", model, cfg, epoch=1, history=[{"epoch": 1}])
    assert path.exists()

    reloaded, payload = load_checkpoint(path, device="cpu")
    assert payload["epoch"] == 1
    assert payload["history"] == [{"epoch": 1}]
    assert tuple(payload["image_hw"]) == (cfg.data.image_height, cfg.data.image_width)

    with torch.no_grad():
        before = model(batch)
        after = reloaded(batch)
    assert torch.equal(before["center"]["heatmap"], after["center"]["heatmap"])
    assert torch.equal(before["occupancy"], after["occupancy"])
    assert torch.equal(before["features"], after["features"])


def test_checkpoint_restores_the_configuration(cfg, model, tmp_path):
    """The checkpoint carries the full config so evaluation is self-contained."""
    path = save_checkpoint(tmp_path / "ckpt.pt", model, cfg, epoch=2, history=[])
    _, payload = load_checkpoint(path)
    restored: Config = payload["config_obj"]
    assert restored.grid.bev_h == cfg.grid.bev_h
    assert restored.grid.z_range == cfg.grid.z_range
    assert restored.model.num_classes == cfg.model.num_classes
    assert tuple(restored.train.modalities) == tuple(cfg.train.modalities)


def test_set_seed_is_reproducible(cfg):
    """Two models built under the same seed start from identical weights."""
    a = _fresh_model(cfg, seed=11)
    b = _fresh_model(cfg, seed=11)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_train_writes_metrics_and_checkpoint(tmp_path, cfg, monkeypatch):
    """A tiny ``train`` run writes ``metrics.jsonl`` and ``ckpt.pt``.

    The run uses one scene for a single epoch so it stays inside a few seconds on
    a CPU-only machine.
    """
    from fusedbev3d import train as train_module

    small = copy.deepcopy(cfg)
    small.train.epochs = 1
    small.train.batch_size = 2
    small.train.num_workers = 0
    small.train.out = str(tmp_path / "run")
    small.train.name = "smoke"
    small.train.log_every = 0
    small.data.val_scenes = 1

    summary = train_module.train(small, num_scenes=1, verbose=False)
    out = tmp_path / "run"
    assert (out / "ckpt.pt").exists()
    log = (out / "metrics.jsonl").read_text().strip().splitlines()
    assert log and "train_loss" in log[0]
    assert summary["steps"] > 0
    assert summary["params"] > 0
    assert summary["val_loss"] == summary["val_loss"]  # not NaN
    assert summary["wall_clock_s"] > 0
    del monkeypatch
