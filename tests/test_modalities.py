"""Modality handling: one code path for camera-only, lidar-only and fused."""

from __future__ import annotations

import pytest
import torch

from fusedbev3d.config import GridConfig, ModelConfig
from fusedbev3d.models.fusedbev3d import FusedBEV3D, normalize_modalities
from fusedbev3d.targets import targets_from_batch


def test_normalize_modalities_parses_and_deduplicates():
    """Modality specs are accepted as strings, lists or ``None``."""
    assert normalize_modalities(None) == ("camera", "lidar")
    assert normalize_modalities("lidar,camera") == ("camera", "lidar")
    assert normalize_modalities("camera+lidar") == ("camera", "lidar")
    assert normalize_modalities(["lidar", "lidar"]) == ("lidar",)


def test_normalize_modalities_rejects_unknown_and_empty():
    """Unknown or empty modality specs fail loudly."""
    with pytest.raises(ValueError):
        normalize_modalities("radar")
    with pytest.raises(ValueError):
        normalize_modalities([])


@pytest.mark.parametrize("mods", [("camera",), ("lidar",), ("camera", "lidar")])
def test_forward_runs_for_every_modality_subset(batch, model, cfg, mods):
    """Forward produces finite detection and occupancy heads for all three subsets."""
    out = model(batch, modalities=mods)
    b = batch["num_boxes"].shape[0]
    assert out["features"].shape == (b, cfg.model.bev_backbone_channels, cfg.grid.bev_h, cfg.grid.bev_w)
    assert torch.isfinite(out["features"]).all()
    assert out["occupancy"].shape == (b, cfg.grid.num_z, cfg.model.occupancy_classes, cfg.grid.bev_h, cfg.grid.bev_w)
    assert torch.isfinite(out["occupancy"]).all()
    assert out["modalities"] == tuple(sorted(mods))


def test_disabled_branch_is_not_computed(batch, model):
    """A disabled modality returns ``None`` for its BEV map instead of running the encoder."""
    only_camera = model(batch, modalities=("camera",))
    assert only_camera["lidar_bev"] is None
    assert only_camera["camera_bev"] is not None
    only_lidar = model(batch, modalities=("lidar",))
    assert only_lidar["camera_bev"] is None
    assert only_lidar["lidar_bev"] is not None
    both = model(batch, modalities=("camera", "lidar"))
    assert both["camera_bev"] is not None and both["lidar_bev"] is not None


def test_single_modality_shares_the_fusion_code_path(batch, model):
    """Swapping one modality for zeros keeps every downstream tensor shape identical."""
    lidar = model.lidar(batch)["bev"]
    camera = model.camera(batch)["bev"]
    fused = model.fusion(lidar, camera)
    lidar_only = model.fusion(lidar, None)
    camera_only = model.fusion(None, camera)
    assert fused.shape == lidar_only.shape == camera_only.shape
    assert not torch.allclose(fused, lidar_only)
    assert not torch.allclose(fused, camera_only)


def test_fusion_requires_at_least_one_modality():
    """Fusing nothing is an error rather than a silent zero tensor."""
    from fusedbev3d.models.fusion import BEVFusion

    fusion = BEVFusion(ModelConfig(), 8, 8)
    with pytest.raises(ValueError):
        fusion(None, None)


def test_loss_is_finite_for_every_subset(batch, cfg):
    """Training losses stay finite for camera-only, lidar-only and fused."""
    torch.manual_seed(0)
    model = FusedBEV3D(cfg.model, cfg.grid, (cfg.data.image_height, cfg.data.image_width))
    model.eval()
    targets = targets_from_batch(batch, cfg.grid, cfg.model.num_classes)
    for mods in (("camera",), ("lidar",), ("camera", "lidar")):
        losses = model.loss(model(batch, modalities=mods), targets)
        assert torch.isfinite(losses["total"]), mods
        assert losses["total"] > 0


def test_missing_modality_does_not_change_parameter_count(cfg):
    """All variants instantiate the same parameter set, so the ablation is like-for-like."""
    counts = set()
    for mods in (("camera",), ("lidar",), ("camera", "lidar")):
        net = FusedBEV3D(ModelConfig(), GridConfig(), (64, 192), mods)
        counts.add(sum(p.numel() for p in net.parameters()))
    assert len(counts) == 1
