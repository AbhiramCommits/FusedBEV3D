"""Shape and finiteness checks for every model component (``pytest -k shapes``)."""

from __future__ import annotations

import torch

from fusedbev3d.config import GridConfig, ModelConfig
from fusedbev3d.models.camera_encoder import CameraEncoder, bev_pool, frustum_grid
from fusedbev3d.models.fusion import BEVFusion
from fusedbev3d.models.heads import CenterHead, OccupancyHead
from fusedbev3d.models.pillar_encoder import PillarEncoder, voxelize
from fusedbev3d.targets import build_center_targets, build_occupancy_targets


def test_pillar_encoder_shapes(batch, cfg):
    """Pillar encoder yields a (B, C, H, W) BEV map with the configured width."""
    enc = PillarEncoder(cfg.grid, cfg.model)
    out = enc(batch)
    b = batch["num_boxes"].shape[0]
    assert out["bev"].shape == (b, cfg.model.pillar_feat, cfg.grid.bev_h, cfg.grid.bev_w)
    assert torch.isfinite(out["bev"]).all()


def test_camera_encoder_shapes(batch, cfg):
    """Camera encoder lifts images onto the shared BEV grid."""
    enc = CameraEncoder(cfg.grid, cfg.model, (cfg.data.image_height, cfg.data.image_width))
    out = enc(batch)
    b = batch["num_boxes"].shape[0]
    assert out["bev"].shape == (b, cfg.model.camera_context, cfg.grid.bev_h, cfg.grid.bev_w)
    assert torch.isfinite(out["bev"]).all()


def test_fusion_shapes_all_modalities(batch, cfg):
    """Fusion accepts lidar-only, camera-only and both."""
    lid = torch.randn(2, cfg.model.pillar_feat, cfg.grid.bev_h, cfg.grid.bev_w)
    cam = torch.randn(2, cfg.model.camera_context, cfg.grid.bev_h, cfg.grid.bev_w)
    fus = BEVFusion(cfg.model, cfg.model.pillar_feat, cfg.model.camera_context)
    for a, b in ((lid, None), (None, cam), (lid, cam)):
        out = fus(a, b)
        assert out.shape == (2, cfg.model.bev_backbone_channels, cfg.grid.bev_h, cfg.grid.bev_w)


def test_center_head_decode_shapes(batch, cfg):
    """Decoding returns exactly k boxes/scores/classes per sample."""
    head = CenterHead(cfg.model, cfg.model.fuse_channels)
    feats = torch.randn(2, cfg.model.fuse_channels, cfg.grid.bev_h, cfg.grid.bev_w)
    out = head(feats)
    assert out["heatmap"].shape == (2, cfg.model.num_classes, cfg.grid.bev_h, cfg.grid.bev_w)
    assert out["reg"].shape == (2, cfg.model.reg_channels, cfg.grid.bev_h, cfg.grid.bev_w)
    dec = head.decode(out, cfg.grid, k=25)
    assert dec["boxes"].shape == (2, 25, 9)
    assert dec["scores"].shape == (2, 25)


def test_occupancy_head_shapes(cfg):
    """Occupancy head outputs Z x 3 logits per BEV cell."""
    head = OccupancyHead(cfg.model, cfg.grid, cfg.model.fuse_channels)
    feats = torch.randn(2, cfg.model.fuse_channels, cfg.grid.bev_h, cfg.grid.bev_w)
    out = head(feats)["occupancy"]
    assert out.shape == (2, cfg.grid.num_z, cfg.model.occupancy_classes, cfg.grid.bev_h, cfg.grid.bev_w)
    assert head.predict(out).shape == (2, cfg.grid.num_z, cfg.grid.bev_h, cfg.grid.bev_w)


def test_voxelize_padding_contract(cfg):
    """Padding, ranks and masks are self-consistent for a hand-built point cloud."""
    pts = torch.tensor(
        [
            [0.1, 0.1, 0.2, 1.0],
            [0.2, 0.1, 0.3, 1.0],
            [0.3, 0.1, 1.0, 1.0],
            [5.05, 5.05, 0.5, 1.0],
        ]
    )
    v = voxelize(pts, cfg.grid, None, max_points_per_pillar=2)
    assert v["features"].shape == (2, 2, 7)
    assert v["mask"].sum() == 3
    assert v["num_points"].tolist() == [3, 1]
    assert v["coords"].shape == (2, 2)


def test_frustum_grid_shapes(cfg):
    """The LSS frustum has D * h * w entries with in-grid validity flags."""
    from fusedbev3d.data.synth import camera_extrinsics, camera_intrinsics

    k = torch.tensor(camera_intrinsics(64, 192, 90.0), dtype=torch.float32)
    e = torch.tensor(camera_extrinsics(0), dtype=torch.float32)
    g = frustum_grid(k, e, cfg.grid, 2.0, 30.0, 8, (32, 96), 2)
    assert g["points_ego"].shape == (8 * 32 * 96, 3)
    assert g["bev_index"].shape == (8 * 32 * 96,)
    assert g["valid"].shape == (8 * 32 * 96,)
    assert bool(g["valid"].any())


def test_bev_pool_conserves_mass():
    """Bucketed-sum pooling is a pure scatter-add, so mass is conserved."""
    grid = GridConfig()
    n = 40
    feats = torch.randn(1, 5, n)
    index = torch.randint(0, grid.bev_h * grid.bev_w, (1, n))
    valid = torch.ones(1, n, dtype=torch.bool)
    out = bev_pool(feats, index, valid, grid, 1)
    assert out.shape == (1, 5, grid.bev_h, grid.bev_w)
    assert torch.allclose(out.sum(), feats.sum(), atol=1e-4)
    valid[:] = False
    assert bev_pool(feats, index, valid, grid, 1).abs().sum() == 0


def test_model_forward_and_loss_finite(batch, model, cfg):
    """The full network runs forward + loss for every modality configuration."""
    from fusedbev3d.targets import targets_from_batch

    targets = targets_from_batch(batch, cfg.grid, cfg.model.num_classes)
    assert targets["center"]["heatmap"].shape[:2] == (2, cfg.model.num_classes)
    assert build_occupancy_targets(batch["occupancy"]).dtype == torch.long
    for mods in (("camera",), ("lidar",), ("camera", "lidar")):
        out = model(batch, modalities=mods)
        losses = model.loss(out, targets)
        assert torch.isfinite(losses["total"]), mods
        for key, value in losses.items():
            assert torch.isfinite(value).all(), (mods, key)


def test_center_targets_peak_at_box_centre(cfg):
    """The rendered heatmap peaks exactly at the projected box centre cell."""
    boxes = torch.tensor([[3.2, -1.4, 1.0, 4.2, 1.8, 1.6, 0.3, 1.0, 0.0, 0, 7.0]])
    t = build_center_targets([boxes], cfg.grid, 3)
    peak = t["heatmap"][0, 0].argmax()
    py, px = int(peak // cfg.grid.bev_w), int(peak % cfg.grid.bev_w)
    exp_px = int((3.2 - cfg.grid.x_range[0]) / cfg.grid.voxel_size)
    exp_py = int((-1.4 - cfg.grid.y_range[0]) / cfg.grid.voxel_size)
    assert (px, py) == (exp_px, exp_py)
    assert t["mask"][0, exp_py, exp_px]


def test_model_config_defaults_are_sane():
    """Guard the documented configuration defaults used by the README."""
    m = ModelConfig()
    assert m.num_classes == 3 and m.class_names == ("car", "pedestrian", "cyclist")
    assert m.reg_channels == 10
    g = GridConfig()
    assert (g.bev_h, g.bev_w, g.num_z) == (64, 64, 5)
