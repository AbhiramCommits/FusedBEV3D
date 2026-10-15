"""Pillar voxelisation: the fast scatter must match a slow reference loop."""

from __future__ import annotations

import math

import torch

from fusedbev3d.config import GridConfig
from fusedbev3d.models.pillar_encoder import voxelize


def _reference_voxelize(points: torch.Tensor, grid: GridConfig, max_points: int):
    """Loop-based pillar scatter used as the ground truth for :func:`voxelize`."""
    buckets: dict[int, list[list[float]]] = {}
    for row in points.tolist():
        x, y, z, inten = row[0], row[1], row[2], row[3]
        ix = math.floor((x - grid.x_range[0]) / grid.voxel_size)
        iy = math.floor((y - grid.y_range[0]) / grid.voxel_size)
        if not (0 <= ix < grid.bev_w and 0 <= iy < grid.bev_h):
            continue
        buckets.setdefault(iy * grid.bev_w + ix, []).append([x, y, z, inten])
    keys = sorted(buckets)
    features = torch.zeros(len(keys), max_points, 7)
    mask = torch.zeros(len(keys), max_points, dtype=torch.bool)
    coords = torch.zeros(len(keys), 2, dtype=torch.long)
    for slot, key in enumerate(keys):
        ix, iy = key % grid.bev_w, key // grid.bev_w
        xc = grid.x_range[0] + (ix + 0.5) * grid.voxel_size
        yc = grid.y_range[0] + (iy + 0.5) * grid.voxel_size
        zc = 0.5 * (grid.z_range[0] + grid.z_range[1])
        coords[slot] = torch.tensor([iy, ix])
        for rank, (x, y, z, inten) in enumerate(buckets[key][:max_points]):
            features[slot, rank] = torch.tensor([x, y, z, inten, x - xc, y - yc, z - zc])
            mask[slot, rank] = True
    return features, mask, coords


def test_voxelize_matches_reference_loop():
    """The vectorised pillar scatter reproduces a slow per-point reference exactly."""
    grid = GridConfig()
    gen = torch.Generator().manual_seed(7)
    points = torch.zeros(400, 4)
    points[:, 0] = torch.rand(400, generator=gen) * 30 - 15
    points[:, 1] = torch.rand(400, generator=gen) * 30 - 15
    points[:, 2] = torch.rand(400, generator=gen) * 2
    points[:, 3] = torch.rand(400, generator=gen)

    max_points = 8
    got = voxelize(points, grid, None, max_points_per_pillar=max_points)
    exp_feat, exp_mask, exp_coords = _reference_voxelize(points, grid, max_points)

    assert got["features"].shape == exp_feat.shape
    assert torch.allclose(got["features"], exp_feat, atol=1e-5)
    assert torch.equal(got["mask"], exp_mask)
    assert torch.equal(got["coords"], exp_coords)
    assert int(got["num_points"].sum()) == int(exp_mask.sum())


def test_voxelize_reports_uncapped_point_counts():
    """``num_points`` counts every point in a pillar, even beyond the cap."""
    grid = GridConfig()
    cell = [0.25, 0.25, 0.5, 1.0]
    points = torch.tensor([[cell[0] + 1e-3 * i, cell[1], cell[2], cell[3]] for i in range(10)])
    v = voxelize(points, grid, None, max_points_per_pillar=4)
    assert v["num_pillars"].item() == 1
    assert v["num_points"].tolist() == [10]
    assert int(v["mask"].sum()) == 4


def test_voxelize_drops_out_of_range_points():
    """Points outside the *xy* grid never create a pillar.

    Pillars are columns spanning the whole z range, so z is deliberately not part
    of the range check - only the BEV footprint is.
    """
    grid = GridConfig()
    inside = torch.tensor([[0.0, 0.0, 0.5, 1.0]])
    outside = torch.tensor([[100.0, 0.0, 0.5, 1.0], [0.0, -100.0, 0.5, 1.0], [-16.5, 0.0, 0.5, 1.0]])
    mixed = torch.cat([inside, outside], dim=0)

    only_inside = voxelize(inside, grid, None, max_points_per_pillar=4)
    with_outside = voxelize(mixed, grid, None, max_points_per_pillar=4)
    assert only_inside["num_pillars"].item() == with_outside["num_pillars"].item() == 1
    assert torch.allclose(only_inside["features"], with_outside["features"])
    assert with_outside["point_index"].numel() == 1
    assert voxelize(outside, grid, None, max_points_per_pillar=4)["num_pillars"].item() == 0


def test_voxelize_is_batch_aware():
    """Points from different samples never share a pillar."""
    grid = GridConfig()
    pts = torch.tensor([[0.1, 0.1, 0.5, 1.0], [0.1, 0.1, 0.5, 1.0]])
    batch = torch.tensor([0, 1])
    v = voxelize(pts, grid, batch, max_points_per_pillar=4)
    assert v["num_pillars"].item() == 2
    assert v["coords"].shape == (2, 3)
    assert sorted(v["coords"][:, 0].tolist()) == [0, 1]


def test_voxelize_empty_input():
    """An empty cloud yields empty, correctly shaped tensors."""
    v = voxelize(torch.zeros(0, 4), GridConfig(), None, max_points_per_pillar=8)
    assert v["features"].shape == (0, 8, 7)
    assert v["coords"].shape == (0, 2)
    assert v["num_pillars"].item() == 0
