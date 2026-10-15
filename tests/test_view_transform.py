"""Learned view transform: projection round-trips and BEV pooling conserves mass."""

from __future__ import annotations

import math

import torch

from fusedbev3d.config import GridConfig
from fusedbev3d.data.synth import camera_extrinsics, camera_intrinsics
from fusedbev3d.models.camera_encoder import (
    bev_cell_from_ego,
    depth_bins,
    depth_to_ego,
    ego_to_pixel_depth,
    frustum_grid,
)
from fusedbev3d.models.camera_encoder import bev_pool as pool_fn


def test_projection_round_trip_recovers_the_point():
    """ego -> (pixel, depth) -> ego is the identity for points in front of the camera."""
    k = torch.tensor(camera_intrinsics(64, 192, 90.0), dtype=torch.float32)
    e = torch.tensor(camera_extrinsics(0), dtype=torch.float32)
    points = torch.tensor([[6.0, 0.5, 0.8], [12.0, -2.0, 1.4], [3.0, 1.2, 0.3]])
    u, v, depth = ego_to_pixel_depth(points, k, e)
    assert (depth > 0).all()
    back = depth_to_ego(u, v, depth, k, e)
    assert torch.allclose(back, points, atol=1e-4)


def test_known_point_lands_in_the_expected_bev_cell():
    """A known 3D point projects into the frustum entry of its own BEV cell."""
    grid = GridConfig()
    k = torch.tensor(camera_intrinsics(64, 192, 90.0), dtype=torch.float32)
    e = torch.tensor(camera_extrinsics(0), dtype=torch.float32)
    fh, fw, stride, bins = 32, 96, 2, 8
    d_min, d_max = 2.0, 30.0

    # The point sits exactly on a depth-bin centre (7.25 m for 8 bins over [2, 30])
    # and comfortably inside its BEV cell, so depth quantisation cannot move it out.
    point = torch.tensor([[7.25, 0.25, 1.6]])
    u, v, depth = ego_to_pixel_depth(point, k, e)
    valid, flat_expected = bev_cell_from_ego(point, grid)
    assert bool(valid[0])
    ix_expected = int((point[0, 0] - grid.x_range[0]) / grid.voxel_size)
    iy_expected = int((point[0, 1] - grid.y_range[0]) / grid.voxel_size)
    assert int(flat_expected[0]) == iy_expected * grid.bev_w + ix_expected

    g = frustum_grid(k, e, grid, d_min, d_max, bins, (fh, fw), stride)
    centres = depth_bins(d_min, d_max, bins)
    du = int(float(u[0]) / stride)
    dv = int(float(v[0]) / stride)
    dd = int(torch.argmin((centres - depth[0]).abs()).item())
    entry = dd * (fh * fw) + dv * fw + du
    assert bool(g["valid"][entry]), "the cell containing a visible point must be in the frustum"
    assert int(g["bev_index"][entry]) == iy_expected * grid.bev_w + ix_expected
    assert torch.allclose(g["points_ego"][entry], point[0], atol=0.5)


def test_frustum_depth_bins_are_uniform():
    """Depth bins are the centres of uniform bins over ``[d_min, d_max]``."""
    centres = depth_bins(2.0, 10.0, 4)
    assert torch.allclose(centres, torch.tensor([3.0, 5.0, 7.0, 9.0]))


def test_bev_pool_conserves_mass():
    """Scatter-add pooling conserves the total feature mass of valid entries."""
    grid = GridConfig()
    n, c = 256, 6
    gen = torch.Generator().manual_seed(3)
    feats = torch.randn(2, c, n, generator=gen)
    index = torch.randint(0, grid.bev_h * grid.bev_w, (2, n), generator=gen)
    valid = torch.rand(2, n, generator=gen) > 0.3
    out = pool_fn(feats, index, valid, grid, num_cameras=1)
    assert out.shape == (2, c, grid.bev_h, grid.bev_w)
    expected = feats[0][:, valid[0]].sum() + feats[1][:, valid[1]].sum()
    assert torch.allclose(out.sum(), expected, atol=1e-4)

    # invalid entries contribute nothing
    out_none = pool_fn(feats, index, torch.zeros_like(valid), grid, num_cameras=1)
    assert out_none.abs().sum() == 0


def test_bev_pool_sums_multiple_cameras_into_one_grid():
    """Two cameras writing to the same cell accumulate rather than overwrite."""
    grid = GridConfig()
    feats = torch.ones(2, 1, 1)
    index = torch.zeros(2, 1, dtype=torch.long)
    valid = torch.ones(2, 1, dtype=torch.bool)
    out = pool_fn(feats, index, valid, grid, num_cameras=2)
    assert out[0, 0, 0, 0].item() == 2.0


def test_camera_rig_geometry_is_sane():
    """The three rig cameras look forward, left-forward and right-forward."""
    ks = camera_intrinsics(64, 192, 90.0)
    assert math.isclose(float(ks[0, 0]), 96.0, rel_tol=1e-6)
    front = camera_extrinsics(0)
    point = torch.tensor([[5.0, 0.0, 0.0]])
    u, v, depth = ego_to_pixel_depth(point, torch.tensor(ks, dtype=torch.float32), torch.tensor(front, dtype=torch.float32))
    assert depth[0] > 0
    assert math.isclose(float(u[0]), 96.0, abs_tol=1e-3)  # image centre
