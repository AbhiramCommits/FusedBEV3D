"""Target building: gaussian heatmap peaks and visibility-derived occupancy labels."""

from __future__ import annotations

import numpy as np
import torch

from fusedbev3d.config import DataConfig, GridConfig
from fusedbev3d.data.synth import build_sequence, occupancy_from_points
from fusedbev3d.targets import build_center_targets, build_occupancy_targets, gaussian_radius


def _cell_of(x: float, y: float, grid: GridConfig) -> tuple[int, int]:
    px = int((x - grid.x_range[0]) / grid.voxel_size)
    py = int((y - grid.y_range[0]) / grid.voxel_size)
    return py, px


def test_heatmap_peak_sits_at_the_box_centre():
    """Every object renders its heatmap peak exactly at its projected centre cell."""
    grid = GridConfig()
    boxes = torch.tensor(
        [
            [3.2, -1.4, 1.0, 4.2, 1.8, 1.6, 0.3, 1.0, 0.0, 0, 7.0],
            [-6.7, 5.3, 0.9, 0.7, 0.7, 1.75, -1.2, 0.0, 0.0, 1, 8.0],
            [10.4, 9.1, 0.85, 1.8, 0.6, 1.7, 2.4, 3.0, 1.0, 2, 9.0],
        ]
    )
    t = build_center_targets([boxes], grid, 3)
    for row in boxes:
        py, px = _cell_of(float(row[0]), float(row[1]), grid)
        assert t["mask"][0, py, px], "the centre cell must be flagged positive"
        assert t["heatmap"][0, int(row[9]), py, px] == 1.0
        # the peak of each class heatmap is that cell
        peak = int(t["heatmap"][0, int(row[9])].argmax())
        assert (peak // grid.bev_w, peak % grid.bev_w) == (py, px)


def test_regression_targets_hold_the_box_values():
    """Regression channels store offset, z, log-dims, sin/cos yaw and velocity."""
    grid = GridConfig()
    box = torch.tensor([[3.2, -1.4, 1.0, 4.2, 1.8, 1.6, 0.3, 1.5, -0.5, 0, 7.0]])
    t = build_center_targets([box], grid, 3)
    py, px = _cell_of(3.2, -1.4, grid)
    reg = t["reg"][0, :, py, px]
    assert torch.allclose(reg[3:6], torch.log(box[0, 3:6]), atol=1e-5)
    assert torch.isclose(reg[6], torch.tensor(0.3).sin(), atol=1e-5)
    assert torch.isclose(reg[7], torch.tensor(0.3).cos(), atol=1e-5)
    assert torch.allclose(reg[8:10], box[0, 7:9], atol=1e-5)
    assert torch.isclose(reg[2], box[0, 2], atol=1e-5)


def test_out_of_grid_boxes_are_skipped():
    """Boxes outside the BEV grid contribute no target at all."""
    grid = GridConfig()
    inside = torch.tensor([[3.2, -1.4, 1.0, 4.2, 1.8, 1.6, 0.3, 1.0, 0.0, 0, 1.0]])
    outside = torch.tensor([[500.0, 0.0, 1.0, 4.2, 1.8, 1.6, 0.0, 0.0, 0.0, 1, 2.0]])
    only_in = build_center_targets([inside], grid, 3)
    both = build_center_targets([inside, outside], grid, 3)
    assert both["num_objects"].tolist() == [1, 1]
    assert torch.equal(only_in["heatmap"], both["heatmap"][:1])
    assert both["mask"][1].sum() == 0


def test_gaussian_radius_grows_with_box_size():
    """Larger footprints get a larger gaussian radius."""
    assert gaussian_radius((2.0, 2.0)) < gaussian_radius((10.0, 4.0))
    assert gaussian_radius((2.0, 2.0)) > 0


def test_occupancy_targets_are_long_labels():
    """``build_occupancy_targets`` just casts to the CE-ready dtype."""
    occ = torch.zeros(2, 5, 64, 64, dtype=torch.uint8)
    out = build_occupancy_targets(occ)
    assert out.dtype == torch.long and out.shape == occ.shape


def test_occupancy_labels_follow_ray_visibility():
    """A single synthetic return carves free space and stamps occupied/unknown correctly."""
    grid = GridConfig()  # 0.5 m cells, 64x64 BEV, z in [-0.5, 2.0]
    cfg = DataConfig()
    # one return at x=5 m, on the ground, coming from the lidar at z=1.73
    ground_z = 0.0
    dz = ground_z - 1.73
    rng = (5.0**2 + dz**2) ** 0.5
    point = torch.tensor([[5.0, 0.0, ground_z, 1.0]])
    labels = occupancy_from_points(point.numpy(), grid, cfg)
    del rng

    def voxel(x, y, z):
        ix = int((x - grid.x_range[0]) / grid.voxel_size)
        iy = int((y - grid.y_range[0]) / grid.voxel_size)
        iz = int((z - grid.z_range[0]) / grid.z_voxel_size)
        return int(labels[iz, iy, ix])

    assert voxel(5.0, 0.0, ground_z) == 1, "the return voxel is occupied"
    assert voxel(3.0, 0.0, 0.75) == 0, "a voxel on the ray is carved free"
    assert voxel(-10.0, -10.0, 1.5) == 2, "an untouched voxel stays unknown"


def test_generated_occupancy_has_all_three_classes():
    """Real generated data contains free, occupied and unknown voxels."""
    cfg = DataConfig()
    frames = build_sequence(0, GridConfig(), cfg, "train")
    stacked = np.stack([f["occupancy"] for f in frames])
    counts = np.bincount(stacked.ravel(), minlength=3)
    assert counts[0] > 0 and counts[1] > 0 and counts[2] > 0
    # unknown must be a strict subset: not everything can be observed
    assert counts[2] < counts.sum()
