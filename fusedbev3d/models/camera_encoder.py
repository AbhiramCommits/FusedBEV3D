"""Camera branch: a Lift-Splat-Shoot style learned view transform.

A lightweight strided-conv CNN produces per-pixel context features and a discrete
depth distribution over ``D`` bins.  Their outer product yields a frustum point
cloud which is pooled (bucketed sum) onto the *same* BEV grid the lidar branch
writes to, which is what makes late fusion trivially well defined.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fusedbev3d.config import GridConfig, ModelConfig
from fusedbev3d.data.synth import ego_to_bev_cell

__all__ = [
    "depth_bins",
    "depth_to_ego",
    "ego_to_pixel_depth",
    "bev_cell_from_ego",
    "frustum_grid",
    "bev_pool",
    "CameraEncoder",
]


def depth_bins(depth_min: float, depth_max: float, num_bins: int, device: torch.device | None = None) -> torch.Tensor:
    """Return the ``(D,)`` centres of uniformly spaced depth bins in metres."""
    edges = torch.linspace(depth_min, depth_max, num_bins + 1, device=device)
    return 0.5 * (edges[:-1] + edges[1:])


def depth_to_ego(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Back-project pixel coordinates + depth into the ego frame.

    Args:
        u: ``(N,)`` pixel column coordinates.
        v: ``(N,)`` pixel row coordinates.
        depth: ``(N,)`` camera-frame z in metres.
        intrinsics: ``(3, 3)`` pinhole matrix.
        extrinsics: ``(4, 4)`` camera-from-ego transform.

    Returns:
        ``(N, 3)`` ego-frame points.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth
    cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    r = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    # p_cam = R p_ego + t  =>  p_ego = R^T (p_cam - t)
    return (cam - t) @ r


def ego_to_pixel_depth(
    points_ego: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project ego-frame points to pixel coordinates plus camera depth.

    Args:
        points_ego: ``(N, 3)`` ego-frame points.
        intrinsics: ``(3, 3)`` pinhole matrix.
        extrinsics: ``(4, 4)`` camera-from-ego transform.

    Returns:
        ``(u, v, depth)`` each ``(N,)``; ``depth <= 0`` marks points behind the camera.
    """
    r = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    cam = points_ego @ r.t() + t
    depth = cam[:, 2]
    safe = torch.where(depth.abs() < 1e-6, torch.full_like(depth, 1e-6), depth)
    u = cam[:, 0] / safe * intrinsics[0, 0] + intrinsics[0, 2]
    v = cam[:, 1] / safe * intrinsics[1, 1] + intrinsics[1, 2]
    return u, v, depth


def bev_cell_from_ego(points_ego: torch.Tensor, grid: GridConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(valid, flat_index)`` mapping ego points to flat BEV cells.

    Args:
        points_ego: ``(N, 3)`` ego-frame points.
        grid: Grid geometry.

    Returns:
        ``valid`` ``(N,)`` boolean in-grid mask, ``flat_index`` ``(N,)`` long
        ``iy * bev_w + ix`` (undefined where ``valid`` is false).
    """
    ix, iy = ego_to_bev_cell(points_ego, grid)
    valid = (ix >= 0) & (ix < grid.bev_w) & (iy >= 0) & (iy < grid.bev_h)
    flat = (iy.clamp(0, grid.bev_h - 1) * grid.bev_w + ix.clamp(0, grid.bev_w - 1)).long()
    return valid, flat


def frustum_grid(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    grid: GridConfig,
    depth_min: float,
    depth_max: float,
    num_bins: int,
    feature_hw: tuple[int, int],
    feature_stride: int = 1,
) -> dict[str, torch.Tensor]:
    """Build the LSS frustum for one camera.

    Every feature-map pixel is back-projected at each of ``D`` depth-bin centres,
    the resulting camera-frame points are moved into the ego frame and bucketed
    into the shared BEV grid.

    Args:
        intrinsics: ``(3, 3)`` pinhole matrix in *image* pixel units.
        extrinsics: ``(4, 4)`` camera-from-ego transform.
        grid: Grid geometry.
        depth_min: Nearest bin edge in metres.
        depth_max: Farthest bin edge in metres.
        num_bins: Number of depth bins ``D``.
        feature_hw: ``(h, w)`` of the feature map (not the raw image).
        feature_stride: Backbone output stride; pixel centres are sampled at
            ``(i + 0.5) * feature_stride`` in image coordinates.

    Returns:
        ``{"points_ego" (N,3), "bev_index" (N,), "valid" (N,), "depths" (D,)}``
        with ``N = D * h * w`` and the flattened ordering ``(d, v, u)``.
    """
    h, w = feature_hw
    dev = intrinsics.device
    depths = depth_bins(depth_min, depth_max, num_bins, device=dev)
    vs = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5) * feature_stride
    us = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) * feature_stride
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
    d = depths.shape[0]
    u = grid_u.reshape(-1).repeat(d)
    v = grid_v.reshape(-1).repeat(d)
    depth = depths.repeat_interleave(h * w)
    pts = depth_to_ego(u, v, depth, intrinsics.to(dev), extrinsics.to(dev))
    valid, flat = bev_cell_from_ego(pts, grid)
    return {"points_ego": pts, "bev_index": flat, "valid": valid, "depths": depths}


def bev_pool(
    features: torch.Tensor,
    bev_index: torch.Tensor,
    valid: torch.Tensor,
    grid: GridConfig,
    num_cameras: int,
) -> torch.Tensor:
    """Bucketed-sum pooling of frustum features onto the BEV grid.

    Args:
        features: ``(B * S, C, N)`` per-camera frustum features.
        bev_index: ``(B * S, N)`` flat BEV indices in ``[0, H*W)``.
        valid: ``(B * S, N)`` boolean mask of in-grid frustum points.
        grid: Grid geometry.
        num_cameras: ``S``, the number of cameras per sample.

    Returns:
        ``(B, C, H, W)`` pooled BEV features.  Pooling is an exact scatter-add, so
        the total mass of the valid input features is conserved.
    """
    bs, c, _ = features.shape
    b = bs // num_cameras
    hw = grid.bev_h * grid.bev_w
    index = torch.where(valid, bev_index, torch.zeros_like(bev_index))
    src = torch.where(valid.unsqueeze(1), features, torch.zeros_like(features))
    sample = torch.arange(bs, device=features.device) // num_cameras
    chan = torch.arange(c, device=features.device)
    target = (sample[:, None, None] * c + chan[None, :, None]) * hw + index[:, None, :]
    out = torch.zeros(b * c * hw, device=features.device, dtype=features.dtype)
    out = out.index_add_(0, target.reshape(-1), src.reshape(-1))
    return out.view(b, c, grid.bev_h, grid.bev_w)


class CameraEncoder(nn.Module):
    """LSS camera branch: CNN features + depth distribution -> BEV feature map.

    Args:
        grid: Grid geometry.
        cfg: Model configuration (``camera_context``, ``depth_bins``, depths).
        image_hw: ``(h, w)`` of the input images.
        stride: Backbone output stride (default ``2``).
    """

    def __init__(
        self,
        grid: GridConfig,
        cfg: ModelConfig,
        image_hw: tuple[int, int],
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.cfg = cfg
        self.image_hw = image_hw
        self.stride = stride
        c = cfg.camera_context
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(c, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, cfg.depth_bins + c, 1),
        )
        self.out_channels = c

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Lift multi-camera images into a ``(B, C, H, W)`` BEV feature map.

        Args:
            batch: Collated batch with ``images (B,S,3,h,w)``, ``intrinsics (B,S,3,3)``
                and ``extrinsics (B,S,4,4)``.

        Returns:
            ``{"bev", "frustum_features", "frustum_grid"}``.
        """
        images = batch["images"]
        b, s, ch, h, w = images.shape
        feat = self.backbone(images.reshape(b * s, ch, h, w))
        raw = self.depth_head(feat)
        d = self.cfg.depth_bins
        depth_logits = raw[:, :d]
        context = raw[:, d:]
        depth_prob = torch.softmax(depth_logits, dim=1)
        # outer product -> (B*S, C, D, h', w') -> (B*S, C, D*h'*w')
        frustum = (context.unsqueeze(2) * depth_prob.unsqueeze(1)).reshape(
            b * s, self.out_channels, -1
        )
        fh, fw = feat.shape[-2:]
        index = torch.zeros(b * s, d * fh * fw, dtype=torch.long, device=images.device)
        valid = torch.zeros(b * s, d * fh * fw, dtype=torch.bool, device=images.device)
        first: dict[str, torch.Tensor] | None = None
        for i in range(b * s):
            g = frustum_grid(
                batch["intrinsics"][i // s, i % s],
                batch["extrinsics"][i // s, i % s],
                self.grid,
                self.cfg.depth_min,
                self.cfg.depth_max,
                d,
                (fh, fw),
                self.stride,
            )
            index[i] = g["bev_index"]
            valid[i] = g["valid"]
            if first is None:
                first = g
        bev = bev_pool(frustum, index, valid, self.grid, s)
        return {"bev": bev, "frustum_features": frustum, "frustum_grid": first}
