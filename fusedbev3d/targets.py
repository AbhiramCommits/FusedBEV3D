"""Target building for the CenterHead and the occupancy head.

The heatmap follows the CenterNet recipe: a radius derived from the box footprint
is rendered as a (max-accumulated) 2D gaussian around the projected box centre,
and the regression map is written only at the integer cell containing that centre
so the network learns the sub-pixel offset itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from fusedbev3d.config import GridConfig
from fusedbev3d.models.heads import REG_LAYOUT

__all__ = [
    "gaussian_radius",
    "draw_gaussian",
    "build_center_targets",
    "build_occupancy_targets",
    "targets_from_batch",
]


def gaussian_radius(det_size: tuple[float, float], min_overlap: float = 0.7) -> float:
    """CenterNet gaussian radius for a detection of size ``det_size`` cells.

    Args:
        det_size: ``(height, width)`` of the object footprint in cells.
        min_overlap: Minimum IoU the gaussian should preserve.

    Returns:
        The radius in cells (floored by the caller).
    """
    height, width = det_size
    a1, b1, c1 = 1.0, height + width, width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1**2 - 4 * a1 * c1, 0.0))) / 2
    a2, b2, c2 = 4.0, 2 * (height + width), (1 - min_overlap) * width * height / (1 + min_overlap)
    r2 = (b2 - math.sqrt(max(b2**2 - 4 * a2 * c2, 0.0))) / 2
    a3 = 4 * min_overlap
    b3, c3 = -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3**2 - 4 * a3 * c3, 0.0))) / (2 * a3)
    return max(r1, r2, r3)


def draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int, k: float = 1.0) -> None:
    """Max-accumulate a clipped 2D gaussian into ``heatmap`` in place.

    Args:
        heatmap: ``(H, W)`` float tensor modified in place.
        cx: Integer cell column of the peak.
        cy: Integer cell row of the peak.
        radius: Gaussian radius in cells.
        k: Peak value.
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    x = torch.arange(-radius, radius + 1, device=heatmap.device, dtype=heatmap.dtype)
    g = torch.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / (2 * sigma * sigma))
    left, right = min(cx, radius), min(heatmap.shape[1] - cx, radius + 1)
    top, bottom = min(cy, radius), min(heatmap.shape[0] - cy, radius + 1)
    if right <= -left or bottom <= -top:
        return
    masked = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    values = g[radius - top : radius + bottom, radius - left : radius + right] * k
    heatmap[cy - top : cy + bottom, cx - left : cx + right] = torch.maximum(masked, values)


def build_center_targets(
    boxes: Sequence[torch.Tensor] | torch.Tensor,
    grid: GridConfig,
    num_classes: int,
    max_objects: int = 64,
) -> dict[str, torch.Tensor]:
    """Render heatmap + regression targets for a batch of box sets.

    Args:
        boxes: Either a list of ``(Mi, 11)`` tensors or a padded ``(B, M, 11)`` tensor.
            Rows are ``x y z l w h yaw vx vy class track_id``.
        grid: Grid geometry.
        num_classes: Number of object classes ``K``.
        max_objects: Safety cap on the number of objects rendered per sample.

    Returns:
        ``{"heatmap" (B,K,H,W), "reg" (B,10,H,W), "mask" (B,H,W) bool,
        "num_objects" (B,)}``.
    """
    if isinstance(boxes, torch.Tensor):
        box_list = [boxes[i] for i in range(boxes.shape[0])]
    else:
        box_list = list(boxes)
    ref = box_list[0] if box_list else torch.zeros(0, 11)
    device = ref.device
    dtype = ref.dtype if ref.numel() else torch.float32
    b = len(box_list)
    heatmap = torch.zeros(b, num_classes, grid.bev_h, grid.bev_w, device=device, dtype=dtype)
    reg = torch.zeros(b, 10, grid.bev_h, grid.bev_w, device=device, dtype=dtype)
    mask = torch.zeros(b, grid.bev_h, grid.bev_w, dtype=torch.bool, device=device)
    counts = torch.zeros(b, dtype=torch.long, device=device)

    for i, bx in enumerate(box_list):
        if bx.numel() == 0:
            continue
        bx = bx[:max_objects]
        counts[i] = int(bx.shape[0])
        cxi = torch.floor((bx[:, 0] - grid.x_range[0]) / grid.voxel_size).long()
        cyi = torch.floor((bx[:, 1] - grid.y_range[0]) / grid.voxel_size).long()
        off_x = (bx[:, 0] - grid.x_range[0]) / grid.voxel_size - cxi.to(dtype)
        off_y = (bx[:, 1] - grid.y_range[0]) / grid.voxel_size - cyi.to(dtype)
        inside = (cxi >= 0) & (cxi < grid.bev_w) & (cyi >= 0) & (cyi < grid.bev_h)
        for j in range(bx.shape[0]):
            if not bool(inside[j]):
                continue
            cls = int(bx[j, 9].item())
            cx, cy = int(cxi[j].item()), int(cyi[j].item())
            major = max(float(bx[j, 3]), float(bx[j, 4])) / grid.voxel_size
            minor = min(float(bx[j, 3]), float(bx[j, 4])) / grid.voxel_size
            radius = max(int(gaussian_radius((major, minor))), 2)
            draw_gaussian(heatmap[i, cls], cx, cy, radius)
            yaw = float(bx[j, 6])
            reg[i, :, cy, cx] = torch.tensor(
                [
                    off_x[j],
                    off_y[j],
                    bx[j, 2],
                    torch.log(bx[j, 3].clamp(min=1e-2)),
                    torch.log(bx[j, 4].clamp(min=1e-2)),
                    torch.log(bx[j, 5].clamp(min=1e-2)),
                    math.sin(yaw),
                    math.cos(yaw),
                    bx[j, 7],
                    bx[j, 8],
                ],
                device=device,
                dtype=dtype,
            )
            mask[i, cy, cx] = True
    return {"heatmap": heatmap, "reg": reg, "mask": mask, "num_objects": counts}


def build_occupancy_targets(occupancy: torch.Tensor) -> torch.Tensor:
    """Return the ``(B, Z, H, W)`` long occupancy labels used by the CE loss.

    Args:
        occupancy: ``(B, Z, H, W)`` labels in ``{0=free, 1=occupied, 2=unknown}``.

    Returns:
        The same tensor as ``torch.long``.
    """
    return occupancy.long()


def targets_from_batch(batch: dict[str, Any], grid: GridConfig, num_classes: int) -> dict[str, Any]:
    """Build every target the model needs from a collated batch.

    Args:
        batch: Collated batch from :func:`fusedbev3d.data.synth.collate_fn`.
        grid: Grid geometry.
        num_classes: Number of object classes.

    Returns:
        ``{"center": {...}, "occupancy": (B,Z,H,W)}`` ready for
        :meth:`fusedbev3d.models.fusedbev3d.FusedBEV3D.loss`.
    """
    return {
        "center": build_center_targets(batch["boxes"], grid, num_classes),
        "occupancy": build_occupancy_targets(batch["occupancy"]),
    }


# re-exported for callers that want the canonical slice layout
REG_SLICES = REG_LAYOUT
