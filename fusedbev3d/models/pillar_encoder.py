"""PointPillars-style lidar branch.

Points are scattered into BEV pillars (columns of the shared grid), a small
PointNet encodes each pillar, and the pillar features are scattered back onto a
dense BEV feature map.  Everything is vectorised and scatters with
:meth:`torch.Tensor.index_add_` / :meth:`torch.Tensor.index_put_`, so no
``torch_scatter`` extension is required.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fusedbev3d.config import GridConfig, ModelConfig
from fusedbev3d.data.synth import ego_to_bev_cell

__all__ = ["voxelize", "PillarEncoder", "PILLAR_FEATURES"]

#: ``x, y, z, intensity, x - xc, y - yc, z - zc``
PILLAR_FEATURES = 7


def voxelize(
    points: torch.Tensor,
    grid: GridConfig,
    point_batch: torch.Tensor | None = None,
    max_points_per_pillar: int = 32,
    max_pillars: int = 0,
) -> dict[str, torch.Tensor]:
    """Scatter points into pillars and build padded per-pillar feature tensors.

    Points are grouped by their ``(sample, iy, ix)`` pillar key, kept in input
    order inside each pillar, truncated to ``max_points_per_pillar`` and packed
    into a dense ``(P, max_points_per_pillar, 7)`` tensor.  No Python loop over
    points or pillars is used anywhere.

    Args:
        points: ``(N, >=4)`` float tensor of ``x, y, z, intensity[, ...]`` in the
            ego frame.  Extra columns are ignored.
        grid: Grid geometry; points outside the grid are dropped.
        point_batch: Optional ``(N,)`` long tensor giving the sample index of every
            point.  When given, pillars from different samples never merge and the
            returned ``coords`` gain a leading sample column.
        max_points_per_pillar: Point cap per pillar (excess points are dropped).
        max_pillars: Pillar cap, keeping the pillars with the most points
            (``0`` keeps every occupied pillar).

    Returns:
        A dictionary with

        ``features``      ``(P, M, 7)`` padded pillar features
        ``mask``          ``(P, M)`` boolean validity mask
        ``coords``        ``(P, 2)`` or ``(P, 3)`` ``(iy, ix)`` or ``(sample, iy, ix)``
        ``num_points``    ``(P,)`` number of points per pillar *before* truncation
        ``point_index``   ``(K,)`` index into ``points`` of every retained point
        ``pillar_index``  ``(K,)`` pillar slot of every retained point
        ``point_rank``    ``(K,)`` rank of every retained point inside its pillar
        ``num_pillars``   scalar tensor with ``P``
    """
    device = points.device
    n = points.shape[0]
    hw = grid.bev_h * grid.bev_w
    empty: dict[str, Any] = {
        "features": torch.zeros(0, max_points_per_pillar, PILLAR_FEATURES, device=device),
        "mask": torch.zeros(0, max_points_per_pillar, dtype=torch.bool, device=device),
        "coords": torch.zeros(0, 3 if point_batch is not None else 2, dtype=torch.long, device=device),
        "num_points": torch.zeros(0, dtype=torch.long, device=device),
        "point_index": torch.zeros(0, dtype=torch.long, device=device),
        "pillar_index": torch.zeros(0, dtype=torch.long, device=device),
        "point_rank": torch.zeros(0, dtype=torch.long, device=device),
        "num_pillars": torch.zeros((), dtype=torch.long, device=device),
    }
    if n == 0:
        return empty

    pts = points[:, :4]
    ix, iy = ego_to_bev_cell(pts[:, :3], grid)
    inside = (ix >= 0) & (ix < grid.bev_w) & (iy >= 0) & (iy < grid.bev_h)
    if not bool(inside.any()):
        return empty

    sample = point_batch if point_batch is not None else torch.zeros_like(ix)
    n_samples = int(sample.max().item()) + 1 if n else 1
    key = sample * hw + iy * grid.bev_w + ix
    key = torch.where(inside, key, torch.full_like(key, -1))

    order = torch.argsort(key, stable=True)
    order = order[key[order] >= 0]
    if order.numel() == 0:
        return empty

    sorted_key = key[order]
    n_keys = n_samples * hw
    counts = torch.bincount(sorted_key, minlength=n_keys)

    occupied = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
    if max_pillars and occupied.numel() > max_pillars:
        top = torch.argsort(counts[occupied], descending=True)[:max_pillars]
        occupied = occupied[top]
    lut = torch.full((n_keys,), -1, dtype=torch.long, device=device)
    lut[occupied] = torch.arange(occupied.numel(), device=device)

    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(order.numel(), device=device) - starts[sorted_key]
    pillar_slot = lut[sorted_key]
    keep = (pillar_slot >= 0) & (rank < max_points_per_pillar)

    point_index = order[keep]
    pillar_index = pillar_slot[keep]
    point_rank = rank[keep]

    p = int(occupied.numel())
    m = int(max_points_per_pillar)
    x, y, z, inten = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    cell_x, cell_y = ego_to_bev_cell(pts[:, :3], grid)
    kx = cell_x[point_index].to(pts.dtype)
    ky = cell_y[point_index].to(pts.dtype)
    xc = grid.x_range[0] + (kx + 0.5) * grid.voxel_size
    yc = grid.y_range[0] + (ky + 0.5) * grid.voxel_size
    zc = 0.5 * (grid.z_range[0] + grid.z_range[1])
    raw = torch.stack(
        [x[point_index], y[point_index], z[point_index], inten[point_index], x[point_index] - xc,
         y[point_index] - yc, z[point_index] - zc],
        dim=1,
    )

    features = torch.zeros(p * m, PILLAR_FEATURES, dtype=pts.dtype, device=device)
    features.index_put_((pillar_index * m + point_rank,), raw)
    mask_flat = torch.zeros(p * m, dtype=torch.bool, device=device)
    mask_flat.index_put_((pillar_index * m + point_rank,), torch.ones_like(point_rank, dtype=torch.bool))

    iy_all = (occupied % hw) // grid.bev_w
    ix_all = occupied % grid.bev_w
    sample_all = occupied // hw
    coords = (
        torch.stack([sample_all, iy_all, ix_all], dim=1)
        if point_batch is not None
        else torch.stack([iy_all, ix_all], dim=1)
    )

    return {
        "features": features.view(p, m, PILLAR_FEATURES),
        "mask": mask_flat.view(p, m),
        "coords": coords,
        "num_points": counts[occupied],
        "point_index": point_index,
        "pillar_index": pillar_index,
        "point_rank": point_rank,
        "num_pillars": torch.tensor(p, dtype=torch.long, device=device),
    }


class PillarEncoder(nn.Module):
    """PointPillars lidar backbone producing a dense BEV feature map.

    Args:
        grid: Grid geometry (pillar layout and output resolution).
        cfg: Model configuration (``pillar_feat``, caps).
        in_features: Number of per-point input features (default :data:`PILLAR_FEATURES`).
    """

    def __init__(
        self,
        grid: GridConfig,
        cfg: ModelConfig,
        in_features: int = PILLAR_FEATURES,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.cfg = cfg
        self.in_features = in_features
        c = cfg.pillar_feat
        self.pfn = nn.Sequential(
            nn.Linear(in_features, c),
            nn.LayerNorm(c),
            nn.ReLU(inplace=True),
            nn.Linear(c, c),
            nn.LayerNorm(c),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True))

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode a collated batch into a ``(B, C, H, W)`` lidar BEV feature map.

        Args:
            batch: Collated batch from :func:`fusedbev3d.data.synth.collate_fn`.

        Returns:
            ``{"bev", "pillar_features", "coords", "num_pillars"}``.
        """
        points = batch["points"]
        point_batch = batch.get("point_batch")
        n_samples = int(batch["num_boxes"].shape[0])
        v = voxelize(
            points,
            self.grid,
            point_batch,
            max_points_per_pillar=self.cfg.max_points_per_pillar,
            max_pillars=self.cfg.max_pillars,
        )
        c = self.cfg.pillar_feat
        bev = torch.zeros(n_samples, c, self.grid.bev_h * self.grid.bev_w, device=points.device, dtype=points.dtype)
        p = int(v["num_pillars"].item())
        feats: torch.Tensor | None = None
        if p:
            feats = self.pfn(v["features"])
            feats = (feats * v["mask"].unsqueeze(-1)).max(dim=1).values
            feats = self.head(feats)  # (P, C)
            coords = v["coords"]
            sample = coords[:, 0] if coords.shape[1] == 3 else torch.zeros_like(coords[:, 0])
            flat = coords[:, -2] * self.grid.bev_w + coords[:, -1]
            # scatter (P, C) into (B, C, H*W) with a single index_add_ over a flat buffer
            ch = torch.arange(c, device=points.device)
            target = (sample[:, None] * c + ch[None, :]) * (self.grid.bev_h * self.grid.bev_w) + flat[:, None]
            bev = bev.reshape(-1).index_add_(0, target.reshape(-1), feats.reshape(-1)).view(
                n_samples, c, self.grid.bev_h, self.grid.bev_w
            )
        return {
            "bev": bev,
            "pillar_features": feats,
            "coords": v["coords"],
            "num_pillars": v["num_pillars"],
        }
