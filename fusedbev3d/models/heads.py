"""Task heads: a CenterNet-style detection head and a 3-class occupancy head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusedbev3d.config import GridConfig, ModelConfig

__all__ = ["CenterHead", "OccupancyHead", "sigmoid_focal_loss", "REG_LAYOUT"]

#: Slice layout of the ``reg_channels`` regression map.
REG_LAYOUT: dict[str, slice] = {
    "offset": slice(0, 2),  # sub-pixel x, y offset in cells
    "z": slice(2, 3),  # centre height in metres
    "dims": slice(3, 6),  # log length, width, height
    "rot": slice(6, 8),  # sin(yaw), cos(yaw)
    "vel": slice(8, 10),  # vx, vy in m/s
}


def sigmoid_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss on raw heatmap logits.

    Args:
        logits: ``(B, C, H, W)`` raw heatmap logits.
        target: ``(B, C, H, W)`` gaussian peaks in ``[0, 1]``.

    Returns:
        Scalar loss normalised by the number of peak cells.
    """
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pos = target >= 1.0 - 1e-4
    pos_loss = -((1 - p) ** 2) * torch.log(p) * target
    neg_loss = -((1 - target) ** 4) * (p**2) * torch.log(1 - p) * (1 - target)
    num_pos = pos.float().sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


class CenterHead(nn.Module):
    """Per-class BEV heatmap + dense regression maps.

    Regression channels follow :data:`REG_LAYOUT`: sub-pixel offset, height ``z``,
    log dimensions, ``sin/cos`` yaw and planar velocity.

    Args:
        cfg: Model configuration.
        in_channels: Channel width of the incoming fused BEV features.
    """

    def __init__(self, cfg: ModelConfig, in_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        w = cfg.head_width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w, 3, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(w, cfg.num_classes, 1)
        self.regression = nn.Conv2d(w, cfg.reg_channels, 1)
        nn.init.constant_(self.heatmap.bias, -math.log((1 - 0.01) / 0.01))
        nn.init.zeros_(self.regression.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict heatmap logits and regression maps.

        Args:
            x: ``(B, C, H, W)`` fused BEV features.

        Returns:
            ``{"heatmap" (B,K,H,W), "reg" (B,reg_channels,H,W)}``.
        """
        f = self.stem(x)
        return {"heatmap": self.heatmap(f), "reg": self.regression(f)}

    def decode(
        self,
        outputs: dict[str, torch.Tensor],
        grid: GridConfig,
        k: int = 100,
        score_threshold: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Turn heatmap peaks into oriented 3D boxes via 3x3 max-pool NMS.

        Args:
            outputs: Result of :meth:`forward`.
            grid: Grid geometry used to map cells back to metres.
            k: Maximum number of detections to keep per sample.
            score_threshold: Minimum sigmoid score for a peak to be decoded.

        Returns:
            ``{"boxes" (B,k,9), "scores" (B,k), "classes" (B,k), "cells" (B,k,2)}``
            where each box is ``x, y, z, l, w, h, yaw, vx, vy``.
        """
        logits = outputs["heatmap"]
        reg = outputs["reg"]
        b, c, hh, ww = logits.shape
        scores_map = torch.sigmoid(logits)
        pooled = F.max_pool2d(scores_map, 3, stride=1, padding=1)
        peaks = (pooled == scores_map).float() * scores_map
        flat = peaks.reshape(b, c * hh * ww)
        k = min(k, flat.shape[-1])
        top_scores, top_idx = torch.topk(flat, k, dim=-1)

        cls = torch.div(top_idx, hh * ww, rounding_mode="floor")
        pix = top_idx % (hh * ww)
        py = torch.div(pix, ww, rounding_mode="floor")
        px = pix % ww

        off = self._gather(reg[:, REG_LAYOUT["offset"]], py, px, ww)  # (B,k,2)
        z = self._gather(reg[:, REG_LAYOUT["z"]], py, px, ww)[..., 0]
        dims = self._gather(reg[:, REG_LAYOUT["dims"]], py, px, ww)
        rot = self._gather(reg[:, REG_LAYOUT["rot"]], py, px, ww)
        vel = self._gather(reg[:, REG_LAYOUT["vel"]], py, px, ww)

        x = (px.to(reg.dtype) + 0.5 + off[..., 0]) * grid.voxel_size + grid.x_range[0]
        y = (py.to(reg.dtype) + 0.5 + off[..., 1]) * grid.voxel_size + grid.y_range[0]
        dims = torch.exp(dims.clamp(-3.0, 3.0))
        yaw = torch.atan2(rot[..., 0], rot[..., 1])
        boxes = torch.stack([x, y, z, dims[..., 0], dims[..., 1], dims[..., 2], yaw, vel[..., 0], vel[..., 1]], dim=-1)
        keep = top_scores >= score_threshold
        return {
            "boxes": boxes,
            "scores": top_scores,
            "classes": cls,
            "cells": torch.stack([py, px], dim=-1),
            "keep": keep,
        }

    @staticmethod
    def _gather(feature: torch.Tensor, py: torch.Tensor, px: torch.Tensor, width: int) -> torch.Tensor:
        """Gather ``(B, C, H, W)`` features at ``(py, px)`` returning ``(B, k, C)``."""
        b, c, _, _ = feature.shape
        flat = feature.reshape(b, c, -1)
        idx = (py * width + px).unsqueeze(1).expand(-1, c, -1)
        return flat.gather(2, idx).transpose(1, 2)


class OccupancyHead(nn.Module):
    """Dense 3-class occupancy logits over the shared BEV grid.

    The BEV channels are reshaped to ``Z x num_classes``: semantically the head
    predicts, for every BEV cell, a full vertical column of free/occupied/unknown
    logits conditioned on the fused BEV context.

    Args:
        cfg: Model configuration (``num_z`` comes from the grid).
        grid: Grid geometry.
        in_channels: Channel width of the incoming fused BEV features.
    """

    def __init__(self, cfg: ModelConfig, grid: GridConfig, in_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = grid
        width = cfg.head_width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(width, grid.num_z * cfg.occupancy_classes, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict ``(B, Z, C, H, W)`` occupancy logits.

        Args:
            x: ``(B, C, H, W)`` fused BEV features.

        Returns:
            ``{"occupancy": (B, Z, num_classes, H, W)}``.
        """
        f = self.stem(x)
        out = self.head(f)
        b, _, h, w = out.shape
        return {
            "occupancy": out.view(b, self.grid.num_z, self.cfg.occupancy_classes, h, w)
        }

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy over the flattened ``Z x H x W`` voxels.

        Args:
            logits: ``(B, Z, C, H, W)`` output of :meth:`forward`.
            target: ``(B, Z, H, W)`` integer labels in ``{0, 1, 2}``.
            class_weights: Optional ``(C,)`` per-class weights.

        Returns:
            Scalar cross-entropy loss.
        """
        b, z, c, h, w = logits.shape
        return F.cross_entropy(
            logits.reshape(b * z, c, h * w),
            target.reshape(b * z, h * w),
            weight=class_weights,
        )

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Return ``(B, Z, H, W)`` argmax occupancy predictions."""
        return logits.argmax(dim=2)
