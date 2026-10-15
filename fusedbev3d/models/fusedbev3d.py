"""End-to-end FusedBEV3D model: encoders -> fusion -> heads.

The module owns the modality switch.  ``forward(batch, modalities=...)`` runs only
the encoders that are enabled and feeds a *zero* BEV map for the disabled
modality, so camera-only, lidar-only and fused runs share one identical graph and
differ only in which encoder produces non-zero input.  That is the ablation hook
used by :mod:`fusedbev3d.ablation`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusedbev3d.config import GridConfig, ModelConfig
from fusedbev3d.models.camera_encoder import CameraEncoder
from fusedbev3d.models.fusion import BEVFusion
from fusedbev3d.models.heads import REG_LAYOUT, CenterHead, OccupancyHead, sigmoid_focal_loss
from fusedbev3d.models.pillar_encoder import PillarEncoder

__all__ = ["FusedBEV3D", "normalize_modalities"]

VALID_MODALITIES: tuple[str, ...] = ("camera", "lidar")


def normalize_modalities(modalities: Iterable[str] | str | None) -> tuple[str, ...]:
    """Normalise a modality spec into a sorted tuple of known modalities.

    Args:
        modalities: ``None`` (both), a comma/plus separated string, or an iterable.

    Returns:
        A tuple containing only ``"camera"`` and/or ``"lidar"``, sorted.

    Raises:
        ValueError: If the spec is empty or contains an unknown modality.
    """
    if modalities is None:
        return VALID_MODALITIES
    if isinstance(modalities, str):
        parts = modalities.replace("+", ",").split(",")
        mods = tuple(p.strip() for p in parts if p.strip())
    else:
        mods = tuple(modalities)
    unknown = [m for m in mods if m not in VALID_MODALITIES]
    if unknown:
        raise ValueError(f"unknown modality {unknown}; valid: {VALID_MODALITIES}")
    if not mods:
        raise ValueError("at least one modality is required")
    return tuple(sorted(set(mods)))


class FusedBEV3D(nn.Module):
    """Multi-modal BEV perception network with detection and occupancy heads.

    Args:
        cfg: Model configuration.
        grid: Grid geometry (also fixes the occupancy volume shape).
        image_hw: ``(h, w)`` of the input camera images.
        modalities: Default enabled modalities; can be overridden per call.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        grid: GridConfig,
        image_hw: tuple[int, int] = (64, 192),
        modalities: Sequence[str] | str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = grid
        self.image_hw = image_hw
        self.modalities = normalize_modalities(modalities)
        self.lidar = PillarEncoder(grid, cfg)
        self.camera = CameraEncoder(grid, cfg, image_hw)
        self.fusion = BEVFusion(cfg, cfg.pillar_feat, cfg.camera_context)
        self.center_head = CenterHead(cfg, cfg.bev_backbone_channels)
        self.occupancy_head = OccupancyHead(cfg, grid, cfg.bev_backbone_channels)
        self.register_buffer(
            "occupancy_class_weights",
            torch.tensor([1.0, 2.2, 1.0], dtype=torch.float32),
            persistent=False,
        )

    # -- forward ----------------------------------------------------------------------

    def encode(
        self,
        batch: dict[str, Any],
        modalities: Sequence[str] | str | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Run the enabled encoders and fusion.

        Args:
            batch: Collated multi-modal batch.
            modalities: Modality override for this call.

        Returns:
            ``{"lidar_bev", "camera_bev", "fused"}`` where disabled modalities are ``None``.
        """
        mods = normalize_modalities(modalities if modalities is not None else self.modalities)
        lidar_bev = self.lidar(batch)["bev"] if "lidar" in mods else None
        camera_bev = self.camera(batch)["bev"] if "camera" in mods else None
        fused = self.fusion(lidar_bev, camera_bev)
        return {"lidar_bev": lidar_bev, "camera_bev": camera_bev, "fused": fused}

    def forward(
        self,
        batch: dict[str, Any],
        modalities: Sequence[str] | str | None = None,
        decode: bool = False,
        topk: int = 100,
    ) -> dict[str, Any]:
        """Run the full network.

        Args:
            batch: Collated multi-modal batch.
            modalities: Modality override (``None`` uses the configured default).
            decode: Also decode oriented boxes from the heatmap.
            topk: Number of detections to decode when ``decode`` is true.

        Returns:
            ``{"center", "occupancy", "features", "modalities"[, "detections"]}``.
        """
        enc = self.encode(batch, modalities)
        assert enc["fused"] is not None
        center = self.center_head(enc["fused"])
        occ = self.occupancy_head(enc["fused"])
        out: dict[str, Any] = {
            "center": center,
            "occupancy": occ["occupancy"],
            "features": enc["fused"],
            "modalities": normalize_modalities(modalities if modalities is not None else self.modalities),
            "lidar_bev": enc["lidar_bev"],
            "camera_bev": enc["camera_bev"],
        }
        if decode:
            out["detections"] = self.center_head.decode(center, self.grid, k=topk)
        return out

    # -- losses -----------------------------------------------------------------------

    def loss(
        self,
        outputs: dict[str, Any],
        targets: dict[str, Any],
        weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute every named loss term plus the weighted total.

        Args:
            outputs: Result of :meth:`forward`.
            targets: Result of :func:`fusedbev3d.targets.targets_from_batch`.
            weights: Optional per-term weights; defaults to
                ``{"heatmap":1, "offset":1, "z":1, "dims":1, "rot":1, "vel":0.2,
                "occupancy":1}``.

        Returns:
            A dictionary of scalar tensors: one entry per term plus ``"total"``.
        """
        w = {
            "heatmap": 1.0,
            "offset": 1.0,
            "z": 1.0,
            "dims": 1.0,
            "rot": 1.0,
            "vel": 0.2,
            "occupancy": 1.0,
        }
        if weights:
            w.update(weights)

        center_t = targets["center"]
        hm_loss = sigmoid_focal_loss(outputs["center"]["heatmap"], center_t["heatmap"])

        mask = center_t["mask"]
        reg_pred = outputs["center"]["reg"].permute(0, 2, 3, 1)[mask]
        reg_true = center_t["reg"].permute(0, 2, 3, 1)[mask]
        losses: dict[str, torch.Tensor] = {"heatmap": hm_loss}
        if reg_pred.numel():
            per_element = F.l1_loss(reg_pred, reg_true, reduction="none")
            for name, sl in REG_LAYOUT.items():
                losses[name] = per_element[:, sl].mean()
        else:
            zero = outputs["features"].sum() * 0.0
            for name in REG_LAYOUT:
                losses[name] = zero

        losses["occupancy"] = self.occupancy_head.loss(
            outputs["occupancy"], targets["occupancy"], self.occupancy_class_weights
        )
        total = sum(w[k] * losses[k] for k in losses)
        losses["total"] = total
        return losses
