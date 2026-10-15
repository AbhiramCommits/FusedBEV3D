"""BEV fusion: modality-agnostic concatenation/sum fusion followed by a BEV FPN.

The fusion block is deliberately a *single* code path: the lidar and camera BEV
feature maps are each projected to the shared width by a 1x1 convolution and
summed.  A disabled modality simply contributes a zero tensor, which is exactly
what the modality-dropout ablation exploits - there is no second "camera-only"
model to keep in sync.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fusedbev3d.config import ModelConfig

__all__ = ["BEVFusion", "ConvBNReLU"]


class ConvBNReLU(nn.Module):
    """3x3 convolution + batch norm + ReLU building block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return self.block(x)


class BEVFusion(nn.Module):
    """Fuse lidar + camera BEV features and refine them with a 2-down/2-up FPN.

    Args:
        cfg: Model configuration; ``fuse_channels`` sets the shared width and
            ``bev_backbone_channels`` the FPN width.
        lidar_channels: Channel width of the lidar BEV map.
        camera_channels: Channel width of the camera BEV map.
    """

    def __init__(self, cfg: ModelConfig, lidar_channels: int, camera_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.fuse_channels
        self.lidar_proj = nn.Conv2d(lidar_channels, c, 1)
        self.camera_proj = nn.Conv2d(camera_channels, c, 1)
        self.fuse = nn.Sequential(nn.Conv2d(c, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        b = cfg.bev_backbone_channels
        self.stem = ConvBNReLU(c, b)
        self.down1 = ConvBNReLU(b, b, stride=2)
        self.down2 = ConvBNReLU(b, b, stride=2)
        self.up1 = ConvBNReLU(b, b)
        self.up2 = ConvBNReLU(b, b)
        self.smooth = nn.Conv2d(b, b, 3, padding=1)

    def forward(
        self,
        lidar_bev: torch.Tensor | None,
        camera_bev: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fuse available modalities and return the refined BEV feature map.

        Either argument may be ``None`` (treated as a zero tensor) but not both.

        Args:
            lidar_bev: ``(B, C_l, H, W)`` lidar BEV features, or ``None``.
            camera_bev: ``(B, C_c, H, W)`` camera BEV features, or ``None``.

        Returns:
            ``(B, bev_backbone_channels, H, W)`` fused BEV features.

        Raises:
            ValueError: If both modalities are ``None``.
        """
        if lidar_bev is None and camera_bev is None:
            raise ValueError("BEVFusion needs at least one modality")
        ref = lidar_bev if lidar_bev is not None else camera_bev
        assert ref is not None
        b, _, h, w = ref.shape
        c = self.cfg.fuse_channels
        shape = (b, c, h, w)
        lidar_term = (
            self.lidar_proj(lidar_bev)
            if lidar_bev is not None
            else ref.new_zeros(shape)
        )
        camera_term = (
            self.camera_proj(camera_bev)
            if camera_bev is not None
            else ref.new_zeros(shape)
        )
        z = self.fuse(lidar_term + camera_term)
        x0 = self.stem(z)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        y = self.up1(torch.nn.functional.interpolate(x2, size=x1.shape[-2:], mode="nearest")) + x1
        y = self.up2(torch.nn.functional.interpolate(y, size=x0.shape[-2:], mode="nearest")) + x0
        return self.smooth(y)
