"""Shared pytest fixtures.

Fixtures deliberately build a *tiny* synthetic split on the fly (no reliance on
``data/cache``) so the suite is self-contained, deterministic and fast.
"""

from __future__ import annotations

import pytest
import torch

from fusedbev3d.config import Config
from fusedbev3d.data.synth import SyntheticDrivingDataset, collate_fn
from fusedbev3d.models.fusedbev3d import FusedBEV3D


@pytest.fixture(scope="session")
def cfg() -> Config:
    """A small deterministic config used across the suite."""
    c = Config()
    c.data.num_scenes = 2
    c.data.val_scenes = 1
    c.data.frames_per_scene = 3
    return c


@pytest.fixture(scope="session")
def samples(cfg: Config) -> list[dict]:
    """Two tensorised frames from one generated sequence."""
    ds = SyntheticDrivingDataset(split="val", num_scenes=1, grid=cfg.grid, cfg=cfg.data)
    return [ds[0], ds[1]]


@pytest.fixture(scope="session")
def batch(samples: list[dict]) -> dict:
    """A collated batch of the two sample frames."""
    return collate_fn(samples)


@pytest.fixture
def model(cfg: Config) -> FusedBEV3D:
    """A freshly seeded model instance."""
    torch.manual_seed(0)
    net = FusedBEV3D(
        cfg.model,
        cfg.grid,
        (cfg.data.image_height, cfg.data.image_width),
        cfg.train.modalities,
    )
    net.eval()
    return net
