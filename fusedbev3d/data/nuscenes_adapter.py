"""nuScenes-mini adapter exposing the exact same sample contract as the generator.

The adapter is *import guarded*: ``nuscenes-devkit`` is only imported when the
dataset is actually instantiated, and a missing package or a missing ``dataroot``
raises a short, instructive error instead of an ``ImportError`` traceback.  Nothing
in the test-suite or in the headline results depends on this module.

Sensor conventions
------------------
* Lidar points are re-expressed in the **ego** frame at the sample timestamp using
  the ``LIDAR_TOP`` calibrated sensor, i.e. ``p_ego = R_cs @ p_lidar + t_cs``.
* Camera extrinsics are the ``camera-from-ego`` transforms ``inv(cs) @ ego_pose``.
* Boxes are converted from the global frame to the ego frame, ``size``
  ``[w, l, h]`` is re-ordered to ``(l, w, h)`` and ``velocity`` is rotated into the
  ego frame, matching the synthetic generator's box layout exactly.
* nuScenes has no occupancy ground truth, so the ``occupancy`` field is *derived*
  from the lidar sweep with the same visibility carve used by the generator.  This
  is documented rather than hidden: numbers computed on it are occupancy-metric
  sanity checks, not a benchmark claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fusedbev3d.config import DataConfig, GridConfig
from fusedbev3d.data.synth import (
    CAMERA_RIG,
    CLASS_NAMES,
    SyntheticDrivingDataset,
    occupancy_from_points,
)

__all__ = ["NuScenesMiniDataset", "NUSCENES_CATEGORY_MAP"]

#: nuScenes detection categories mapped onto the three classes used by FusedBEV3D.
NUSCENES_CATEGORY_MAP: dict[str, int] = {
    "vehicle.car": 0,
    "vehicle.emergency.ambulance": 0,
    "vehicle.emergency.police": 0,
    "human.pedestrian.adult": 1,
    "human.pedestrian.child": 1,
    "human.pedestrian.police_officer": 1,
    "human.pedestrian.construction_worker": 1,
    "vehicle.bicycle": 2,
    "vehicle.motorcycle": 2,
}

CAMERA_CHANNELS: tuple[str, ...] = tuple(name for name, _, _ in CAMERA_RIG)

_MISSING_HELP = (
    "NuScenesMiniDataset requires the optional 'nuscenes-devkit' package.\n"
    "Install it with:  pip install -e '.[nuscenes]'\n"
    "and download the mini split with:  \n"
    "    mkdir -p data/nuscenes && cd data/nuscenes && \\\n"
    "    wget https://www.nuscenes.org/data/v1.0-mini.tgz && tar -xf v1.0-mini.tgz"
)


def _require_nuscenes():
    """Import and return the devkit entry points, or raise a helpful error."""
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import Box as NuScenesBox  # noqa: F401
        from nuscenes.utils.geometry_utils import view_points  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(f"{_MISSING_HELP}\n\nOriginal error: {exc}") from exc
    return NuScenes


def _quaternion_to_matrix(q: list[float]) -> np.ndarray:
    """Convert a ``[w, x, y, z]`` quaternion to a ``(3, 3)`` rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_matrix(record: dict[str, Any]) -> np.ndarray:
    """Build a ``(4, 4)`` homogeneous matrix from a record's translation+rotation."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = _quaternion_to_matrix(record["rotation"])
    m[:3, 3] = np.asarray(record["translation"], dtype=np.float64)
    return m


class NuScenesMiniDataset(SyntheticDrivingDataset):
    """nuScenes-mini reader behind the synthetic dataset interface.

    Args:
        split: ``"train"`` or ``"val"``; split by *scene* so sequences stay intact.
        num_scenes: Optional cap on the number of scenes (useful on tiny laptops).
        grid: Grid geometry.
        cfg: Data configuration; ``cfg.version`` and ``cfg.dataroot`` select the
            nuScenes release and data root.
        cache: Optional ``.pt`` cache path used exactly like the synthetic split.

    Raises:
        ImportError: If ``nuscenes-devkit`` is not installed.
        FileNotFoundError: If ``dataroot`` (or its version sub-folder) is absent.
    """

    def __init__(
        self,
        split: str = "train",
        num_scenes: int | None = None,
        grid: GridConfig | None = None,
        cfg: DataConfig | None = None,
        cache: str | Path | None = None,
    ) -> None:
        nuscenes_cls = _require_nuscenes()
        self.cfg = cfg or DataConfig(name="nuscenes-mini")
        self.grid = grid or GridConfig()
        self.split = split
        self.num_scenes = int(num_scenes) if num_scenes else 0
        self.frames: list[dict[str, Any]] | None = None

        root = Path(self.cfg.dataroot or "data/nuscenes")
        if not root.exists():
            raise FileNotFoundError(
                f"nuScenes dataroot '{root}' does not exist.\n{_MISSING_HELP}"
            )
        if not (root / self.cfg.version).exists():
            raise FileNotFoundError(
                f"nuScenes version folder '{root / self.cfg.version}' not found.\n{_MISSING_HELP}"
            )

        self.nusc = nuscenes_cls(version=self.cfg.version, dataroot=str(root), verbose=False)
        self._scene_tokens = [s["token"] for s in self.nusc.scene]
        if self.num_scenes:
            self._scene_tokens = self._scene_tokens[: self.num_scenes]
        if cache is not None and Path(cache).exists():
            self.load_cache(cache)

    # -- generation -------------------------------------------------------------------

    def _frames_of_scene(self, scene_token: str) -> list[dict[str, Any]]:
        """Read one nuScenes scene into the shared frame-record format."""
        scene = self.nusc.get("scene", scene_token)
        token = scene["first_sample_token"]
        out: list[dict[str, Any]] = []
        frame_id = 0
        while token:
            sample = self.nusc.get("sample", token)
            out.append(self._read_sample(sample, scene_token, frame_id))
            frame_id += 1
            token = sample["next"]
        return out

    def _read_sample(self, sample: dict[str, Any], scene_token: str, frame_id: int) -> dict[str, Any]:
        """Convert a single nuScenes sample into a frame record."""
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import transform_matrix

        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_from_ego = transform_matrix(
            lidar_cs["translation"], lidar_cs["rotation"], inverse=False
        )
        ego_from_lidar = np.linalg.inv(lidar_from_ego)

        pc = LidarPointCloud.from_file(str(Path(self.nusc.dataroot) / lidar_sd["filename"]))
        pts = pc.points[:3, :]
        hmg = np.vstack([pts, np.ones((1, pts.shape[1]))])
        ego_pts = (ego_from_lidar @ hmg)[:3, :].T
        intensity = pc.points[3, :] if pc.points.shape[0] > 3 else np.ones(pts.shape[1])
        keep = (
            (np.abs(ego_pts[:, 0]) <= 60)
            & (np.abs(ego_pts[:, 1]) <= 60)
            & (np.abs(ego_pts[:, 2]) <= 10)
        )
        points = np.concatenate(
            [ego_pts[keep].astype(np.float32), intensity[keep, None].astype(np.float32)], axis=1
        )

        images, ks, es = self._read_cameras(sample)
        boxes = self._read_boxes(sample)
        occupancy = occupancy_from_points(points, self.grid, self.cfg)
        return {
            "points": points,
            "images": images,
            "intrinsics": ks,
            "extrinsics": es,
            "boxes": boxes,
            "occupancy": occupancy,
            "scene_id": self._scene_tokens.index(scene_token),
            "frame_id": frame_id,
            "token": f"{self.split}/{scene_token[:8]}/f{frame_id:03d}",
        }

    def _read_cameras(self, sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and resize the three rig cameras, returning images/K/E per camera."""
        from nuscenes.utils.data_classes import Box as _Box  # noqa: F401
        from nuscenes.utils.geometry_utils import transform_matrix
        from PIL import Image

        h, w = self.cfg.image_height, self.cfg.image_width
        images = np.zeros((len(CAMERA_CHANNELS), 3, h, w), dtype=np.uint8)
        ks = np.zeros((len(CAMERA_CHANNELS), 3, 3), dtype=np.float32)
        es = np.zeros((len(CAMERA_CHANNELS), 4, 4), dtype=np.float32)
        for ci, channel in enumerate(CAMERA_CHANNELS):
            sd = self.nusc.get("sample_data", sample["data"][channel])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
            cam_from_ego = np.linalg.inv(transform_matrix(cs["translation"], cs["rotation"], False))
            cam_from_ego = cam_from_ego @ transform_matrix(ep["translation"], ep["rotation"], False)
            k = np.asarray(cs["camera_intrinsic"], dtype=np.float64).copy()
            img = Image.open(str(Path(self.nusc.dataroot) / sd["filename"])).convert("RGB")
            sx, sy = w / img.width, h / img.height
            img = img.resize((w, h), Image.BILINEAR)
            images[ci] = np.transpose(np.asarray(img, dtype=np.uint8), (2, 0, 1))
            k[0, :] *= sx
            k[1, :] *= sy
            ks[ci] = k.astype(np.float32)
            es[ci] = cam_from_ego.astype(np.float32)
        return images, ks, es

    def _read_boxes(self, sample: dict[str, Any]) -> np.ndarray:
        """Convert annotated nuScenes boxes into the shared ``(M, 11)`` ego layout."""
        from nuscenes.utils.geometry_utils import transform_matrix

        rows: list[list[float]] = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            cls = NUSCENES_CATEGORY_MAP.get(ann["category_name"])
            if cls is None:
                continue
            sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
            ego_from_global = np.linalg.inv(
                transform_matrix(ep["translation"], ep["rotation"], inverse=False)
            )
            box_from_global = transform_matrix(ann["translation"], ann["rotation"], inverse=False)
            ego_from_box = ego_from_global @ box_from_global
            w, length, h = ann["size"]
            yaw = float(np.arctan2(ego_from_box[1, 0], ego_from_box[0, 0]))
            vel_global = np.asarray(ann["velocity"], dtype=np.float64)
            rot = ego_from_global[:3, :3]
            vel_ego = rot @ np.array([vel_global[0], vel_global[1], 0.0])
            track_id = self.nusc.get("sample_annotation", ann_token)
            rows.append(
                [
                    float(ego_from_box[0, 3]),
                    float(ego_from_box[1, 3]),
                    float(ego_from_box[2, 3]),
                    float(length),
                    float(w),
                    float(h),
                    yaw,
                    float(vel_ego[0]),
                    float(vel_ego[1]),
                    float(cls),
                    float(abs(hash(track_id["instance_token"])) % 100000),
                ]
            )
        if not rows:
            return np.zeros((0, 11), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def build(self, verbose: bool = False) -> "NuScenesMiniDataset":
        """Read every scene of the split into memory."""
        frames: list[dict[str, Any]] = []
        for i, tok in enumerate(self._scene_tokens):
            frames.extend(self._frames_of_scene(tok))
            if verbose:
                print(f"    read scene {i + 1}/{len(self._scene_tokens)}")
        self.frames = frames
        return self

    def __len__(self) -> int:
        """Number of samples in the split (requires :meth:`build` for an exact count)."""
        if self.frames is not None:
            return len(self.frames)
        return len(self._scene_tokens) * 40

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the tensorised sample at ``index`` (building lazily if needed)."""
        if self.frames is None:
            self.build()
        assert self.frames is not None
        return self._to_tensors(self.frames[index])

    @staticmethod
    def _to_tensors(frame: dict[str, Any]) -> dict[str, Any]:
        """Tensorise a nuScenes frame record into the shared sample dict."""
        return SyntheticDrivingDataset._to_tensors(frame)

    def class_names(self) -> tuple[str, ...]:
        """Return the class-name tuple used by this adapter."""
        return CLASS_NAMES


def smoke_check() -> str:
    """Return a short human-readable status string for the optional dependency."""
    try:
        _require_nuscenes()
    except ImportError as exc:
        return f"nuscenes-devkit unavailable: {str(exc).splitlines()[0]}"
    return "nuscenes-devkit available"

# torch is imported for the cache format used by the shared base class
_ = torch
