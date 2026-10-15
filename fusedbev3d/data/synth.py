"""Procedural multi-modal driving-scene generator.

This module ships a fully deterministic, dependency-light driving-scene simulator
that produces *genuinely multi-modal* data on a CPU laptop:

* **Lidar** - a 32-beam spinning sensor is ray-cast analytically against the
  oriented bounding boxes of every actor plus the ground plane (slab test in the
  box frame).  Returns get range noise and a dropout probability.
* **Cameras** - three pinhole cameras (front / front-left / front-right) with real
  intrinsics and extrinsics.  Every actor is projected and rasterised as a
  depth-sorted filled polygon, on top of a rendered ground plane, horizon and road
  markings, so the images carry genuine class and depth information.
* **Labels** - oriented 3D boxes with velocities and persistent track ids, plus a
  3-class voxel occupancy volume ``{0=free, 1=occupied, 2=unknown}`` where
  *unknown* is exactly the set of voxels no lidar ray passed through (visibility
  derived, not hand-authored).

Frames are grouped into sequences; actors move with constant velocity plus a small
yaw-rate and keep their ``track_id`` across the whole sequence, which is what makes
the tracking evaluation in Step 4 meaningful.

Sample dictionary (shapes use ``N`` lidar points, ``M`` boxes, ``S=3`` cameras,
``Z=8`` occupancy levels, ``H/W`` the BEV grid, ``h/w`` the image size):

===========================  ==========================  ============================
key                          shape                       dtype / meaning
===========================  ==========================  ============================
``points``                   ``(N, 4)``                   float32 ``x, y, z, intensity``
``images``                   ``(S, 3, h, w)``             uint8->float32 RGB in ``[0,1]``
``intrinsics``               ``(S, 3, 3)``                float32 pinhole ``K``
``extrinsics``               ``(S, 4, 4)``                float32 camera-from-ego
``boxes``                    ``(M, 11)``                  float32 ``x y z l w h yaw vx vy class track_id``
``occupancy``                ``(Z, H, W)``                int64 in ``{0, 1, 2}``
``scene_id``                 scalar                       int64 sequence index
``frame_id``                 scalar                       int64 frame index inside the sequence
``token``                    str                          unique ``"{split}/s{scene}/f{frame}"``
===========================  ==========================  ============================

All randomness is derived from ``DataConfig.seed`` and the scene index, so a given
``(split, num_scenes, frames_per_scene, seed)`` quadruple always regenerates byte
identical data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from fusedbev3d.config import DataConfig, GridConfig

__all__ = [
    "CLASS_NAMES",
    "CLASS_PROPS",
    "CAMERA_RIG",
    "SyntheticDrivingDataset",
    "collate_fn",
    "camera_intrinsics",
    "camera_extrinsics",
    "ray_box_intersection",
    "cast_lidar",
    "render_cameras",
    "occupancy_from_points",
    "ego_to_bev_cell",
]

# --------------------------------------------------------------------------------------
# Class definitions
# --------------------------------------------------------------------------------------

CLASS_NAMES: tuple[str, ...] = ("car", "pedestrian", "cyclist")

#: ``(length, width, height, speed_lo, speed_hi, rgb, lidar_reflectivity)`` per class.
CLASS_PROPS: dict[int, dict[str, Any]] = {
    0: dict(l=4.2, w=1.8, h=1.6, vmin=3.0, vmax=8.0, rgb=(215, 62, 62), refl=0.95),
    1: dict(l=0.7, w=0.7, h=1.75, vmin=0.6, vmax=1.4, rgb=(64, 120, 230), refl=0.45),
    2: dict(l=1.8, w=0.6, h=1.70, vmin=2.5, vmax=5.5, rgb=(240, 200, 60), refl=0.75),
}

#: ``(name, yaw offset [rad], position offset in ego frame [m])``.
CAMERA_RIG: tuple[tuple[str, float, tuple[float, float, float]], ...] = (
    ("CAM_FRONT", 0.0, (0.0, 0.0, 1.60)),
    ("CAM_FRONT_LEFT", 0.62, (0.45, 0.55, 1.60)),
    ("CAM_FRONT_RIGHT", -0.62, (0.45, -0.55, 1.60)),
)

#: Camera-from-ego rotation *before* the per-camera yaw offset is applied.
#: Ego is ``x=forward, y=left, z=up``; camera is ``x=right, y=down, z=forward``.
R_CAM_FROM_EGO = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

LIDAR_HEIGHT = 1.73
GROUND_Z = 0.0


# --------------------------------------------------------------------------------------
# Camera geometry
# --------------------------------------------------------------------------------------


def _rot_z(angle: float) -> np.ndarray:
    """Return the 3x3 rotation matrix about +z for ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_intrinsics(height: int, width: int, fov_deg: float) -> np.ndarray:
    """Build a pinhole ``K`` for an ``(height, width)`` image with horizontal FOV.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        fov_deg: Horizontal field of view in degrees.

    Returns:
        ``(3, 3)`` float64 intrinsic matrix with the principal point at the
        image centre.
    """
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def camera_extrinsics(index: int) -> np.ndarray:
    """Return the ``(4, 4)`` camera-from-ego transform for camera ``index``.

    Args:
        index: Index into :data:`CAMERA_RIG`.

    Returns:
        Homogeneous rigid transform mapping ego-frame points to that camera's frame.
    """
    _, yaw, offset = CAMERA_RIG[index]
    r = R_CAM_FROM_EGO @ _rot_z(yaw)
    t = -r @ np.asarray(offset, dtype=np.float64)
    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = r
    ext[:3, 3] = t
    return ext


def build_camera_rig(height: int, width: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return stacked ``(S, 3, 3)`` intrinsics and ``(S, 4, 4)`` extrinsics for the rig."""
    ks = np.stack([camera_intrinsics(height, width, fov_deg) for _ in CAMERA_RIG])
    es = np.stack([camera_extrinsics(i) for i in range(len(CAMERA_RIG))])
    return ks, es


def project_ego_to_image(
    points_ego: np.ndarray, k: np.ndarray, ext: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame points into one camera image.

    Args:
        points_ego: ``(P, 3)`` points in the ego frame.
        k: ``(3, 3)`` intrinsics.
        ext: ``(4, 4)`` camera-from-ego transform.

    Returns:
        ``(uv, depth)`` where ``uv`` is ``(P, 2)`` pixel coordinates and ``depth``
        is ``(P,)`` camera-frame z (metres).  Points behind the camera get
        ``depth <= 0`` and must be discarded by the caller.
    """
    pts = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    cam = pts @ ext[:3, :3].T + ext[:3, 3]
    depth = cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uv = (cam[:, :2] / safe[:, None]) @ k[:2, :2].T + k[:2, 2]
    return uv, depth


# --------------------------------------------------------------------------------------
# Lidar ray casting
# --------------------------------------------------------------------------------------


def _lidar_directions(num_beams: int, num_azimuth: int, elev_lo: float, elev_hi: float) -> np.ndarray:
    """Return unit ray directions ``(B*A, 3)`` for the spinning lidar."""
    elev = np.deg2rad(np.linspace(elev_hi, elev_lo, num_beams))
    azim = np.linspace(0.0, 2.0 * math.pi, num_azimuth, endpoint=False)
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    dirs = np.stack(
        [
            np.outer(ce, ca).ravel(),
            np.outer(ce, sa).ravel(),
            np.repeat(se, num_azimuth),
        ],
        axis=1,
    )
    return dirs.astype(np.float32)


def ray_box_intersection(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    boxes: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic oriented-box ray intersection via a slab test in the box frame.

    Args:
        origins: ``(N, 3)`` ray origins in the ego frame.
        dirs: ``(N, 3)`` unit ray directions in the ego frame.
        boxes: ``(M, 11)`` boxes ``x y z l w h yaw vx vy class track_id``.
        eps: Directional epsilon guarding against divide-by-zero.

    Returns:
        ``(t, hit)`` where ``t`` is ``(N, M)`` entry distances in metres
        (``inf`` when there is no hit) and ``hit`` is the matching boolean mask.
    """
    n, m = origins.shape[0], boxes.shape[0]
    if m == 0:
        return (
            torch.full((n, 0), float("inf"), dtype=origins.dtype, device=origins.device),
            torch.zeros((n, 0), dtype=torch.bool, device=origins.device),
        )
    center = boxes[:, :3]
    yaw = boxes[:, 6]
    half = torch.stack([boxes[:, 3], boxes[:, 4], boxes[:, 5]], dim=-1) * 0.5

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    rel = origins[:, None, :] - center[None, :, :]  # (N, M, 3)
    # rotate into the box frame: R_z(-yaw) applied to the xy components
    vx = cy[None, :] * rel[..., 0] + sy[None, :] * rel[..., 1]
    vy = -sy[None, :] * rel[..., 0] + cy[None, :] * rel[..., 1]
    vz = rel[..., 2]
    v = torch.stack([vx, vy, vz], dim=-1)

    dx = cy[None, :] * dirs[:, None, 0] + sy[None, :] * dirs[:, None, 1]
    dy = -sy[None, :] * dirs[:, None, 0] + cy[None, :] * dirs[:, None, 1]
    dz = dirs[:, None, 2].expand_as(dx)
    d = torch.stack([dx, dy, dz], dim=-1)
    d = torch.where(d.abs() < eps, torch.full_like(d, eps), d)

    h = half[None, :, :].expand_as(v)
    t1 = (-h - v) / d
    t2 = (h - v) / d
    t_near = torch.minimum(t1, t2)
    t_far = torch.maximum(t1, t2)
    t_enter = t_near.max(dim=-1).values
    t_exit = t_far.min(dim=-1).values
    hit = (t_exit >= torch.clamp(t_enter, min=0.0)) & (t_exit > 0.0)
    t = torch.where(t_enter > 0.0, t_enter, torch.zeros_like(t_enter))
    t = torch.where(hit, t, torch.full_like(t, float("inf")))
    return t, hit


def cast_lidar(
    boxes: torch.Tensor,
    cfg: DataConfig,
    rng: np.random.Generator,
    grid: GridConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-cast the spinning lidar against ``boxes`` and the ground plane.

    Args:
        boxes: ``(M, 11)`` ego-frame boxes.
        cfg: Data configuration (beam count, range, noise, dropout).
        rng: NumPy generator used for range noise and dropout.
        grid: Unused placeholder kept for interface symmetry with the adapter.

    Returns:
        ``(points, hit_distance, hit_is_actor)`` where ``points`` is ``(N, 4)``
        float32 ``x y z intensity``, ``hit_distance`` is ``(N,)`` float32 range of
        each surviving return and ``hit_is_actor`` is ``(N,)`` bool telling whether
        the return came off an actor box (as opposed to the ground plane).
    """
    del grid
    dirs_np = _lidar_directions(
        cfg.num_beams, cfg.num_azimuth, cfg.elevation_min_deg, cfg.elevation_max_deg
    )
    origin = np.array([0.0, 0.0, LIDAR_HEIGHT], dtype=np.float32)

    dirs = torch.from_numpy(dirs_np)
    origins = torch.from_numpy(np.tile(origin, (dirs.shape[0], 1)))
    t_box, _ = ray_box_intersection(origins, dirs, boxes)
    t_box_min = t_box.min(dim=1).values if t_box.shape[1] else torch.full((dirs.shape[0],), float("inf"))

    dz = dirs[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = (GROUND_Z - LIDAR_HEIGHT) / dz.numpy()
    t_ground = np.where(dz.numpy() < -1e-6, t_ground, np.inf).astype(np.float32)
    t_ground_t = torch.from_numpy(t_ground)

    t = torch.minimum(t_box_min, t_ground_t)
    has_actor = t_box_min <= t_ground_t
    valid = torch.isfinite(t) & (t <= cfg.max_range) & (t > 0.2)

    t_np = t.numpy()
    valid_np = valid.numpy()
    actor_np = has_actor.numpy()

    ranges = t_np[valid_np].astype(np.float32)
    ranges = ranges + rng.normal(0.0, cfg.range_noise, size=ranges.shape).astype(np.float32)
    keep = rng.random(ranges.shape[0]) >= cfg.dropout

    pts = origin[None, :] + dirs_np[valid_np][keep] * ranges[keep, None]
    is_actor = actor_np[valid_np][keep]

    # reflectivity: class-dependent for actor hits, fixed for ground
    intensity = np.full(pts.shape[0], 0.5, dtype=np.float32)
    if boxes.shape[0] and pts.shape[0]:
        cls = boxes[:, 9].numpy().astype(np.int64)
        refl = np.array([CLASS_PROPS[int(c)]["refl"] for c in cls], dtype=np.float32)
        ray_idx = np.flatnonzero(valid_np)[keep]
        t_b = t_box.numpy()[ray_idx]
        nearest = t_b.argmin(axis=1)
        intensity[is_actor] = refl[nearest][is_actor]
    intensity *= np.clip(1.0 - ranges[keep] / (1.6 * cfg.max_range), 0.25, 1.0)
    intensity = np.clip(intensity + rng.normal(0.0, 0.02, size=intensity.shape), 0.02, 1.0)

    points = np.concatenate([pts.astype(np.float32), intensity.astype(np.float32)[:, None]], axis=1)
    return points, ranges[keep].astype(np.float32), is_actor.astype(bool)


# --------------------------------------------------------------------------------------
# Occupancy labels
# --------------------------------------------------------------------------------------


def ego_to_bev_cell(points: torch.Tensor, grid: GridConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ego-frame points to BEV cell indices.

    Args:
        points: ``(..., 3)`` ego-frame points.
        grid: Grid geometry.

    Returns:
        ``(ix, iy)`` integer tensors matching ``points`` in shape, where ``ix`` is
        the forward cell index and ``iy`` the lateral one.  No range check is
        applied; use :func:`cell_in_range` (or compare against ``grid.bev_w/w``).
    """
    ix = torch.floor((points[..., 0] - grid.x_range[0]) / grid.voxel_size).long()
    iy = torch.floor((points[..., 1] - grid.y_range[0]) / grid.voxel_size).long()
    return ix, iy


def occupancy_from_points(
    points: np.ndarray,
    grid: GridConfig,
    cfg: DataConfig,
) -> np.ndarray:
    """Derive the 3-class occupancy volume from the *actual* lidar returns.

    Every surviving return carves free space along the segment
    ``lidar_origin -> return`` and stamps its own voxel as occupied.  Voxels that
    no return ever touched stay ``2`` (unknown), so the unknown class genuinely
    encodes sensor visibility rather than a heuristic.

    Args:
        points: ``(N, 4)`` lidar returns ``x y z intensity`` in the ego frame.
        grid: Grid geometry.
        cfg: Data configuration (``occupancy_range``, ``occupancy_step``).

    Returns:
        ``(num_z, bev_h, bev_w)`` uint8 array with values in ``{0, 1, 2}``.
    """
    nz, nh, nw = grid.num_z, grid.bev_h, grid.bev_w
    free = np.zeros(nz * nh * nw, dtype=bool)

    if points.shape[0] == 0:
        return np.full((nz, nh, nw), 2, dtype=np.uint8)

    origin = np.array([0.0, 0.0, LIDAR_HEIGHT], dtype=np.float64)
    rel = points[:, :3].astype(np.float64) - origin
    dist = np.linalg.norm(rel, axis=1)
    keep = dist > 1e-3
    rel, dist = rel[keep], dist[keep]
    dirs = rel / np.maximum(dist, 1e-9)[:, None]

    step = cfg.occupancy_step
    t_max = np.minimum(dist, cfg.occupancy_range)
    n_steps = int(math.ceil(cfg.occupancy_range / step)) + 1
    t = (np.arange(1, n_steps + 1) * step)[None, :]
    mask = t <= t_max[:, None]

    pts = origin[None, None, :] + t[..., None] * dirs[:, None, :]
    flat_idx = _voxel_flat_index(pts, grid)
    valid = mask & (flat_idx >= 0)
    if valid.any():
        free[flat_idx[valid]] = True

    # endpoints: the return voxel itself is occupied
    ends = points[:, :3].astype(np.float64)
    end_idx = _voxel_flat_index(ends, grid)
    occ = np.zeros_like(free)
    sel = end_idx >= 0
    occ[end_idx[sel]] = True

    labels = np.where(occ, 1, np.where(free, 0, 2)).astype(np.uint8)
    return labels.reshape(nz, nh, nw)


def _voxel_flat_index(points: np.ndarray, grid: GridConfig) -> np.ndarray:
    """Return flat ``z*H*W + y*W + x`` voxel indices, or ``-1`` when out of grid."""
    nz, nh, nw = grid.num_z, grid.bev_h, grid.bev_w
    ix = np.floor((points[..., 0] - grid.x_range[0]) / grid.voxel_size).astype(np.int64)
    iy = np.floor((points[..., 1] - grid.y_range[0]) / grid.voxel_size).astype(np.int64)
    iz = np.floor((points[..., 2] - grid.z_range[0]) / grid.z_voxel_size).astype(np.int64)
    inb = (ix >= 0) & (ix < nw) & (iy >= 0) & (iy < nh) & (iz >= 0) & (iz < nz)
    flat = iz * nh * nw + iy * nw + ix
    return np.where(inb, flat, -1)


# --------------------------------------------------------------------------------------
# Camera rendering
# --------------------------------------------------------------------------------------


def _canvas(height: int, width: int) -> np.ndarray:
    """Render the static background: sky gradient above the horizon, road below."""
    img = np.zeros((height, width, 3), dtype=np.float32)
    horizon = height / 2.0
    rows = np.arange(height, dtype=np.float32)[:, None]
    above = np.clip(1.0 - rows / max(horizon, 1.0), 0.0, 1.0)
    below = np.clip((rows - horizon) / max(height - horizon, 1.0), 0.0, 1.0)
    sky_top = np.array([168.0, 196.0, 232.0])
    sky_bot = np.array([214.0, 226.0, 244.0])
    road_near = np.array([42.0, 42.0, 46.0])
    road_far = np.array([104.0, 104.0, 110.0])
    sky = sky_bot[None, None, :] * (1 - above[..., None]) + sky_top[None, None, :] * above[..., None]
    road = road_far[None, None, :] * (1 - below[..., None]) + road_near[None, None, :] * below[..., None]
    img[:] = np.where(rows[:, :, None] < horizon, sky, road)
    return img


def _hull(points_2d: np.ndarray) -> np.ndarray:
    """Order 2D points counter-clockwise about their centroid (projected boxes are convex)."""
    c = points_2d.mean(axis=0, keepdims=True)
    ang = np.arctan2(points_2d[:, 1] - c[:, 1], points_2d[:, 0] - c[:, 0])
    return points_2d[np.argsort(ang)]


def _box_corners(boxes: np.ndarray) -> np.ndarray:
    """Return ``(M, 8, 3)`` ego-frame corners for ``(M, 11)`` boxes."""
    m = boxes.shape[0]
    l, w, h = boxes[:, 3] / 2, boxes[:, 4] / 2, boxes[:, 5] / 2
    xs = np.stack([-l, l, l, -l, -l, l, l, -l], axis=1)
    ys = np.stack([-w, -w, w, w, -w, -w, w, w], axis=1)
    zs = np.stack([-h, -h, -h, -h, h, h, h, h], axis=1)
    local = np.stack([xs, ys, zs], axis=-1)  # (M, 8, 3)
    cy = np.cos(boxes[:, 6])[:, None]
    sy = np.sin(boxes[:, 6])[:, None]
    zero = np.zeros_like(cy)
    one = np.ones_like(cy)
    rot = np.stack(
        [
            np.concatenate([cy, -sy, zero], axis=1),
            np.concatenate([sy, cy, zero], axis=1),
            np.concatenate([zero, zero, one], axis=1),
        ],
        axis=1,
    )  # (M, 3, 3)
    world = np.einsum("mij,mkj->mki", rot, local) + boxes[:, None, :3]
    return world


def render_cameras(
    boxes: np.ndarray,
    grid: GridConfig,
    cfg: DataConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rasterise the scene for all rig cameras.

    Args:
        boxes: ``(M, 11)`` ego-frame boxes.
        grid: Grid geometry (used for the road-marking extent).
        cfg: Data configuration (image size, FOV).
        rng: NumPy generator for sensor noise.

    Returns:
        ``(S, 3, h, w)`` uint8 RGB images, channels last in the drawing code.
    """
    del grid
    h, w = cfg.image_height, cfg.image_width
    ks, es = build_camera_rig(h, w, cfg.camera_fov)
    out = np.zeros((len(CAMERA_RIG), h, w, 3), dtype=np.uint8)
    corners = _box_corners(boxes) if boxes.shape[0] else np.zeros((0, 8, 3))

    # road markings in the ego frame: two solid lane boundaries + centre dashes
    marks: list[np.ndarray] = []
    for x0 in np.arange(2.0, 34.0, 4.0):
        for y0 in (-1.85, 1.85):
            marks.append(
                np.array(
                    [[x0, y0 - 0.09, 0.02], [x0 + 2.4, y0 - 0.09, 0.02],
                     [x0 + 2.4, y0 + 0.09, 0.02], [x0, y0 + 0.09, 0.02]]
                )
            )
    for x0 in np.arange(1.0, 34.0, 6.0):
        marks.append(
            np.array(
                [[x0, -0.09, 0.02], [x0 + 3.0, -0.09, 0.02],
                 [x0 + 3.0, 0.09, 0.02], [x0, 0.09, 0.02]]
            )
        )

    for ci in range(len(CAMERA_RIG)):
        img = _canvas(h, w)
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)

        for quad in marks:
            uv, depth = project_ego_to_image(quad, ks[ci], es[ci])
            if np.any(depth <= 0.3):
                continue
            poly = [(float(u), float(v)) for u, v in uv]
            if max(u for u, _ in poly) < 0 or min(u for u, _ in poly) > w:
                continue
            draw.polygon(poly, fill=(228, 228, 214))

        if boxes.shape[0]:
            order = np.argsort(-boxes[:, 0])  # draw far boxes first
            for bi in order:
                uv, depth = project_ego_to_image(corners[bi], ks[ci], es[ci])
                if np.all(depth <= 0.3):
                    continue
                if np.any(depth <= 0.3):
                    continue
                if np.all(uv[:, 0] < -w) or np.all(uv[:, 0] > 2 * w):
                    continue
                if np.all(uv[:, 1] < -h) or np.all(uv[:, 1] > 2 * h):
                    continue
                cls = int(boxes[bi, 9])
                base = np.array(CLASS_PROPS[cls]["rgb"], dtype=np.float64)
                d = float(depth.mean())
                shade = float(np.clip(1.25 - 0.05 * d, 0.35, 1.0))
                color = tuple(int(np.clip(c * shade, 0, 255)) for c in base)
                poly = [(float(u), float(v)) for u, v in _hull(uv)]
                # ground-contact shadow, then the body, then a darker outline
                draw.polygon([(u, v + 2.0) for u, v in poly], fill=(28, 28, 30))
                draw.polygon(poly, fill=color)
                draw.line(poly + [poly[0]], fill=tuple(int(c * 0.55) for c in color), width=1)

        arr = np.asarray(pil, dtype=np.float32)
        arr += rng.normal(0.0, 2.5, size=arr.shape)
        out[ci] = np.clip(arr, 0, 255).astype(np.uint8)

    return np.transpose(out, (0, 3, 1, 2))  # (S, 3, h, w)


# --------------------------------------------------------------------------------------
# Scene / sequence generation
# --------------------------------------------------------------------------------------


def _sample_actor(rng: np.random.Generator, track_id: int) -> dict[str, float]:
    """Sample one actor with a random class, pose and constant velocity."""
    cls = int(rng.choice(3, p=[0.5, 0.3, 0.2]))
    props = CLASS_PROPS[cls]
    while True:
        x = rng.uniform(-13.0, 13.0)
        y = rng.uniform(-13.0, 13.0)
        if math.hypot(x, y) > 4.0:
            break
    heading = math.atan2(y, x)
    drift = float(rng.normal(0.0, 0.35))
    speed = rng.uniform(props["vmin"], props["vmax"])
    return {
        "class": cls,
        "track_id": track_id,
        "x": x,
        "y": y,
        "z": props["h"] / 2.0,
        "l": props["l"],
        "w": props["w"],
        "h": props["h"],
        "yaw": heading + drift,
        "yaw_drift": drift,
        "vx": speed * math.cos(heading),
        "vy": speed * math.sin(heading),
        "yaw_rate": rng.normal(0.0, 0.15),
    }


def _step_actor(a: dict[str, float], dt: float) -> None:
    """Advance one actor by ``dt`` seconds: constant velocity + slight yaw rate.

    Actors bounce off the ``+-13.5 m`` scene bounds so they stay observable for the
    whole sequence, and their heading follows the (possibly reflected) velocity
    direction plus an accumulating, bounded yaw drift.
    """
    a["x"] += a["vx"] * dt
    a["y"] += a["vy"] * dt
    for axis in ("x", "y"):
        key = "v" + axis
        if a[axis] > 13.5 and a[key] > 0:
            a[key] = -a[key]
        elif a[axis] < -13.5 and a[key] < 0:
            a[key] = -a[key]
    drift = float(np.clip(a["yaw_drift"] + a["yaw_rate"] * dt, -0.6, 0.6))
    a["yaw_drift"] = drift
    a["yaw"] = math.atan2(a["vy"], a["vx"]) + drift


def _boxes_from_actors(actors: Sequence[dict[str, float]]) -> np.ndarray:
    """Pack actor dictionaries into an ``(M, 11)`` float32 box array."""
    if not actors:
        return np.zeros((0, 11), dtype=np.float32)
    rows = [
        [
            a["x"], a["y"], a["z"], a["l"], a["w"], a["h"],
            a["yaw"], a["vx"], a["vy"], a["class"], a["track_id"],
        ]
        for a in actors
    ]
    return np.asarray(rows, dtype=np.float32)


def build_sequence(
    scene_id: int,
    grid: GridConfig,
    cfg: DataConfig,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Generate one full multi-modal sequence of ``cfg.frames_per_scene`` frames.

    Args:
        scene_id: Used to derive the per-scene RNG and the track-id namespace.
        grid: Grid geometry.
        cfg: Data configuration.
        split: Split tag embedded in the frame token (``train``/``val``).

    Returns:
        A list of frame dictionaries (the CPU/NumPy form of the sample dict).
    """
    rng = np.random.default_rng(cfg.seed + 1000003 * scene_id + (7 if split == "val" else 0))
    n_actors = int(rng.integers(cfg.min_actors, cfg.max_actors + 1))
    actors = [_sample_actor(rng, track_id=scene_id * 100 + i + 1) for i in range(n_actors)]

    ks, es = build_camera_rig(cfg.image_height, cfg.image_width, cfg.camera_fov)
    dt = 0.5
    frames: list[dict[str, Any]] = []
    for t in range(cfg.frames_per_scene):
        boxes_np = _boxes_from_actors(actors)
        boxes = torch.from_numpy(boxes_np)
        points, _, _ = cast_lidar(boxes, cfg, rng, grid)
        occ = occupancy_from_points(points, grid, cfg)
        images = render_cameras(boxes_np, grid, cfg, rng)
        frames.append(
            {
                "points": points,
                "images": images,
                "intrinsics": ks.astype(np.float32),
                "extrinsics": es.astype(np.float32),
                "boxes": boxes_np,
                "occupancy": occ,
                "scene_id": scene_id,
                "frame_id": t,
                "token": f"{split}/s{scene_id:03d}/f{t:03d}",
            }
        )
        for a in actors:
            _step_actor(a, dt)
    return frames


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class SyntheticDrivingDataset(Dataset):
    """Deterministic procedural multi-modal driving dataset.

    See the module docstring for the full sample dictionary contract.  Sequences
    are generated lazily per scene and can be materialised to a ``.pt`` cache with
    :meth:`save_cache` / re-loaded with :meth:`from_cache`.

    Args:
        split: ``"train"`` or ``"val"``.
        num_scenes: Number of sequences in this split.
        grid: Grid geometry (defaults to :class:`GridConfig`).
        cfg: Data configuration (defaults to :class:`DataConfig`).
        cache: Optional path to a materialised cache; loaded when it exists.
    """

    def __init__(
        self,
        split: str = "train",
        num_scenes: int | None = None,
        grid: GridConfig | None = None,
        cfg: DataConfig | None = None,
        cache: str | Path | None = None,
    ) -> None:
        self.split = split
        self.cfg = cfg or DataConfig()
        self.grid = grid or GridConfig()
        if num_scenes is None:
            num_scenes = self.cfg.val_scenes if split == "val" else self.cfg.num_scenes
        self.num_scenes = int(num_scenes)
        self.frames: list[dict[str, Any]] | None = None
        if cache is not None and Path(cache).exists():
            self.load_cache(cache)

    # -- generation -------------------------------------------------------------------

    def build(self, verbose: bool = False) -> "SyntheticDrivingDataset":
        """Generate every frame of the split into memory (and return ``self``)."""
        frames: list[dict[str, Any]] = []
        for s in range(self.num_scenes):
            frames.extend(build_sequence(s, self.grid, self.cfg, self.split))
            if verbose and (s + 1) % 10 == 0:
                print(f"    generated {s + 1}/{self.num_scenes} scenes")
        self.frames = frames
        return self

    def __len__(self) -> int:
        """Number of frames in the split (``num_scenes * frames_per_scene``)."""
        return self.num_scenes * self.cfg.frames_per_scene

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the tensorised frame dictionary at ``index``."""
        if self.frames is None:
            self.frames = []
            for s in range(self.num_scenes):
                self.frames.extend(build_sequence(s, self.grid, self.cfg, self.split))
        return self._to_tensors(self.frames[index])

    @staticmethod
    def _to_tensors(frame: dict[str, Any]) -> dict[str, Any]:
        """Convert one NumPy frame record into the public tensor sample dict."""
        return {
            "points": torch.as_tensor(frame["points"], dtype=torch.float32),
            "images": torch.as_tensor(frame["images"], dtype=torch.float32).div_(255.0),
            "intrinsics": torch.as_tensor(frame["intrinsics"], dtype=torch.float32),
            "extrinsics": torch.as_tensor(frame["extrinsics"], dtype=torch.float32),
            "boxes": torch.as_tensor(frame["boxes"], dtype=torch.float32).reshape(-1, 11),
            "occupancy": torch.as_tensor(frame["occupancy"], dtype=torch.long),
            "scene_id": torch.tensor(frame["scene_id"], dtype=torch.long),
            "frame_id": torch.tensor(frame["frame_id"], dtype=torch.long),
            "token": str(frame["token"]),
        }

    # -- caching ----------------------------------------------------------------------

    def save_cache(self, path: str | Path) -> Path:
        """Materialise the split to ``path`` (``.pt``) and return the path."""
        if self.frames is None:
            self.build()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": self.split,
            "num_scenes": self.num_scenes,
            "frames_per_scene": self.cfg.frames_per_scene,
            "grid": self.grid,
            "data": self.cfg,
            "frames": self.frames,
        }
        torch.save(payload, path)
        return path

    def load_cache(self, path: str | Path) -> "SyntheticDrivingDataset":
        """Load a previously materialised split from ``path``."""
        payload = torch.load(Path(path), weights_only=False)
        self.frames = payload["frames"]
        self.num_scenes = payload["num_scenes"]
        self.cfg = payload["data"]
        self.grid = payload["grid"]
        return self

    def statistics(self) -> dict[str, Any]:
        """Compute the summary statistics printed by ``scripts/make_dataset.py``."""
        if self.frames is None:
            self.build()
        assert self.frames is not None
        class_counts = {name: 0 for name in CLASS_NAMES}
        total_pts = 0
        occ_counts = np.zeros(3, dtype=np.int64)
        for f in self.frames:
            b = f["boxes"]
            for c in range(3):
                class_counts[CLASS_NAMES[c]] += int((b[:, 9] == c).sum()) if b.shape[0] else 0
            total_pts += int(f["points"].shape[0])
            occ_counts += np.bincount(f["occupancy"].ravel(), minlength=3)
        n = max(len(self.frames), 1)
        total_occ = int(occ_counts.sum())
        return {
            "split": self.split,
            "frames": len(self.frames),
            "scenes": self.num_scenes,
            "boxes_per_class": class_counts,
            "boxes_total": int(sum(class_counts.values())),
            "points_total": total_pts,
            "points_per_frame": total_pts / n,
            "occupancy_counts": {
                "free": int(occ_counts[0]),
                "occupied": int(occ_counts[1]),
                "unknown": int(occ_counts[2]),
            },
            "occupancy_fraction": {
                "free": float(occ_counts[0] / max(total_occ, 1)),
                "occupied": float(occ_counts[1] / max(total_occ, 1)),
                "unknown": float(occ_counts[2] / max(total_occ, 1)),
            },
        }


def collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate ragged samples into one training batch.

    Lidar points are concatenated along dim 0 and a ``point_batch`` index vector
    records which sample each point belongs to.  Boxes stay a ragged list because
    the object count differs per frame.

    Args:
        batch: Sequence of sample dictionaries produced by the dataset.

    Returns:
        A batch dictionary with ``points (N,4)``, ``point_batch (N,)``,
        ``images (B,S,3,h,w)``, ``intrinsics (B,S,3,3)``, ``extrinsics (B,S,4,4)``,
        ``boxes`` (list of ``(Mi, 11)``), ``occupancy (B,Z,H,W)``,
        ``scene_id (B,)``, ``frame_id (B,)`` and ``tokens`` (list of str).
    """
    points = [torch.as_tensor(s["points"], dtype=torch.float32).reshape(-1, 4) for s in batch]
    counts = torch.tensor([p.shape[0] for p in points], dtype=torch.long)
    return {
        "points": torch.cat(points, dim=0),
        "point_batch": torch.repeat_interleave(torch.arange(len(batch)), counts),
        "images": torch.stack([torch.as_tensor(s["images"], dtype=torch.float32) for s in batch]),
        "intrinsics": torch.stack([torch.as_tensor(s["intrinsics"], dtype=torch.float32) for s in batch]),
        "extrinsics": torch.stack([torch.as_tensor(s["extrinsics"], dtype=torch.float32) for s in batch]),
        "boxes": [torch.as_tensor(s["boxes"], dtype=torch.float32).reshape(-1, 11) for s in batch],
        "occupancy": torch.stack([torch.as_tensor(s["occupancy"], dtype=torch.long) for s in batch]),
        "scene_id": torch.stack([torch.as_tensor(s["scene_id"], dtype=torch.long) for s in batch]),
        "frame_id": torch.stack([torch.as_tensor(s["frame_id"], dtype=torch.long) for s in batch]),
        "tokens": [str(s["token"]) for s in batch],
        "num_boxes": counts.new_tensor(
            [int(torch.as_tensor(s["boxes"]).reshape(-1, 11).shape[0]) for s in batch]
        ),
    }
