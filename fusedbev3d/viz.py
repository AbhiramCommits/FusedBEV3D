"""Visualisation: matplotlib BEV overlays and an import-guarded Open3D viewer.

The matplotlib path only needs matplotlib (a hard dependency) and writes PNGs that
are embedded in the README.  The Open3D path is entirely optional: if ``open3d``
is not installed the functions report that clearly instead of raising, so the
headline results never depend on the ``[viz]`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fusedbev3d.config import GridConfig

__all__ = [
    "CLASS_COLORS",
    "OCCUPANCY_COLORS",
    "render_bev",
    "render_qualitative_frame",
    "open3d_available",
    "open3d_viewer",
]

CLASS_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.85, 0.24, 0.24),  # car
    (0.25, 0.47, 0.90),  # pedestrian
    (0.94, 0.78, 0.24),  # cyclist
)

#: free / occupied / unknown
OCCUPANCY_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.16, 0.16, 0.20),
    (0.95, 0.45, 0.15),
    (0.55, 0.55, 0.60),
)


def _as_boxes(boxes: Any) -> np.ndarray:
    """Normalise a box-like object (tensor, array, detection dict) to ``(N, >=7)`` numpy."""
    if isinstance(boxes, dict):
        boxes = boxes.get("boxes", np.zeros((0, 9)))
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 9))
    return arr.reshape(-1, arr.shape[-1])


def _box_polygon(box: np.ndarray) -> np.ndarray:
    """Return the ``(5, 2)`` closed BEV footprint of a box ``x y z l w h yaw ...``."""
    x, y, length, width, yaw = box[0], box[1], box[3], box[4], box[6]
    corners = np.array(
        [[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2],
         [-length / 2, width / 2], [-length / 2, -width / 2]]
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([x, y])


def render_bev(
    sample: dict[str, Any],
    detections: Any,
    occupancy: Any,
    path: str | Path,
    grid: GridConfig | None = None,
    occupancy_pred: Any = None,
    title: str | None = None,
    dpi: int = 120,
) -> Path:
    """Render a two-panel BEV figure: boxes over lidar points, and the occupancy grid.

    Args:
        sample: A dataset sample dict (``points``, ``boxes``, ``occupancy``).
        detections: Predicted boxes: a ``(K, 9)`` array/dict/tensor, or a list of
            detection dicts.
        occupancy: Ground-truth occupancy volume ``(Z, H, W)``.
        path: Output PNG path (parent directories are created).
        grid: Grid geometry (defaults to :class:`GridConfig`).
        occupancy_pred: Optional predicted occupancy volume ``(Z, H, W)`` for a third panel.
        title: Optional figure title.
        dpi: Output resolution.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    grid = grid or GridConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = 3 if occupancy_pred is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 6.0))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    pts = np.asarray(sample["points"], dtype=np.float64)
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], s=0.6, c=np.clip(pts[:, 2], 0, 2), cmap="viridis", alpha=0.5)
    for row in _as_boxes(sample["boxes"]):
        cls = int(row[9])
        ax.add_patch(
            Polygon(_box_polygon(row), closed=True, fill=False, edgecolor="lime", linewidth=1.6,
                    linestyle="--", label="GT" if cls == 0 else None)
        )
    preds = (
        np.stack([d["box"] for d in detections])
        if isinstance(detections, (list, tuple)) and detections and isinstance(detections[0], dict)
        else _as_boxes(detections)
    )
    for row in preds:
        color = CLASS_COLORS[min(int(row[9]) if row.shape[0] > 9 else 0, 2)]
        ax.add_patch(
            Polygon(_box_polygon(row), closed=True, fill=True, alpha=0.35, edgecolor=color, linewidth=1.6)
        )
    ax.set_xlim(grid.x_range)
    ax.set_ylim(grid.y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x forward [m]")
    ax.set_ylabel("y left [m]")
    ax.set_title("BEV: lidar + GT (dashed) vs predicted (filled)")
    ax.grid(alpha=0.25)

    _draw_occupancy(axes[1], occupancy, grid, "occupancy ground truth")
    if occupancy_pred is not None:
        _draw_occupancy(axes[2], occupancy_pred, grid, "occupancy prediction")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _draw_occupancy(ax, occupancy: Any, grid: GridConfig, title: str) -> None:
    """Draw one occupancy volume collapsed along ``z`` as a class-coloured image."""
    vol = np.asarray(occupancy)
    if vol.ndim == 4:
        vol = vol[0]
    # collapse z: occupied wins over free wins over unknown
    collapsed = np.full(vol.shape[-2:], 2, dtype=np.int64)
    for z in range(vol.shape[0]):
        layer = vol[z]
        collapsed = np.where(layer == 0, 0, collapsed)
        collapsed = np.where(layer == 1, 1, collapsed)
    img = np.zeros((*collapsed.shape, 3))
    for cls, color in enumerate(OCCUPANCY_COLORS):
        img[collapsed == cls] = color
    ax.imshow(
        img,
        origin="lower",
        extent=(grid.x_range[0], grid.x_range[1], grid.y_range[0], grid.y_range[1]),
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("x forward [m]")
    ax.set_ylabel("y left [m]")
    handles = [
        plt_patch(color, name)
        for name, color in zip(("free", "occupied", "unknown"), OCCUPANCY_COLORS, strict=False)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)


def plt_patch(color, label):
    """Build a matplotlib legend patch for a colour/label pair."""
    from matplotlib.patches import Patch

    return Patch(facecolor=color, edgecolor="black", label=label)


def render_qualitative_frame(
    model,
    batch: dict[str, Any],
    index: int,
    path: str | Path,
    grid: GridConfig,
    modalities: Sequence[str] | str | None = None,
    config: dict[str, Any] | None = None,
    title: str | None = None,
) -> Path:
    """Decode one frame of a batch and render the BEV overlay to ``path``.

    Args:
        model: A loaded :class:`~fusedbev3d.models.fusedbev3d.FusedBEV3D`.
        batch: A collated batch (already on the right device).
        index: Sample index inside the batch.
        path: Output PNG path.
        grid: Grid geometry.
        modalities: Optional modality override.
        config: Optional config dict (used for the image tensor normalisation only).
        title: Optional figure title.

    Returns:
        The written path.
    """
    del config
    sample = {
        "points": batch["points"][batch["point_batch"] == index].cpu().numpy(),
        "boxes": batch["boxes"][index].cpu().numpy(),
        "occupancy": batch["occupancy"][index].cpu().numpy(),
    }
    with_nograd = _no_grad()
    with with_nograd:
        out = model(batch, modalities=modalities, decode=True, topk=100)
    det = out["detections"]
    keep = det["keep"][index]
    preds = det["boxes"][index][keep].cpu().numpy()
    if preds.shape[0]:
        preds = np.concatenate([preds, det["classes"][index][keep].cpu().numpy()[:, None]], axis=1)
    occ_pred = model.occupancy_head.predict(out["occupancy"])[index].cpu().numpy()
    token = batch["tokens"][index] if "tokens" in batch else str(index)
    return render_bev(
        sample,
        preds,
        sample["occupancy"],
        path,
        grid=grid,
        occupancy_pred=occ_pred,
        title=title or f"{token}",
    )


def _no_grad():
    """Return a ``torch.no_grad`` context (isolated so viz stays import-light)."""
    import torch

    return torch.no_grad()


def open3d_available() -> bool:
    """Return whether the optional ``open3d`` package can be imported."""
    try:
        import open3d  # noqa: F401

        return True
    except ImportError:
        return False


def open3d_viewer(
    points: np.ndarray,
    boxes: Iterable[np.ndarray] = (),
    labels: Iterable[Any] | None = None,
    colors: Iterable[tuple[float, float, float]] | None = None,
    window_name: str = "FusedBEV3D",
) -> Any:
    """Show a lidar cloud with oriented boxes in an Open3D window.

    This is a convenience viewer for interactive use; it is *not* used by any
    script that produces a reported number.

    Args:
        points: ``(N, 4)`` lidar points ``x y z intensity``.
        boxes: Iterable of ``(9,)`` boxes ``x y z l w h yaw vx vy``.
        labels: Optional per-box labels (unused, kept for API symmetry).
        colors: Optional per-box RGB colours in ``[0, 1]``.
        window_name: Open3D window title.

    Returns:
        The Open3D ``Visualizer`` if ``open3d`` is installed, otherwise ``None``
        (with a printed hint).
    """
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is not installed; skipping the 3D viewer. Install with: pip install -e '.[viz]'")
        return None
    del labels
    pts = np.asarray(points, dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts[:, :3])
    if pts.shape[1] > 3:
        shade = np.clip(pts[:, 3], 0, 1)
        cloud.colors = o3d.utility.Vector3dVector(np.stack([shade, 1 - shade, 0.4 * shade], axis=1))
    geoms: list[Any] = [cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)]
    color_list = list(colors) if colors is not None else None
    for i, box in enumerate(boxes):
        color = color_list[i] if color_list else CLASS_COLORS[min(int(box[9]) if len(box) > 9 else 0, 2)]
        geoms.append(_o3d_box(o3d, np.asarray(box, dtype=np.float64), color))
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()
    return vis


def _o3d_box(o3d, box: np.ndarray, color: tuple[float, float, float]) -> Any:
    """Build an Open3D oriented bounding-box line set for one box."""
    length, width, height = box[3], box[4], box[5]
    mesh = o3d.geometry.TriangleMesh.create_box(length, width, height)
    mesh.translate((-length / 2, -width / 2, -height / 2))
    rot = mesh.get_rotation_matrix_from_xyz((0.0, 0.0, -box[6]))
    mesh.rotate(rot, center=(0, 0, 0))
    mesh.translate((box[0], box[1], box[2]))
    obj = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    obj.paint_uniform_color(list(color))
    return obj
