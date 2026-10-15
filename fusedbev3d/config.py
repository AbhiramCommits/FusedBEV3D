"""Central, dataclass-based configuration for FusedBEV3D.

The whole stack is configured by four nested dataclasses:

* :class:`GridConfig`   - the shared BEV / voxel grid geometry.
* :class:`ModelConfig`  - encoder, fusion and head hyper-parameters.
* :class:`DataConfig`   - dataset / procedural generator parameters.
* :class:`TrainConfig`  - optimisation, schedule and runtime parameters.

:func:`load_config` starts from the defaults and applies YAML/JSON overrides, which
may be nested (``train.epochs``) or given as nested dictionaries.  This keeps every
experiment reproducible: a config file plus a seed fully determines a run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "GridConfig",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "Config",
    "load_config",
    "save_config",
]


@dataclass
class GridConfig:
    """Geometry of the shared bird's-eye-view grid and the occupancy volume.

    The BEV feature map is ``(C, bev_h, bev_w)`` where ``bev_h`` indexes the
    **y** axis (lateral) and ``bev_w`` indexes the **x** axis (forward).  The
    occupancy volume is ``(num_z, bev_h, bev_w)`` with ``2`` meaning *unknown*
    (unobserved by any lidar ray).

    Attributes:
        x_range: ``(min, max)`` forward extent in metres (ego frame).
        y_range: ``(min, max)`` lateral extent in metres (ego frame).
        z_range: ``(min, max)`` vertical extent in metres (ego frame).
        voxel_size: BEV cell size in metres (square cells in x/y).
        z_voxel_size: Occupancy voxel height in metres.
        bev_stride: Output stride of the BEV backbone relative to cell size.
    """

    x_range: tuple[float, float] = (-16.0, 16.0)
    y_range: tuple[float, float] = (-16.0, 16.0)
    z_range: tuple[float, float] = (-0.5, 2.0)
    voxel_size: float = 0.5
    z_voxel_size: float = 0.5
    bev_stride: int = 1

    @property
    def bev_w(self) -> int:
        """Number of BEV cells along the forward (x) axis."""
        return int(round((self.x_range[1] - self.x_range[0]) / self.voxel_size))

    @property
    def bev_h(self) -> int:
        """Number of BEV cells along the lateral (y) axis."""
        return int(round((self.y_range[1] - self.y_range[0]) / self.voxel_size))

    @property
    def num_z(self) -> int:
        """Number of vertical occupancy levels."""
        return int(round((self.z_range[1] - self.z_range[0]) / self.z_voxel_size))

    @property
    def center(self) -> tuple[float, float]:
        """BEV centre in metres (``None`` extent handling aside, ranges are symmetric)."""
        return (
            0.5 * (self.x_range[0] + self.x_range[1]),
            0.5 * (self.y_range[0] + self.y_range[1]),
        )

    def describe(self) -> str:
        """Return a one-line human readable summary (used in logs/README)."""
        return (
            f"BEV {self.bev_h}x{self.bev_w} cells @ {self.voxel_size} m "
            f"({self.x_range[0]:.0f}..{self.x_range[1]:.0f} m x, "
            f"{self.y_range[0]:.0f}..{self.y_range[1]:.0f} m y), "
            f"occupancy {self.num_z}x{self.bev_h}x{self.bev_w} @ {self.z_voxel_size} m z"
        )


@dataclass
class ModelConfig:
    """Encoder / fusion / head hyper-parameters.

    Attributes:
        num_classes: Number of object classes (car, pedestrian, cyclist).
        pillar_feat: Output channels of the PointPillars branch.
        max_points_per_pillar: Point cap per pillar (rest are dropped, like PointPillars).
        max_pillars: Pillar cap (0 == keep every occupied pillar).
        camera_context: LSS per-pixel context feature width.
        depth_bins: Number of discrete depth bins for the LSS frustum.
        depth_min: Nearest depth bin edge in metres.
        depth_max: Farthest depth bin edge in metres.
        fuse_channels: Width of the fused BEV feature map.
        bev_backbone_channels: Width of the BEV FPN backbone.
        head_width: Hidden width of the shared head stem.
        reg_channels: Per-cell regression channels (offset, z, dims, rot, vel).
    """

    num_classes: int = 3
    class_names: tuple[str, ...] = ("car", "pedestrian", "cyclist")

    pillar_feat: int = 64
    max_points_per_pillar: int = 32
    max_pillars: int = 0

    camera_context: int = 48
    depth_bins: int = 8
    depth_min: float = 2.0
    depth_max: float = 30.0

    fuse_channels: int = 64
    bev_backbone_channels: int = 64
    head_width: int = 64
    reg_channels: int = 10

    occupancy_classes: int = 3


@dataclass
class DataConfig:
    """Dataset and procedural generator parameters.

    Attributes:
        name: ``"synth"`` or ``"nuscenes-mini"``.
        num_scenes: Number of *training* scenes for the synthetic generator.
        val_scenes: Number of *validation* scenes for the synthetic generator.
        frames_per_scene: Temporal length of every generated sequence.
        seed: Master seed; every scene derives its own RNG from it.
        min_actors: Minimum simultaneous actors per frame.
        max_actors: Maximum simultaneous actors per frame.
        num_beams: Lidar elevation channels.
        num_azimuth: Azimuth samples per revolution.
        max_range: Lidar maximum range in metres.
        range_noise: Standard deviation of per-return range noise in metres.
        dropout: Probability that a valid return is dropped.
        image_height: Camera image height in pixels.
        image_width: Camera image width in pixels.
        camera_fov: Horizontal field of view of every camera in degrees.
        occupancy_range: Range (metres) out to which free space is carved.
        cache_dir: Directory for materialised split caches.
        dataroot: nuScenes data root (only used by the nuScenes adapter).
        version: nuScenes version string.
    """

    name: str = "synth"
    num_scenes: int = 40
    val_scenes: int = 10
    frames_per_scene: int = 20
    seed: int = 1234
    min_actors: int = 3
    max_actors: int = 8

    num_beams: int = 32
    num_azimuth: int = 192
    max_range: float = 50.0
    range_noise: float = 0.02
    dropout: float = 0.08
    elevation_min_deg: float = -25.0
    elevation_max_deg: float = 3.0

    image_height: int = 64
    image_width: int = 192
    camera_fov: float = 90.0

    occupancy_range: float = 20.0
    occupancy_step: float = 0.35
    cache_dir: str = "data/cache"

    dataroot: str | None = None
    version: str = "v1.0-mini"


@dataclass
class TrainConfig:
    """Optimisation, schedule and runtime parameters.

    Attributes:
        epochs: Number of passes over the training split.
        lr: Peak AdamW learning rate.
        weight_decay: AdamW decoupled weight decay.
        batch_size: Per-process batch size.
        amp: Use CUDA automatic mixed precision when CUDA is available.
        modalities: Enabled sensing modalities, a subset of ``("camera", "lidar")``.
        warmup_frac: Fraction of total steps used for linear LR warmup.
        grad_clip: Gradient-norm clipping value (``<=0`` disables).
        num_workers: DataLoader worker processes.
        seed: Global training seed.
        name: Run name; checkpoints go to ``runs/<name>``.
        out: Output directory (overrides ``name`` when set).
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        log_every: Logging interval in steps.
        eval_every: Validation interval in epochs (0 disables mid-training eval).
        max_steps: Optional global step budget override (used by the ablation).
        loss_weights: Per-loss-term weights.
    """

    epochs: int = 12
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    amp: bool = True
    modalities: tuple[str, ...] = ("camera", "lidar")
    warmup_frac: float = 0.05
    grad_clip: float = 5.0
    num_workers: int = 0
    seed: int = 1234
    name: str = "fused"
    out: str | None = None
    device: str = "auto"
    log_every: int = 20
    eval_every: int = 0
    max_steps: int = 0
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "heatmap": 1.0,
            "offset": 1.0,
            "z": 1.0,
            "dims": 1.0,
            "rot": 1.0,
            "vel": 0.2,
            "occupancy": 1.0,
        }
    )

    def __post_init__(self) -> None:
        """Normalise ``modalities`` so configs round-trip cleanly through YAML."""
        self.modalities = tuple(self.modalities)

    def resolved_out(self) -> Path:
        """Return the run directory (``out`` if set, else ``runs/<name>``)."""
        return Path(self.out) if self.out else Path("runs") / self.name


@dataclass
class Config:
    """Bundle of every configuration group."""

    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of the whole config."""
        out = asdict(self)
        out["train"]["modalities"] = list(self.train.modalities)
        out["model"]["class_names"] = list(self.model.class_names)
        return out


_GROUPS: Mapping[str, type] = {
    "grid": GridConfig,
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
}


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce a YAML/JSON scalar into the annotation of a dataclass field."""
    if target_type is tuple and isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(value)
    if isinstance(value, float) and target_type is not float:
        return value
    return value


def _apply_group(group: Any, overrides: Mapping[str, Any]) -> None:
    """Apply ``{field: value}`` overrides to a dataclass instance in place."""
    valid = {f.name for f in fields(group)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(
                f"Unknown config key {key!r} for {type(group).__name__}. Valid keys: {sorted(valid)}"
            )
        if is_dataclass(getattr(group, key)) and isinstance(value, Mapping):
            _apply_group(getattr(group, key), value)
        else:
            setattr(group, key, value)
    if hasattr(group, "__post_init__"):
        group.__post_init__()


def _flatten(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Expand ``{"train.epochs": 3}`` style keys into nested dicts."""
    nested: dict[str, Any] = {}
    for key, value in overrides.items():
        if "." in key and not isinstance(value, Mapping):
            head, _, tail = key.partition(".")
            nested.setdefault(head, {})[tail] = value
        elif isinstance(value, Mapping):
            nested.setdefault(key, {}).update(value)
        else:
            nested[key] = value
    # merge dotted and mapping forms
    for key, value in list(nested.items()):
        if isinstance(value, Mapping):
            flat_children = _flatten(value)
            nested[key] = flat_children
    return nested


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Build a :class:`Config`, applying YAML/JSON file contents then keyword overrides.

    Args:
        path: Optional ``.yaml``/``.yml``/``.json`` file holding partial overrides.
            It may be flat (``train.epochs: 5``), nested (``train: {epochs: 5}``),
            or a mix.
        **overrides: Extra overrides applied after the file, same syntax as the file.

    Returns:
        A fully populated :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        ValueError: If the file suffix is neither YAML nor JSON.
    """
    cfg = Config()
    payload: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        text = p.read_text()
        if p.suffix in {".yaml", ".yml"}:
            import yaml

            payload = yaml.safe_load(text) or {}
        elif p.suffix == ".json":
            payload = json.loads(text)
        else:
            raise ValueError(f"unsupported config suffix {p.suffix!r} (use .yaml/.yml/.json)")
    payload = {**payload, **overrides}
    for group_name, group_overrides in _flatten(payload).items():
        if group_name not in _GROUPS:
            raise KeyError(
                f"Unknown config group {group_name!r}. Valid groups: {sorted(_GROUPS)}"
            )
        if not isinstance(group_overrides, Mapping):
            raise ValueError(f"Config group {group_name!r} must map to a dictionary")
        _apply_group(getattr(cfg, group_name), group_overrides)
    return cfg


def save_config(cfg: Config, path: str | Path) -> Path:
    """Write ``cfg`` to ``path`` (YAML when available, else JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.to_dict()
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        path.write_text(yaml.safe_dump(payload, sort_keys=False))
    else:
        path.write_text(json.dumps(payload, indent=2))
    return path
