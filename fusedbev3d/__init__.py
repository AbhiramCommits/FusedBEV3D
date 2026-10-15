"""FusedBEV3D - multi-modal camera + lidar BEV perception.

Public entry points are re-exported lazily through :data:`__all__` and the
module-level ``__getattr__`` so that importing the bare package stays cheap and
does not require a working torch install just to read the version.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["__version__", "Config", "load_config", "build_dataset"]

_LAZY = {
    "Config": ("fusedbev3d.config", "Config"),
    "load_config": ("fusedbev3d.config", "load_config"),
    "build_dataset": ("fusedbev3d.data.registry", "build_dataset"),
}


def __getattr__(name: str) -> Any:
    """Resolve the lazily exported names on first attribute access."""
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
