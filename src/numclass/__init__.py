from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Version
try:
    __version__ = _pkg_version("numclass")
except PackageNotFoundError:
    __version__ = "0+unknown"

# Public API re-exports
from .classify import classify
from .config import has_profile, load_settings, read_current_profile
from .registry import discover
from .runtime import APPLY, CFG
from .utility import build_ctx
from .workspace import workspace_dir

__all__ = [
    "APPLY",
    "CFG",
    "__version__",
    "build_ctx",
    "classify",
    "discover",
    "has_profile",
    "load_settings",
    "read_current_profile",
    "workspace_dir"
]
