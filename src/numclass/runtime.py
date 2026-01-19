# runtime.py
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict as _asdict
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

from colorama import Fore, Style


@dataclass
class Runtime:
    profile_name: str = "default"
    settings: dict[str, Any] = field(default_factory=dict)
    debug: bool = False  # controls verbosity / tracebacks
    fast_mode: bool = True  # True = skip expensive work

    def apply(self, settings: Any) -> None:
        self.profile_name = (
            getattr(settings, "name", None)
            or getattr(settings, "_source", None)
            or "default"
        )

        if hasattr(settings, "as_dict") and callable(settings.as_dict):
            cfg = settings.as_dict()
        elif isinstance(settings, dict):
            cfg = settings
        else:
            # grab UPPERCASE attributes from simple objects / modules
            cfg = {k: getattr(settings, k) for k in dir(settings) if k.isupper()}

        try:
            self.settings = dict(cfg)  # ensure plain dict
        except Exception:
            self.settings = _asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else {}

        # --- NEW: sync runtime flags from profile ---------------------------
        fm = self.get("BEHAVIOUR.FAST_MODE", None)
        if isinstance(fm, bool):
            self.fast_mode = fm

        dbg = self.get("BEHAVIOUR.DEBUG", None)
        if isinstance(dbg, bool):
            self.debug = dbg

    def get(self, key: str, default: Any = None) -> Any:
        """Support dotted lookups, e.g., 'FEATURE_FLAGS.SHOW_SKIPPED'."""
        if not key:
            return default
        cur = self.settings
        if isinstance(key, str) and "." in key:
            for part in key.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return default
            return cur
        return cur.get(key, default)


# --- Context management ---

_current_runtime: ContextVar[Runtime | None] = ContextVar("numclass_runtime", default=None)


def current() -> Runtime:
    rt = _current_runtime.get()
    if rt is None:
        rt = Runtime()
        _current_runtime.set(rt)
    return rt


def APPLY(settings: Any) -> None:
    current().apply(settings)


def CFG(key: str, default: Any = None) -> Any:
    return current().get(key, default)


# ---- Dependency check --------------------------------------------------------

def ensure_runtime_deps(strict: bool = True) -> bool:
    """
    Verify core runtime deps are available. Uses find_spec() to avoid importing
    inside this function (satisfies PLC0415).
    If strict=True, prints a friendly error and returns False when missing.
    """
    required = ("sympy", "gmpy2")
    missing = [name for name in required if find_spec(name) is None]

    if not missing:
        return True

    msg = (
        f"{Fore.RED}{Style.BRIGHT}\nMissing dependencies:{Style.RESET_ALL} "
        + ", ".join(missing)
        + "\nInstall with: "
        + f"{Fore.YELLOW}pip install " + " ".join(missing) + f"{Style.RESET_ALL}"
    )
    print(msg)
    return not strict
