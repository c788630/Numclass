from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib as toml
except Exception:
    import tomli as toml  # type: ignore

from numclass.utility import UserInputError
from numclass.workspace import ensure_workspace_seeded, workspace_dir


@dataclass
class Settings:
    """
    Wrap the full TOML dict (without the [PROFILE] section).
    .as_dict() feeds runtime.apply().

    Added fields:
      - name:        resolved profile name (FILE.stem if not provided in [PROFILE])
      - description: one-line description from [PROFILE] or "(no description)"
    """
    data: dict[str, Any]
    name: str
    description: str
    _source: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.data


# --- Paths -----------------------------------------------------------------

def _profiles_dir() -> Path:
    return workspace_dir() / "profiles"


def _profile_path(name: str) -> Path:
    return _profiles_dir() / f"{name}.toml"


# --- I/O -------------------------------------------------------------------


def _load_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as f:
            return toml.load(f)
    except Exception as e:
        lineno = getattr(e, "lineno", None)
        colno = getattr(e, "colno", None)
        msg = getattr(e, "msg", str(e))
        where = []
        if lineno is not None:
            where.append(f"line {lineno}")
        if colno is not None:
            where.append(f"column {colno}")
        loc = f" (at {', '.join(where)})" if where else ""
        # No traceback chaining
        raise UserInputError(f"reading {path.name}: {msg}{loc}.") from None


# --- Metadata handling -----------------------------------------------------


def _sanitize_oneline(s: str) -> str:
    return " ".join(str(s).split()) or "(no description)"


def _split_profile_data(raw: dict[str, Any], fallback_name: str) -> tuple[dict[str, Any], str, str]:
    """
    Extract [PROFILE] meta (name, description) and return:
      (settings_without_profile, resolved_name, resolved_description)
    """
    meta = raw.get("_PROFILE_") or {}
    # Remove [PROFILE] from the settings that go to runtime
    if "_PROFILE_" in raw:
        raw = {k: v for k, v in raw.items() if k != "PROFILE"}

    name = str(meta.get("name") or fallback_name)
    description = _sanitize_oneline(str(meta.get("description") or ""))

    return raw, name, description


# --- Public API ------------------------------------------------------------


def list_all_profiles() -> list[str]:
    """
    Return the list of available profile *names* (filename stems).
    """
    try:
        ensure_workspace_seeded()
    except Exception:
        pass
    pdir = _profiles_dir()
    if not pdir.exists():
        return []
    return sorted(p.stem for p in pdir.glob("*.toml"))


def list_profiles_with_descriptions() -> list[tuple[str, str]]:
    """
    Return [(name, description), ...] for all profiles.
    Profiles lacking [PROFILE] get "(no description)".
    """
    items: list[tuple[str, str]] = []
    for p in (_profiles_dir().glob("*.toml")):
        try:
            raw = _load_toml(p)
            _, nm, desc = _split_profile_data(raw, p.stem)
            items.append((nm, desc))
        except Exception:
            # Best-effort listing; fall back to filename
            items.append((p.stem, "(no description)"))
    return sorted(items, key=lambda t: t[0].lower())


def has_profile(name: str) -> bool:
    return _profile_path(name).exists()


def load_settings(name: str | None) -> Settings:
    """
    Load a profile by name (default 'default'), strip the [PROFILE] metadata,
    normalize CATEGORIES/CLASSIFIERS to {str: bool}, and return
    Settings(data=..., name=..., description=..., _source=path).
    """
    if not name:
        name = "default"

    path = _profile_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found at {path}")

    raw = _load_toml(path)

    # Pull out metadata (name/description) and remove [PROFILE] from settings
    data, resolved_name, description = _split_profile_data(raw, path.stem)

    # Sanitize booleans for the two heavy-use sections
    cats = data.get("CATEGORIES", {}) or {}
    labs = data.get("CLASSIFIERS", {}) or {}
    cats = {str(k): bool(v) for k, v in cats.items() if isinstance(v, bool)}
    labs = {str(k): bool(v) for k, v in labs.items() if isinstance(v, bool)}
    data["CATEGORIES"] = cats
    data["CLASSIFIERS"] = labs

    # Everything else remains type-preserving from TOML
    return Settings(
        data=data,
        name=resolved_name,
        description=description,
        _source=path,
    )


def _current_profile_path() -> Path:
    p = _profiles_dir()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p / ".current"


def read_current_profile() -> str | None:
    try:
        s = _current_profile_path().read_text(encoding="utf-8").strip()
        return s[:-5] if s.lower().endswith(".toml") else (s or None)
    except Exception:
        return None


def write_current_profile(name: str) -> None:
    nm = (name or "").strip()
    if nm.lower().endswith(".toml"):
        nm = nm[:-5]
    _current_profile_path().write_text(nm, encoding="utf-8")
