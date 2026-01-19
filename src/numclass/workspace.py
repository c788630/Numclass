from __future__ import annotations

import os
import shutil
from importlib.resources import as_file
from importlib.resources import files as pkg_files
from pathlib import Path

SUBDIRS = ("profiles", "data", "classifiers")


def workspace_dir() -> Path:
    env = os.environ.get("NUMCLASS_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "Documents" / "Numclass").resolve()


def _should_copy_file(p: Path, sub: str) -> bool:
    # Skip caches/compiled/temporary/hidden files
    if any(part == "__pycache__" for part in p.parts):
        return False
    if p.suffix.lower() in {".pyc", ".pyo"}:
        return False
    if p.name.endswith("~"):
        return False
    # dotfiles in data are fine; in code dirs we usually skip hidden files
    if sub in {"classifiers", "profiles"} and p.name.startswith("."):
        return False
    # Subdir-specific allowlists
    if sub == "classifiers":
        return p.suffix.lower() == ".py"  # copy only .py (optionally include __init__.py)
    if sub == "profiles":
        return p.suffix.lower() == ".toml"
    # data: copy everything else (except filtered above)
    return True


def _copy_tree(src: Path, dst: Path, *, overwrite: bool, sub: str) -> int:
    count = 0
    if not src.exists():
        return 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if not _should_copy_file(p, sub):
            continue
        rel = p.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not target.exists():
            shutil.copy2(p, target)
            count += 1
    return count


def seed_workspace(*, overwrite: bool = False, subsets: set[str] | None = None) -> tuple[Path, dict[str, int]]:
    """
    Copy packaged sample files into the user's workspace.

    overwrite=False → copy-if-missing (normal users)
    overwrite=True  → force replace (dev use, guarded in CLI)
    subsets: optional subset of {"profiles","data","classifiers"} to limit copying

    Returns: (workspace_path, {section: files_copied})
    """
    root = workspace_dir()
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)

    parts = subsets or set(SUBDIRS)
    copied = {k: 0 for k in SUBDIRS}

    for sub in parts:
        ref = pkg_files("numclass") / sub
        try:
            with as_file(ref) as real:
                copied[sub] = _copy_tree(Path(real), root / sub, overwrite=overwrite, sub=sub)
        except Exception:
            copied[sub] = 0

    return root, copied


def ensure_workspace_seeded() -> tuple[Path, bool, dict[str, int]]:
    root = workspace_dir()
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    _, copied = seed_workspace(overwrite=False)
    seeded_any = any(copied.values())
    return root, seeded_any, copied
