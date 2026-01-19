# src/numclass/dataio.py
from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file
from importlib.resources import files as pkg_files
from pathlib import Path

from numclass.workspace import workspace_dir

try:
    import tomllib as _toml  # py311+
except Exception:  # pragma: no cover
    import tomli as _toml  # type: ignore


def _workspace() -> Path | None:
    """
    Returns the user workspace root (e.g., ~/Documents/Numclass on Win/Linux),
    or None if the workspace module isn't available.
    """
    try:
        return workspace_dir()
    except Exception:
        return None


def data_path(rel: str) -> Path:
    """
    Resolve a data file path with override semantics:

      1) <Workspace>/data/<rel>  (if present)
      2) Packaged resource: numclass/data/<rel>

    Returns a filesystem Path you can open.
    """
    rel = rel.lstrip("/\\")
    ws = _workspace()
    if ws:
        p = ws / "data" / rel
        if p.exists():
            return p

    ref = pkg_files("numclass") / "data" / rel
    # materialize to a real path (needed for zip/egg resources)
    with as_file(ref) as real:
        return Path(real)


# ------------------------ OEIS b-file helpers ------------------------

def _oeis_digits(oeis: str) -> str:
    s = oeis.strip().upper()
    if not s.startswith("A") or not s[1:].isdigit():
        raise ValueError(f"Bad OEIS id: {oeis!r}")
    return s[1:].zfill(6)


def load_fun_numbers() -> dict[int, str]:
    """
    Read fun_numbers.tsv (TSV with at least two columns):
        <number>\t<description>[...]
    Returns { number: description } using column 2 as the detail text.
    Ignores blank lines and lines starting with '#'.
    """
    p = data_path("fun_numbers.tsv")
    mapping: dict[int, str] = {}
    _COL2 = 2
    try:
        with p.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line or line.lstrip().startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < _COL2:
                    continue
                try:
                    k = int(parts[0].strip())
                except ValueError:
                    continue
                desc = parts[1].strip()
                mapping[k] = desc
    except FileNotFoundError:
        # Safe fallback: empty mapping (classifier will just return False)
        mapping = {}
    return mapping


@lru_cache(maxsize=1)
def load_intersections() -> list[dict]:
    """
    Canonical format only:

      [[intersections]]
      label    = "Palindromic Harshad number"
      of       = ["Palindrome", "Harshad number"]   # atomic *display labels*
      category = "Digit-based"                      # optional
      description = "..."                           # optional
      oeis = "Axxxxx"                               # optional
    """
    p: Path = data_path("intersections.toml")
    try:
        with p.open("rb") as f:
            doc = _toml.load(f)
    except FileNotFoundError:
        return []

    raw = doc.get("intersections")
    if not isinstance(raw, list):
        return []

    items: list[dict] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        label = it.get("label")
        of = it.get("of")
        if not isinstance(label, str) or not isinstance(of, list) or not all(isinstance(s, str) for s in of):
            continue
        items.append({
            "label": label,
            "of": of,
            "category": it.get("category") or "Uncategorized",
            "description": it.get("description"),
            "oeis": it.get("oeis"),
        })
    return items
