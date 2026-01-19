# -----------------------------------------------------------------------------
# Curiosity constants (data-driven singleton / finite-set "constants")
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import io
import re
from collections.abc import Callable

from numclass.registry import classifier
from numclass.utility import _read_text_from_workspace_or_package

CATEGORY = "Mathematical Curiosities"
DATA_FILE = "curiosity_constants.tsv"


def _safe_ident(label: str, i: int) -> str:
    # Create a stable, import-safe python identifier.
    # Example: "Kaprekar Constant (4 digit)" -> "kaprekar_constant_4_digit_03"
    s = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    if not s:
        s = "curiosity_constant"
    if s[0].isdigit():
        s = "n_" + s
    return f"{s}_{i:03d}"


def _parse_int_list(s: str) -> set[int]:
    return {int(x) for x in (s or "").split(",") if x}


def _load_rows() -> list[dict[str, str]]:
    txt = _read_text_from_workspace_or_package(DATA_FILE)
    if not txt:
        return []

    lines = txt.lstrip("\ufeff").splitlines()

    # Keep non-empty lines; allow a commented header like "# label\t n\t ..."
    kept: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        # If it's a comment, only keep it if it looks like the header
        if s.startswith("#"):
            maybe = s.lstrip("#").strip()
            # header heuristic: must contain tabs and required column names
            if "\t" in maybe:
                cols = [c.strip().lower() for c in maybe.split("\t")]
                if "label" in cols and "n" in cols:
                    kept.append(maybe)  # "uncomment" header
            continue

        kept.append(ln)

    if not kept:
        return []

    f = io.StringIO("\n".join(kept))
    reader = csv.DictReader(f, delimiter="\t")

    rows: list[dict[str, str]] = []
    for r in reader:
        if not r:
            continue
        label = (r.get("label") or "").strip()
        n = (r.get("n") or "").strip()
        if not label or not n:
            continue
        rows.append(r)

    return rows


def _make_checker(values: set[int], details: str | None) -> Callable[[int, object], tuple[bool, str | None]]:
    details = (details or "").strip() or None

    def check(n: int, ctx=None) -> tuple[bool, str | None]:
        if int(n) in values:
            return True, details
        return False, None

    return check


# --- register one classifier per TSV row -------------------------------------

for i, row in enumerate(_load_rows(), start=1):
    label = (row.get("label") or "").strip()
    n_field = (row.get("n") or "").strip()
    desc = (row.get("description") or "").strip()
    details = (row.get("details") or "").strip()
    oeis = (row.get("oeis") or "").strip() or None

    values = _parse_int_list(n_field)
    fn = _make_checker(values, details)

    fn.__name__ = "is_" + _safe_ident(label, i)
    fn.__doc__ = desc or f"Data-driven curiosity constant: {label}"

    # Register in index
    fn = classifier(
        label=label,
        category=CATEGORY,
        description=desc,
        oeis=oeis,
    )(fn)

    # Export name into module globals so discovery sees it as a top-level callable
    globals()[fn.__name__] = fn
