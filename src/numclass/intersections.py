# intersections.py

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from numclass.dataio import data_path
from numclass.runtime import current as _rt_current
from numclass.utility import _token

# only these keys are allowed inside [[intersections]]
_ALLOWED_KEYS = {"of", "label", "category", "description", "oeis"}


@dataclass(frozen=True)
class Intersection:
    label: str
    token: str
    requires: list[str]
    category: str | None = None
    description: str = ""
    oeis: str | None = None


def _as_tokens(labels: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for name in labels:
        tok = _token(str(name))
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def load(workspace: Path | None = None) -> list[Intersection]:
    # read debug from runtime
    rt = _rt_current()
    debug = bool(getattr(rt, "debug", False))

    # 1) read TOML from the standard location(s)
    candidates = ["intersections.toml", "intersections/intersections.toml"]
    src_path: Path | None = None
    tried = []
    for rel in candidates:
        try:
            p = data_path(rel)
            tried.append(str(p))
            if p.exists():
                src_path = p
                break
        except Exception as e:
            tried.append(f"{rel} [resolve error: {type(e).__name__}: {e}]")

    if src_path is None:
        if debug:
            print("[discovery] intersections: none found; checked: " + " | ".join(tried), file=sys.stderr)
        return []

    # 2) parse TOML (strict: require top-level [[intersections]])
    try:
        data = tomllib.loads(src_path.read_text(encoding="utf-8"))
    except Exception as e:
        if debug:
            print(f"[discovery] intersections FAIL parse {src_path}: {e!r}", file=sys.stderr)
        return []

    raw_items = data.get("intersections")
    if not isinstance(raw_items, list):
        if debug:
            print("[discovery] intersections FAIL: expected [[intersections]] array of tables.", file=sys.stderr)
        return []

    out: list[Intersection] = []
    for i, item in enumerate(raw_items, 1):
        if not isinstance(item, dict):
            if debug:
                print(f"[discovery] intersections SKIP item {i}: must be a table", file=sys.stderr)
            continue

        # STRICT: unknown keys
        unknown = set(item.keys()) - _ALLOWED_KEYS
        if unknown:
            if debug:
                print(f"[discovery] intersections SKIP item {i}: unknown keys {sorted(unknown)}", file=sys.stderr)
            continue

        # Required fields
        of = item.get("of")
        label = item.get("label")
        if not isinstance(of, list) or not label or not all(isinstance(x, str) for x in of):
            if debug:
                print(f"[discovery] intersections SKIP item {i}: requires 'of' (list[str]) and 'label' (str)", file=sys.stderr)
            continue

        req = _as_tokens(of)
        if not req:
            if debug:
                print(f"[discovery] intersections SKIP item {i}: empty 'of' after tokenization", file=sys.stderr)
            continue

        cat = item.get("category")
        desc = item.get("description") or ""
        oeis = item.get("oeis")

        out.append(Intersection(
            label=str(label),
            token=_token(str(label)),
            requires=req,
            category=str(cat) if isinstance(cat, str) else None,
            description=str(desc),
            oeis=str(oeis) if isinstance(oeis, str) else None,
        ))

    # de-dup by token (first wins)
    seen = set()
    uniq = [r for r in out if not (r.token in seen or seen.add(r.token))]
    if debug:
        print(f"[discovery] intersections OK: {len(uniq)} from {src_path}", file=sys.stderr)
    return uniq
