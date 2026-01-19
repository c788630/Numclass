# src/numclass/registry.py
from __future__ import annotations

import inspect
import re
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from importlib.resources import as_file
from importlib.resources import files as pkg_files
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from numclass.dataio import load_intersections
from numclass.utility import _token

limits: dict[str, int] = {}


@dataclass
class IntersectionRule:
    label: str
    category: str
    requires: tuple[str, ...]           # tokens of atomic labels
    description: str | None = None
    oeis: str | None = None

# --------------------- Discovery → Index (immutable) ----------------------


@dataclass
class Index:
    funcs: dict[str, callable]                 # atomic label -> func
    categories: dict[str, str]                 # atomic label -> category
    descriptions: dict[str, str]               # label -> short description
    oeis: dict[str, str | None]             # label -> A-code or None
    label_to_token: dict[str, str]             # label -> TOKEN
    intersections: list[IntersectionRule] = field(default_factory=list)
    limits: dict[str, int] = field(default_factory=dict)

    # helper for tokenization identical to profiles/tokens rule
    @staticmethod
    def to_token(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", name).upper().strip("_")


# --- rich report of the discovery step ---
@dataclass
class DiscoveryReport:
    # per-source results
    ws_loaded: list[tuple[str, int]] = field(default_factory=list)         # (filename.py, count)
    ws_failed: list[tuple[str, str]] = field(default_factory=list)         # (filename.py, error)
    pkg_loaded: list[tuple[str, int]] = field(default_factory=list)        # (module.name, count)
    pkg_failed: list[tuple[str, str]] = field(default_factory=list)        # (module.name, error)

    # label-level details
    added: list[tuple[str, str]] = field(default_factory=list)             # (label, source)
    skipped_duplicates: list[tuple[str, str, str]] = field(default_factory=list)  # (label, skipped_source, kept_source)

    # labels per module/file for printing
    module_labels: dict[str, list[str]] = field(default_factory=dict)      # source -> [labels]


def _is_classifier(obj) -> bool:
    return callable(obj) and getattr(obj, "__is_classifier__", False)


def _import_module_from_file(path: Path, name_hint: str):
    spec = spec_from_file_location(name_hint, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot import {path}")
    mod = module_from_spec(spec)
    sys.modules[name_hint] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _collect_from_module(mod) -> list[Callable[[int], object]]:
    out = []
    for _, o in inspect.getmembers(mod):
        if _is_classifier(o):
            out.append(o)
    return out


# ---------- Decorator (only tags the function; no side effects) ----------


def classifier(*, label: str, category: str, description: str = "",
               oeis: str | None = None, limit: int | None = None):
    def deco(fn: Callable[[int], object]):
        fn.__is_classifier__ = True
        fn.label = label
        fn.category = category
        fn.description = description
        if oeis is not None:
            fn.oeis = oeis
        if limit is not None:
            fn.limit = int(limit)
        return fn
    return deco


def discover(workspace: Path | None = None) -> Index:
    """Discover classifiers from workspace and package; workspace overrides package by label."""
    funcs: OrderedDict[str, Callable[[int], object]] = OrderedDict()
    cats: dict[str, str] = {}
    desc: dict[str, str] = {}
    toks: dict[str, str] = {}
    refs: dict[str, str | None] = {}

    # 1) Workspace (*.py)
    if workspace:
        ws_dir = workspace / "classifiers"
        if ws_dir.is_dir():
            for file in sorted(ws_dir.glob("*.py")):
                if file.name == "__init__.py":
                    continue
                modname = f"_nc_user_cls_{file.stem}"
                try:
                    mod = _import_module_from_file(file, modname)
                    for fn in _collect_from_module(mod):
                        label = fn.label
                        if label in funcs:                # keep first (workspace file order stable)
                            continue
                        funcs[label] = fn
                        cats[label] = getattr(fn, "category", "General")
                        desc[label] = getattr(fn, "description", "")
                        toks[label] = _token(label)
                        refs[label] = getattr(fn, "oeis", None)
                except Exception:
                    # Skip broken module; don't crash the app
                    continue

    # 2) Packaged (numclass.classifiers.*)
    try:
        pkg_dir = pkg_files("numclass") / "classifiers"
        with as_file(pkg_dir) as real:
            for file in sorted(Path(real).glob("*.py")):
                if file.name == "__init__.py":
                    continue
                modname = f"numclass.classifiers.{file.stem}"
                try:
                    mod = import_module(modname)
                    for fn in _collect_from_module(mod):
                        label = fn.label
                        if label in funcs:                # workspace already provided this label
                            continue
                        funcs[label] = fn
                        cats[label] = getattr(fn, "category", "General")
                        desc[label] = getattr(fn, "description", "")
                        toks[label] = _token(label)
                        refs[label] = getattr(fn, "oeis", None)
                        lim = getattr(fn, "limit", None)
                        if isinstance(lim, int):
                            limits[label] = lim
                except Exception:
                    continue
    except Exception:
        pass

    idx = Index(
        funcs=funcs,
        categories=cats,
        descriptions=desc,
        oeis=refs,
        label_to_token=toks,   # IMPORTANT: name must be label_to_token
        limits=limits,
    )
    idx = _attach_intersections(idx)
    return idx


def discover_with_report(workspace: Path | None = None) -> tuple[Index, list[str]]:
    """Discover classifiers with a detailed report (loaded, failed, skipped)."""
    report = DiscoveryReport()

    funcs: OrderedDict[str, Callable[[int], object]] = OrderedDict()
    cats: dict[str, str] = {}
    desc: dict[str, str] = {}
    toks: dict[str, str] = {}
    refs: dict[str, str | None] = {}

    label_source: dict[str, str] = {}  # label -> source string ("ws: file.py" or "pkg: module")

    def _add_from_module(mod, source_name: str):
        found = 0
        for fn in _collect_from_module(mod):
            label = getattr(fn, "label", getattr(fn, "__name__", "UNKNOWN"))
            if label in funcs:
                report.skipped_duplicates.append((label, source_name, label_source[label]))
                continue
            funcs[label] = fn
            cats[label] = getattr(fn, "category", "General")
            desc[label] = getattr(fn, "description", "")
            toks[label] = _token(label)
            refs[label] = getattr(fn, "oeis", None)
            lim = getattr(fn, "limit", None)
            if isinstance(lim, int):
                limits[label] = lim
            label_source[label] = source_name
            report.added.append((label, source_name))
            report.module_labels.setdefault(source_name, []).append(label)
            found += 1
        return found

    # 1) Workspace (*.py)
    if workspace:
        ws_dir = workspace / "classifiers"
        if ws_dir.is_dir():
            for file in sorted(ws_dir.glob("*.py")):
                if file.name == "__init__.py":
                    continue
                modname = f"_nc_user_cls_{file.stem}"
                try:
                    mod = _import_module_from_file(file, modname)
                    cnt = _add_from_module(mod, f"ws:{file.name}")
                    report.ws_loaded.append((file.name, cnt))
                except Exception as e:
                    report.ws_failed.append((file.name, f"{type(e).__name__}: {e}"))

    # 2) Packaged (numclass.classifiers.*)
    try:
        pkg_dir = pkg_files("numclass") / "classifiers"
        with as_file(pkg_dir) as real:
            for file in sorted(Path(real).glob("*.py")):
                if file.name == "__init__.py":
                    continue
                modname = f"numclass.classifiers.{file.stem}"
                try:
                    mod = import_module(modname)
                    cnt = _add_from_module(mod, f"pkg:{modname}")
                    report.pkg_loaded.append((modname, cnt))
                except Exception as e:
                    report.pkg_failed.append((modname, f"{type(e).__name__}: {e}"))
    except Exception as e:
        # failure to access packaged dir itself
        report.pkg_failed.append(("numclass.classifiers", f"{type(e).__name__}: {e}"))

    idx = Index(
        funcs=funcs,
        categories=cats,
        descriptions=desc,
        oeis=refs,
        label_to_token=toks,
        limits=limits,
    )
    idx = _attach_intersections(idx)
    return idx, report


def _attach_intersections(idx: Index) -> Index:
    """Read intersections.toml and attach as Index.intersections; also merge description/oeis/category metadata."""
    rules_raw = load_intersections()
    rules: list[IntersectionRule] = []

    for r in rules_raw:
        req_tokens = []
        for item in r["of"]:
            # allow either exact label names or pre-tokenized strings
            if item in idx.label_to_token:
                req_tokens.append(idx.label_to_token[item])
            else:
                req_tokens.append(Index.to_token(item))
        rule = IntersectionRule(
            label=r["label"],
            category=r["category"] or "Uncategorized",
            requires=tuple(req_tokens),
            description=r.get("description"),
            oeis=r.get("oeis"),
        )
        rules.append(rule)

        # let intersections show a short line & OEIS in listings (optional)
        if rule.description and rule.label not in idx.descriptions:
            idx.descriptions[rule.label] = rule.description
        if rule.oeis and rule.label not in idx.oeis:
            idx.oeis[rule.label] = rule.oeis
        if rule.label not in idx.categories:
            idx.categories[rule.label] = rule.category

    idx.intersections = rules
    return idx
