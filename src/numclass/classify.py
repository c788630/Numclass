from __future__ import annotations

import inspect
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from colorama import Fore, Style

from numclass.context import NumCtx
from numclass.fmt import abbr_int_fast, get_terminal_width, visible_len
from numclass.runtime import CFG
from numclass.runtime import current as _rt_current

_CLEARLINE = "\r\x1b[2K"


# ---------- Data models -------------------------------------------------------

@dataclass
class Outcome:
    label: str
    category: str
    ok: bool
    detail: Any  # str for atomics; list[(prefix, body)] for intersections
    oeis: str | None
    is_intersection: bool


@dataclass
class ClassifyResult:
    outcomes: list[Outcome]
    skipped: list[str]


# ---------- Helpers -----------------------------------------------------------

def _to_token(lbl: str) -> str:
    tok = re.sub(r"[^A-Za-z0-9]+", "_", lbl).upper().strip("_")
    tok = re.sub(r"__+", "_", tok)
    return tok


def _fmt_ms(ms: float) -> str:
    return f"{ms:6.2f} ms"


def _clear_progress() -> None:
    try:
        sys.stdout.write(_CLEARLINE)
        sys.stdout.flush()
    except Exception:
        pass


def _progress_line(label: str) -> None:
    """Render one-line progress to STDOUT (no-op in debug; caller decides)."""
    width = max(40, get_terminal_width())
    msg = f" ⏳ evaluating: {label}  (Ctrl-C to skip)"
    if len(msg) >= width:
        msg = msg[: max(0, width - 2)] + "…"
    try:
        sys.stdout.write(_CLEARLINE + msg)
        sys.stdout.flush()
    except Exception:
        pass


def _print_debug_result(label: str, status: str, dt_ms: float, detail: str | None = None) -> None:
    """Emit a single debug line with timing and colored status (to STDERR)."""
    _clear_progress()  # keep streams tidy

    if status == "OK":
        stat = f"{Fore.GREEN}{Style.BRIGHT}OK  {Style.RESET_ALL}"
    elif status == "NO":
        stat = f"{Style.DIM}NO  {Style.RESET_ALL}"
    elif status == "SKIP":
        stat = f"{Fore.YELLOW}{Style.BRIGHT}SKIP{Style.RESET_ALL}"
    else:  # "ERR "
        stat = f"{Fore.RED}{Style.BRIGHT}ERR {Style.RESET_ALL}"

    tm = f"{Style.DIM}[{_fmt_ms(dt_ms)}]{Style.RESET_ALL}"
    line = f"{tm} {stat}  {label}"

    if detail:
        width = max(60, get_terminal_width())
        vis_len = visible_len(line)
        max_tail = max(10, width - vis_len - 5)
        d = str(detail)
        if len(d) > max_tail:
            d = d[: max_tail - 1] + "…"
        line += f" — {Style.DIM}{d}{Style.RESET_ALL}"

    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _coerce_result(res: Any) -> tuple[bool, str | None]:
    """Normalize classifier return into (ok, detail)."""
    if isinstance(res, tuple):
        if not res:
            return False, None
        ok, *rest = res
        detail = rest[0] if rest else None
        return bool(ok), detail
    return bool(res), None


def _get_limit(index: Any, label: str, fn: Callable) -> int | None:
    """
    Get per-label limit from Index or function attribute.
    Accepts int or (int, ...); returns the int part or None.
    """
    lim = None
    try:
        lim = getattr(index, "limits", {}).get(label)
    except Exception:
        pass
    if lim is None:
        lim = getattr(fn, "limit", None)
    if isinstance(lim, (tuple, list)) and lim:
        lim = lim[0]
    return int(lim) if isinstance(lim, int) else None


def _enabled_labels(index: Any) -> list[str]:
    """
    Profile filtering:
      - CATEGORIES: { "Primes and Prime-related Numbers": true/false, ... }
      - INCLUDE_EXCLUDE: { TOKEN: true/false }
        * if any True present → allowlist
        * else blacklist False
    """
    cats = CFG("CATEGORIES", {}) or {}
    lbls = CFG("INCLUDE_EXCLUDE", {}) or {}

    cats = {str(k): bool(v) for k, v in cats.items() if isinstance(v, bool)}
    lbls = {str(k): bool(v) for k, v in lbls.items() if isinstance(v, bool)}

    tokens_true = {k.upper() for k, v in lbls.items() if v is True}
    tokens_false = {k.upper() for k, v in lbls.items() if v is False}
    use_allowlist = len(tokens_true) > 0

    tokmap = getattr(index, "label_to_token", {}) or {}
    funcs = getattr(index, "funcs", {}) or {}

    def label_enabled(lbl: str) -> bool:
        cat = (getattr(index, "categories", {}) or {}).get(lbl) or "Uncategorized"
        if cats and cats.get(cat, True) is False:
            return False
        tok = (tokmap.get(lbl) or _to_token(lbl)).upper()
        if use_allowlist:
            return tok in tokens_true
        return tok not in tokens_false

    return [lbl for lbl in funcs if label_enabled(lbl)]


def _detail_prefix(lbl: str) -> str:
    low = lbl.lower()
    if low.startswith("palindrom"):
        return "Palindromic"
    if low.endswith(" number"):
        return lbl[:-7]
    if low.endswith(" prime"):
        return lbl[:-6]
    return lbl


# ---------- Main API ----------------------------------------------------------

def classify(
    n: int,
    index: Any,
    progress: bool = True,             # <- True = built-in live line, False = silent
    ctx: NumCtx | None = None,
) -> ClassifyResult:
    """
    Run atomic classifiers (filtered by profile), then intersections:
      * per-label limit honored unless fast_mode=False
      * KeyboardInterrupt skips current classifier
      * TimeoutError → SKIP (no traceback)
      * other errors recorded to 'skipped'
      * intersections: keep maximal; suppress constituent atomics; inherit multi-details
    """

    n_abs = abs(int(n))
    last_debug_category: str | None = None
    show_details = CFG("DISPLAY_SETTINGS.SHOW_CLASSIFIER_DETAILS", True)

    rt = _rt_current()
    debug = bool(getattr(rt, "debug", False))
    fast_mode = bool(getattr(rt, "fast_mode", True))

    # Which labels to run (discovery order filtered by profile)
    labels = _enabled_labels(index)
    funcs: dict[str, Callable] = getattr(index, "funcs", {}) or {}
    categories: dict[str, str] = getattr(index, "categories", {}) or {}
    oeis_map: dict[str, str | None] = getattr(index, "oeis", {}) or {}

    # Sort classifiers by (category, label) for stable grouping
    def _cat_key(lbl: str) -> tuple[str, str]:
        cat = categories.get(lbl) or "Uncategorized"
        return (cat.casefold(), lbl.casefold())

    labels = sorted(labels, key=_cat_key)

    # token maps
    tokmap: dict[str, str] = getattr(index, "label_to_token", {}) or {lbl: _to_token(lbl) for lbl in funcs}
    rev_tok: dict[str, str] = {tok: lbl for lbl, tok in tokmap.items()}

    outcomes: list[Outcome] = []
    skipped: list[str] = []
    present_tokens: set[str] = set()
    detail_by_label: dict[str, str | None] = {}

    # --- tiny helper: emit built-in live line only when requested ---
    def _emit(msg: str) -> None:
        if progress and not debug:
            _progress_line(msg)

    # ---------- atomic loop ----------
    for label in labels:
        fn = funcs.get(label)
        if not fn:
            continue

        if debug:
            cat = categories.get(label, "Uncategorized")
            if cat != last_debug_category:
                last_debug_category = cat
                print(f"\n{Fore.YELLOW}{Style.BRIGHT}{cat}{Style.RESET_ALL}")

        lim = _get_limit(index, label, fn)
        if isinstance(lim, int):
            digits = len(str(lim))
            if all(c == "9" for c in str(lim)):
                info = f"{digits} digits"
            else:
                short = abbr_int_fast(lim, 10, 10, 80)
                info = f"{short} ({digits} digits)"
        else:
            info = None

        if fast_mode and isinstance(lim, int) and n_abs > lim:
            if debug:
                _print_debug_result(label, "SKIP", 0.0, f"|n| > {info}")
            else:
                _emit(f" skipped (limit): {label}")
            skipped.append(f"{label}: |n| > {info}")
            continue

        # Live progress per label (silent in debug)
        if not debug:
            _emit(label)

        # Prepare kwargs reflectively
        kwargs = {}
        P = inspect.signature(fn).parameters
        if "ctx" in P:
            kwargs["ctx"] = ctx

        t0 = time.perf_counter()
        ok: bool = False
        detail: str | None = None
        status_override: str | None = None

        try:
            res = fn(n, **kwargs)
            ok, detail = _coerce_result(res)

        except KeyboardInterrupt:
            if debug:
                _print_debug_result(label, "SKIP", 0.0, "KeyboardInterrupt")
            else:
                _emit(f" skipped (Ctrl-C): {label}")
            skipped.append(f"{label}: aborted by user (Ctrl-C)")
            continue

        except TimeoutError as e:
            # Treat timeouts as SKIP with a message; no traceback.
            msg = str(e) or "incomplete factorization"
            skipped.append(f"{label}: {msg}")
            ok = False
            detail = msg
            status_override = "SKIP"

        except Exception as e:
            if debug:
                # clear any live progress line before stderr (best effort)
                try:
                    sys.stdout.write("\r\x1b[2K")
                    sys.stdout.flush()
                except Exception:
                    pass
                _print_debug_result(label, "ERR", 0.0, f"{e.__class__.__name__}: {e}")
            skipped.append(f"{label}: {e}")
            ok, detail = False, None

        dt = (time.perf_counter() - t0) * 1000.0

        if debug:
            status = status_override or ("OK" if ok else "NO")
            _print_debug_result(label, status, dt, str(detail) if detail is not None else None)

        if ok:
            cat = categories.get(label, "Uncategorized")
            oe = oeis_map.get(label)
            detail_to_store = detail if show_details else None
            outcomes.append(Outcome(label, cat, True, detail_to_store, oe, False))
            detail_by_label[label] = detail  # keep raw detail for inheritance logic
            tok = tokmap.get(label)
            if tok:
                present_tokens.add(tok)

    # ---------- intersections ----------
    rules = list(getattr(index, "intersections", []) or [])
    matched = [r for r in rules if set(r.requires or []).issubset(present_tokens)]

    # dominance: keep only maximal (not strictly subsumed by a bigger set)
    reqmap = {r.label: set(r.requires or []) for r in matched}
    keep: list[Any] = []
    for r in matched:
        s = reqmap[r.label]
        dominated = any(reqmap[q.label] > s for q in matched if q is not r)
        if not dominated:
            keep.append(r)

    # build intersection outcomes with multi-detail inheritance
    for rule in keep:
        parts: list[tuple[str, str]] = []
        for tok in (rule.requires or []):
            base_lbl = rev_tok.get(tok)
            if not base_lbl or base_lbl == "Prime number":
                continue
            d = detail_by_label.get(base_lbl)
            if d:
                parts.append((_detail_prefix(base_lbl), str(d)))
        inherited = parts if (parts and show_details) else None
        cat = getattr(rule, "category", None) or "Uncategorized"
        outcomes.append(
            Outcome(rule.label, cat, True, inherited, getattr(rule, "oeis", None), True)
        )

    # suppress atomics that are constituents of any kept intersection
    suppress_atomic_labels: set[str] = set()
    for rule in keep:
        for tok in rule.requires or []:
            base_lbl = rev_tok.get(tok)
            if base_lbl:
                suppress_atomic_labels.add(base_lbl)

    if suppress_atomic_labels:
        outcomes = [
            o for o in outcomes
            if not (not o.is_intersection and o.label in suppress_atomic_labels)
        ]

    # final ordering
    outcomes.sort(key=lambda o: (o.category.lower(), o.label.lower()))

    # Clear our own live line only if we actually drew it
    if progress and not debug:
        _clear_progress()
    elif debug:
        print()

    return ClassifyResult(outcomes=outcomes, skipped=skipped)
