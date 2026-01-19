# src/numclass/fmt.py
from __future__ import annotations

import re
import textwrap
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

from colorama import Fore, Style

from numclass.runtime import CFG
from numclass.utility import dec_digits, get_terminal_width, zeckendorf_decomposition

if TYPE_CHECKING:
    # Type-only; no runtime import → avoids circulars
    pass  # optional, for editors only

# Single source of truth for ANSI stripping
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def abbr_int_fast(n: int, head: int = 10, tail: int = 10, threshold: int = 35, ellipsis: str = "…") -> str:
    """Abbreviate very large ints as first<head>…last<tail> without str(n)."""
    # Keep non-ints and small ints simple
    if not isinstance(n, int):
        return str(n)
    if n == 0:
        return "0"

    sign = "-" if n < 0 else ""
    a = -n if n < 0 else n

    # If not long enough, fall back to normal str()
    d = dec_digits(a)
    if d <= threshold or head + tail >= d:
        return sign + str(a)

    # compute first/last blocks exactly
    k = d - head
    pow10_k = 10 ** k
    first = a // pow10_k

    last_mod = 10 ** tail
    last = a % last_mod
    # zero-pad last block to width 'tail'
    last_str = f"{last:0{tail}d}"
    return f"{sign}{first}{ellipsis}{last_str}"


def format_peak_token(
    peak: int,
    *,
    mode: str | None = None,
) -> str | None:
    """
    Return a colored 'peak …' token or None if suppressed.
    mode: "full" | "shortened" | "none" (case-insensitive).
    Uses FORMATTING.NUM_ABBR_* for abbreviation when shortened.
    """
    if mode is None:
        mode = str(CFG("DISPLAY_SETTINGS.SHOW_COLLATZ_PEAK", "Full"))
    m = str(mode).strip().lower()

    if m in ("none", "off", "false", "0"):
        return None

    if m in ("short", "shortened", "abbr", "abbreviated"):
        ELL = CFG("FORMATTING.ELLIPSIS", "…")
        HEAD = int(CFG("FORMATTING.NUM_ABBR_HEAD", 10))
        TAIL = int(CFG("FORMATTING.NUM_ABBR_TAIL", 10))
        THR = int(CFG("FORMATTING.NUM_ABBR_THRESHOLD", 10))
        tok = abbr_int_fast(int(peak), HEAD, TAIL, THR, ELL)
    else:
        # Full (default)
        tok = str(int(peak))

    return f"{Fore.YELLOW}{Style.BRIGHT}peak {tok}{Style.RESET_ALL}"


def _fmt_odd_even_ratio(odd: int, even: int) -> str:
    """
    Return a concise string with odd/even ratio and odd percentage.
    Examples: 'odd/even 2:5 ≈ 0.400, odd% 28.6%'
    Handles division-by-zero gracefully.
    """
    total = odd + even
    bits = []
    if even > 0:
        r = Fraction(odd, even)
        bits.append(f"odd/even {r.numerator}:{r.denominator} ≈ {odd / even:.3f}")
    # No even steps: show '∞' if odd>0, else 0:0
    elif odd > 0:
        bits.append("odd/even ∞ (even=0)")
    else:
        bits.append("odd/even 0:0")
    if total > 0:
        bits.append(f"odd% {100.0 * odd / total:.1f}%")
    return ", ".join(bits)


def _collect_window_anb(n: int, a: int, b: int, *, total_len: int, print_limit: int, truncate: str = "middle"):
    """
    Second pass: emit only the values that will be printed.
    Returns (idxs, values) where both are aligned lists of absolute indices and ints.
    """
    if print_limit <= 0 or total_len <= print_limit:
        # collect all
        need = set(range(total_len))
    elif truncate == "end":
        need = set(range(print_limit))
    elif truncate == "start":
        need = set(range(total_len - print_limit, total_len))
    else:
        head = max(1, print_limit // 2)
        tail = max(1, print_limit - head)
        need = set(range(head)) | set(range(total_len - tail, total_len))

    idxs: list[int] = []
    vals: list[int] = []
    x = n
    if 0 in need:
        idxs.append(0)
        vals.append(x)
    for i in range(1, total_len):
        x = (x >> 1) if (x & 1) == 0 else (a * x + b)
        if i in need:
            idxs.append(i)
            vals.append(x)

    return idxs, vals


def render_anb_preview(
    n: int, a: int, b: int,
    *,
    total_len: int,             # steps + 1
    print_limit: int,
    truncate: TruncateMode = "middle",
    peak_idx: int | None,
    mu: int | None,             # loop entry index
    end: int | None,            # closing index (first repeat position)
) -> tuple[str, list[str]]:
    """
    Build the short, colored preview line for an (an+b) Collatz-like sequence
    WITHOUT materializing the full sequence.

    Returns: (seq_str, status_lines)
      - seq_str like ", sequence: 7 → 22 → 11 → … → 1"
      - status_lines e.g. ["cycle from term #6 (length 4)", "Truncated: showing 50 of 1234 terms"]
    """
    idxs, vals = _collect_window_anb(
        n, a, b, total_len=total_len, print_limit=print_limit, truncate=truncate
    )
    seq_str, status = _render_window_with_markers(
        idxs, vals,
        total_len=total_len,
        print_limit=print_limit,
        truncate=truncate,
        mu=mu, end=end,
        peak_idx=peak_idx,
    )
    return seq_str, status


def _render_window_with_markers(
    idxs: list[int], vals: list[int], *,
    total_len: int,
    print_limit: int,
    truncate: str,
    mu: int | None,
    end: int | None,
    peak_idx: int | None,
) -> tuple[str, list[str]]:
    SEQ_ARROW = CFG("FORMATTING.SEQUENCE_ARROW", " \u2192 ")
    ELLIPSIS_NUM = CFG("FORMATTING.ELLIPSIS", "…")          # Small ellipsis for simple number truncation
    ELLIPSIS_SPLIT = CFG("FORMATTING.ELLIPSIS_SPLIT", "•••")  # Big ellipsis for splitting a sequence

    ABBR = CFG("FORMATTING.NUM_ABBR_ENABLED", True)
    H = CFG("FORMATTING.NUM_ABBR_HEAD", 10)
    T = CFG("FORMATTING.NUM_ABBR_TAIL", 10)
    TH = CFG("FORMATTING.NUM_ABBR_THRESHOLD", 10)
    status: list[str] = []

    loop_idx = set()
    if mu is not None and end is not None:
        loop_idx.add(mu)
        loop_idx.add(end)

    def abbr(x: int) -> str:
        if not ABBR:
            return str(x)
        return abbr_int_fast(x, H, T, TH, ELLIPSIS_NUM)

    def colorize(i: int, tok: str) -> str:
        if peak_idx is not None and i == peak_idx:
            return f"{Fore.YELLOW}{Style.BRIGHT}{tok}{Style.RESET_ALL}"
        if i in loop_idx:
            return f"{Fore.MAGENTA}{Style.BRIGHT}{tok}{Style.RESET_ALL}"
        return tok

    tokens = [colorize(i, abbr(v)) for i, v in zip(idxs, vals, strict=False)]

    if print_limit <= 0 or total_len <= len(idxs):
        return f", sequence: {SEQ_ARROW.join(tokens)}", status

    if truncate == "end":
        s = SEQ_ARROW.join(tokens) + f" {ELLIPSIS_NUM}"
    elif truncate == "start":
        s = f"{ELLIPSIS_NUM} " + SEQ_ARROW.join(tokens)
    else:
        # ---- Add big split ellipsis between head/tail ----
        cut = len(tokens) // 2
        head = SEQ_ARROW.join(tokens[:cut])
        tail = SEQ_ARROW.join(tokens[cut:])
        # colored split: ...white bold '•••'..., then a bright green arrow into the tail
        split = (
            f"{SEQ_ARROW}"
            f"{Fore.WHITE}{Style.BRIGHT}{ELLIPSIS_SPLIT}{Style.RESET_ALL}"
            f"{SEQ_ARROW}"
        )
        s = f"{head}{split}{tail}"

    if mu is not None and end is not None:
        status.append(f"{Fore.MAGENTA}{Style.BRIGHT}cycle{Style.RESET_ALL} from term #{mu+1} (length {end-mu})")
    status.append(f"{Fore.YELLOW}Truncated: showing {len(tokens)} of {total_len} terms{Style.RESET_ALL}")
    return f", sequence: {s}", status


def strip_ansi(s: str | None) -> str:
    """Return s with ANSI escape sequences removed."""
    return "" if s is None else ANSI_RE.sub("", s)


def visible_len(s: str | None) -> int:
    """Printable length (without ANSI)."""
    return len(strip_ansi(s))


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
TOK_RE = re.compile(r'(\x1b\[[0-9;]*m|.)', re.DOTALL)


def _chunk_by_visible_width(s: str, width: int) -> list[str]:
    if width <= 0:
        return [s]
    out, buf, vis = [], [], 0
    for tok in TOK_RE.findall(s):
        if ANSI_RE.fullmatch(tok):
            buf.append(tok)
            continue
        if vis == width:
            out.append("".join(buf))
            buf = []
            vis = 0
        buf.append(tok)
        vis += 1
    if buf:
        out.append("".join(buf))
    return out


def wrap_after_label(
    label: str,
    text: str,
    *,
    width: int | None = None,
    wrap_long_tokens: bool = False,
) -> str:
    """
    ANSI-safe wrap where the body (text) may contain color codes.
    First line capacity = width - visible_len(label)
    Subsequent lines    = same indent under the label.
    Preserves spaces. When wrap_long_tokens=False, long non-space tokens
    are moved wholly to the next line instead of being chopped.
    """
    W = int(width or get_terminal_width())
    W = max(20, W)

    start = visible_len(label)
    cap = max(5, W - start)

    s = text or ""
    if not s:
        return label

    tokens = re.split(r'(\s+)', s)  # keep whitespace

    lines: list[str] = []
    cur = ""
    cur_len = 0

    def flush_line():
        nonlocal cur, cur_len
        lines.append(cur)
        cur, cur_len = "", 0

    for tok in tokens:
        vis = visible_len(tok)
        is_space = (tok.strip() == "")

        # avoid leading whitespace on a new wrapped line
        if cur_len == 0 and is_space:
            continue

        # fits as-is?
        if cur_len + vis <= cap:
            cur += tok
            cur_len += vis
            continue

        # token doesn't fit
        if is_space:
            # just break line; skip leading spaces on next line
            flush_line()
            continue

        if not wrap_long_tokens:
            # move whole token to next line (no chopping)
            flush_line()
            cur = tok
            cur_len = vis
            continue

        # wrap_long_tokens=True: chop long token to fill remainder then full-width lines
        rem = cap - cur_len  # remainder on current line
        if rem > 0:
            parts = _chunk_by_visible_width(tok, rem)
            first = parts[0]
            cur += first
            cur_len += visible_len(first)
            flush_line()
            tail = "".join(parts[1:])
        else:
            tail = tok

        if tail:
            full = _chunk_by_visible_width(tail, cap)
            for mid in full[:-1]:
                lines.append(mid)
            cur = full[-1]
            cur_len = visible_len(cur)
            if cur_len == cap:
                flush_line()

    if cur or not lines:
        lines.append(cur)

    out = label + (lines[0] if lines else "")
    indent = " " * start
    for ln in lines[1:]:
        out += "\n" + indent + ln
    return out


def _intersection_label_set(inters: Iterable[Any] | None) -> set[str]:
    """Return set of labels that are intersections (accepts rules or plain strings)."""
    s: set[str] = set()
    if not inters:
        return s
    for r in inters:
        if isinstance(r, str):
            s.add(r)
        else:
            lab = getattr(r, "label", None)
            if isinstance(lab, str):
                s.add(lab)
    return s


def format_factorization(fac: Mapping[int, int]) -> str:
    """
    Turn {p: e, ...} into a tidy string like: 2^3 × 3 × 5^2
    """
    parts: list[str] = []
    for p, e in sorted(fac.items()):
        parts.append(f"{p}^{e}" if e > 1 else f"{p}")
    return " × ".join(parts) if parts else "1"


def format_sigma_terms(parts: Iterable[tuple[str, int]]) -> tuple[str, str]:
    """
    Accept a list of (symbolic_term, numeric_value) for each prime power factor
    of σ(n), and return two strings:
      - syms: "(1+2+4) × (1+3)" style symbolic expansion
      - vals: "7 × 4" numeric product
    """
    syms: list[str] = []
    vals: list[str] = []
    for sym, val in parts:
        syms.append(sym)
        vals.append(str(val))
    if not syms:
        return "1", "1"
    return " × ".join(syms), " × ".join(vals)


TruncateMode = Literal["start", "middle", "end"]


def format_duration(seconds: float) -> str:
    """ms if <1s; s with millis if <60s; else mm:ss.mmm (and hh:mm:ss.mmm if ≥1h)."""
    MAX_SECONDS = 60
    if seconds < 1:
        ms = round(seconds * 1000)
        return f"{ms} ms"
    if seconds < MAX_SECONDS:
        return f"{seconds:.3f} s"
    m, s = divmod(seconds, MAX_SECONDS)
    if m < MAX_SECONDS:
        return f"{int(m)}:{s:06.3f}"               # mm:ss.mmm
    h, m = divmod(int(m), MAX_SECONDS)
    return f"{h}:{m:02d}:{s:06.3f}"                # hh:mm:ss.mmm


def format_sequence(
    seq,
    highlight_idx,
    aborted: bool,
    skipped,
    *,
    print_limit: int | None = None,
    max_steps: int | None = None,
    arrow: str = " → ",
    peak_idx: int | None = None,
) -> tuple[str, list[str], int]:
    """
    Returns (seq_str, status_lines, steps).

    Highlighting:
      - loop entry markers ONLY (first and closing repeat) in bright magenta
      - peak term (if provided) in bright yellow
      - if the peak coincides with a loop marker, PEAK (yellow) takes precedence
    """
    # Normalize to ints/strings
    raw: list[int] = []
    parts: list[str] = []
    for x in (seq or []):
        try:
            xi = int(x)
            raw.append(xi)
            parts.append(str(xi))
        except Exception:
            raw.append(x)
            parts.append(str(x))

    steps = max(0, len(parts) - 1)
    status: list[str] = []

    if not parts:
        status.append("empty sequence")
        return "—", status, steps

    # Resolve (start, end) for loop markers
    TWO_HIGHLITES = 2
    start = None
    end = None
    if isinstance(highlight_idx, tuple) and len(highlight_idx) == TWO_HIGHLITES:
        a, b = highlight_idx
        if isinstance(a, int) and isinstance(b, int) and 0 <= a < len(parts) and 0 <= b < len(parts):
            start, end = a, b
    elif isinstance(highlight_idx, int) and 0 <= highlight_idx < len(parts):
        start, end = highlight_idx, len(parts) - 1

    # Build index sets
    loop_idx: set[int] = set()
    if start is not None and end is not None and end >= start:
        # ONLY first and closing repeat
        loop_idx.add(start)
        try:
            if raw[end] == raw[start]:
                loop_idx.add(end)
        except Exception:
            loop_idx.add(end)

    peak_i: int | None = None
    if isinstance(peak_idx, int) and 0 <= peak_idx < len(parts):
        peak_i = peak_idx

    # Build colored tokens (peak overrides loop color)
    tokens: list[str] = []
    for i, tok in enumerate(parts):
        if peak_i is not None and i == peak_i:
            tokens.append(f"{Fore.YELLOW}{Style.BRIGHT}{tok}{Style.RESET_ALL}")
        elif i in loop_idx:
            tokens.append(f"{Fore.MAGENTA}{Style.BRIGHT}{tok}{Style.RESET_ALL}")
        else:
            tokens.append(tok)

    def _join(toks: list[str]) -> str:
        return arrow.join(toks)

    # Truncation (after coloring)
    if print_limit is not None and len(tokens) > max(1, int(print_limit)):
        limit = max(1, int(print_limit))
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        if head + tail > limit:
            tail = max(1, limit - head)

        head_tokens = tokens[:head]
        tail_tokens = tokens[-tail:] if tail > 0 else []
        seq_str = _join(head_tokens) + f"{arrow}…{arrow}" + _join(tail_tokens)

        if start is not None and end is not None and end >= start:
            loop_len = max(1, end - start)
            status.append(
                f"{Fore.MAGENTA}{Style.BRIGHT}cycle{Style.RESET_ALL} from term #{start + 1} (length {loop_len})"
            )
        status.append(f"{Style.RESET_ALL}{Fore.YELLOW}Truncated: showing {head + tail} of {len(parts)} terms{Style.RESET_ALL}")
    else:
        seq_str = _join(tokens)
        if start is not None and end is not None and end >= start:
            loop_len = max(1, end - start)
            status.append(
                f"{Fore.MAGENTA}{Style.BRIGHT}cycle{Style.RESET_ALL} from term #{start + 1} (length {loop_len})"
            )

    return seq_str, status, steps


def format_seq_line(
    seq: Iterable[Any],
    print_limit: int | None = None,
    truncate: str = "middle",   # "start" | "middle" | "end"
) -> str:
    """
    Minimal, fast sequence preview (no cycle/peak), with:
      - per-number abbreviation via abbr_int_fast
      - head/tail truncation
      - colored split ellipsis for "middle"
    Returns: ", sequence: a → b → … → z"
    """
    SEQ_ARROW = CFG("FORMATTING.SEQUENCE_ARROW", " \u2192 ")
    ELLIPSIS_NUM = CFG("FORMATTING.ELLIPSIS", "…")
    ELLIPSIS_SPLIT = CFG("FORMATTING.ELLIPSIS_SPLIT", "•••")  # fallback if not present

    ABBR_ENABLED = CFG("FORMATTING.NUM_ABBR_ENABLED", True)
    HEAD = int(CFG("FORMATTING.NUM_ABBR_HEAD", 10))
    TAIL = int(CFG("FORMATTING.NUM_ABBR_TAIL", 10))
    THR = int(CFG("FORMATTING.NUM_ABBR_THRESHOLD", 10))

    parts = list(seq or [])
    total = len(parts)

    if total == 0:
        return ", sequence: —"

    # choose window
    if isinstance(print_limit, int) and print_limit > 0 and total > print_limit:
        if truncate == "end":
            window = parts[:print_limit]
            mode = "end"
        elif truncate == "start":
            window = parts[-print_limit:]
            mode = "start"
        else:
            head = max(1, print_limit // 2)
            tail = max(1, print_limit - head)
            window = parts[:head] + parts[-tail:]
            mode = "middle"
    else:
        window = parts
        mode = "full"

    def to_str_fast(x: Any) -> str:
        if not ABBR_ENABLED:
            try:
                return str(int(x))
            except Exception:
                return str(x)
        if isinstance(x, int):
            return abbr_int_fast(x, HEAD, TAIL, THR, ELLIPSIS_NUM)
        try:
            xi = int(x)
            return abbr_int_fast(xi, HEAD, TAIL, THR, ELLIPSIS_NUM)
        except Exception:
            return str(x)

    toks = [to_str_fast(x) for x in window]

    if mode == "full":
        rendered = SEQ_ARROW.join(toks)
    elif mode == "end":
        rendered = SEQ_ARROW.join(toks) + f" {ELLIPSIS_NUM}"
    elif mode == "start":
        rendered = f"{ELLIPSIS_NUM} " + SEQ_ARROW.join(toks)
    else:
        # colored split: head …→ tail
        cut = len(toks) // 2
        head = SEQ_ARROW.join(toks[:cut])
        tail = SEQ_ARROW.join(toks[cut:])
        split = (
            f"{SEQ_ARROW}"
            f"{Fore.WHITE}{Style.BRIGHT}{ELLIPSIS_SPLIT}{Style.RESET_ALL}"
            f"{Fore.GREEN}{Style.BRIGHT}{SEQ_ARROW}{Style.RESET_ALL}"
        )
        rendered = f"{head}{split}{tail}"

    return rendered


def _abbr_bits(bits: str, max_len: int = 120, ell: str = "…") -> str:
    if max_len <= 0 or len(bits) <= max_len:
        return bits
    h = max_len // 2
    t = max_len - h
    return bits[:h] + ell + bits[-t:]


def format_zeckendorf(n: int) -> str | None:
    if n < 0:
        return None

    terms, idxs, bits = zeckendorf_decomposition(n)
    if not terms:
        return "0"

    # Settings
    ELL = str(CFG("FORMATTING.ELLIPSIS", "…"))

    max_terms = int(CFG("DISPLAY_SETTINGS.ZECKENDORF_MAX_TERMS", 20))
    show_bits = bool(CFG("DISPLAY_SETTINGS.SHOW_ZECKENDORF_BITS", True))
    trunc_preview = bool(CFG("DISPLAY_SETTINGS.ZECKENDORF_TRUNC_PREVIEW", True))

    # Abbreviation knobs
    ABBR = bool(CFG("FORMATTING.NUM_ABBR_ENABLED", True))
    H = int(CFG("FORMATTING.NUM_ABBR_HEAD", 10))
    T = int(CFG("FORMATTING.NUM_ABBR_TAIL", 10))
    TH = int(CFG("FORMATTING.NUM_ABBR_THRESHOLD", 10))

    def fmt_int(x: int) -> str:
        """Format integers with optional abbreviation, controlled by ABBR."""
        x = int(x)
        if not ABBR:
            return str(x)
        return abbr_int_fast(x, H, T, TH, ELL)

    def fmt_bits(s: str, max_len: int = 120) -> str:
        """Abbreviate bitstrings (independent from ABBR; can be made cfg-driven later)."""
        if max_len <= 0 or len(s) <= max_len:
            return s
        h = max_len // 2
        t = max_len - h
        return s[:h] + ELL + s[-t:]

    # Prefer printing Fibonacci indices in descending order (largest first).
    idxs_desc = sorted(idxs, reverse=True)
    k = idxs_desc[0] if idxs_desc else 0


    # Build body
    if len(terms) <= max_terms:
        # Small/medium: show full numeric decomposition + Fibonacci indices
        parts = " + ".join(fmt_int(t) for t in terms)
        fparts = " + ".join(f"F{j}" for j in idxs_desc)
        body = f"{parts} ({fparts})"
        if show_bits and k >= 2:
            body += f", bits(F{k}..F2)={fmt_bits(bits)}"
    else:
        # Huge: show a compact Fibonacci-index preview (no duplication)
        if trunc_preview:
            head = " + ".join(f"F{j}" for j in idxs_desc[:3])
            tail = " + ".join(f"F{j}" for j in idxs_desc[-3:])
            body = f"{head} + {ELL} + {tail}"
        else:
            # Minimal preview: first two terms (or one if only one exists)
            if len(idxs_desc) >= 2:
                body = f"F{idxs_desc[0]} + F{idxs_desc[1]}"
            else:
                body = f"F{idxs_desc[0]}"

        if show_bits and k >= 2:
            body += f", bits(F{k}..F2)={fmt_bits(bits)}"

    return f"{fmt_int(n)} = {body}"


def wrap_description_bullet(prefix: str, desc: str, *, width: int | None = None, indent_cols: int = 5) -> str:
    """
    Wrap description so that continuation lines start at a fixed indent (indent_cols),
    not under the label. ANSI-safe (prefix may contain color codes).

    First line capacity  = width - visible_len(prefix)
    Subsequent capacity  = width - indent_cols
    """

    W = width or get_terminal_width()
    W = max(20, int(W))  # sanity floor

    first_cap = max(5, W - visible_len(prefix))
    cont_cap = max(5, W - int(indent_cols))

    words = (desc or "").split()
    if not words:
        return prefix  # nothing to append, just show the bullet/label

    # Build first line
    first_parts = []
    cur = 0
    for w in words:
        need = len(w) if cur == 0 else (1 + len(w))  # space before non-first
        if cur + need <= first_cap:
            first_parts.append(w)
            cur += need
        else:
            break

    used = len(first_parts)
    out = prefix + (" ".join(first_parts) if first_parts else "")

    # Remaining lines at fixed indent
    if used < len(words):
        body = " ".join(words[used:])
        wrap = textwrap.wrap(body, width=cont_cap) if body else []
        indent = " " * int(indent_cols)
        for ln in wrap:
            out += "\n" + indent + ln

    return out
