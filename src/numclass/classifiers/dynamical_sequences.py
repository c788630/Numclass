# -----------------------------------------------------------------------------
#  Dynamical sequences test functions
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isinf

from colorama import Fore, Style

from numclass.fmt import _fmt_odd_even_ratio, abbr_int_fast, format_peak_token, format_seq_line, format_sequence, render_anb_preview
from numclass.registry import classifier
from numclass.runtime import CFG
from numclass.runtime import current as _rt_current

CATEGORY = "Dynamical Sequences"


# ----------------------------- helper functions -----------------------------


def _as_opt_number(v, *, cast):
    """
    Map 'infinite' sentinels to None; else cast to the requested type.
    Accepts:
      - None / ""  -> None
      - float('inf'), float('-inf') -> None
      - "inf", "+inf", "-inf", "∞" (any case, with spaces) -> None
    Otherwise returns cast(v).
    """
    if v is None or v == "":
        return None
    # numeric infinities
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and isinf(v)) else cast(v)
    # string tokens
    if isinstance(v, str):
        tok = v.strip().lower()
        if tok in {"inf", "+inf", "-inf", "∞"}:
            return None
    return cast(v)


def _detail_for_anb(n: int, a: int, b: int) -> tuple[bool, str]:
    S = _compute_anb_stats(n, a, b)

    PRINT_LIMIT = int(CFG("DISPLAY_SETTINGS.SEQUENCE_PRINT_LIMIT", 300))
    total_len = S.steps + 1

    seq_str, status = render_anb_preview(
        n, a, b,
        total_len=total_len,
        print_limit=PRINT_LIMIT,
        truncate="middle",
        peak_idx=S.peak_idx,
        mu=S.mu, end=S.end,
    )

    peak_tok = format_peak_token(S.peak)
    bits = [
        f"rule: a={a}, b={b} (odd → {a}n{b:+d}; even → n/2)",
        f"total steps: {S.steps}",
        f"n/2 steps: {S.evensteps}",
        f"{a}n{b:+d} steps: {S.oddsteps}",
    ]
    # ratio + percentage
    bits.append(_fmt_odd_even_ratio(S.oddsteps, S.evensteps))

    if peak_tok:
        bits.append(peak_tok)

    detail = ", ".join(bits) + f"{seq_str}"

    if S.aborted and S.reason:
        status.append(f"{Style.BRIGHT}{Fore.YELLOW}Aborted: {S.reason}{Style.RESET_ALL}")
    if status:
        detail += "  [" + "; ".join(status) + "]"
    return True, detail


@dataclass(frozen=True)
class AnbStats:
    steps: int           # total transitions performed
    evensteps: int
    oddsteps: int
    peak: int
    peak_idx: int | None
    mu: int | None       # loop entry index (0-based in [x0, x1, ...])
    end: int | None      # closing index (first repeat position)
    aborted: bool
    reason: str | None


def _compute_anb_stats(n: int, a: int, b: int) -> AnbStats:
    """
    Single-pass Collatz-like (an+b) compute:
      - counts even/odd steps
      - tracks peak (+ index)
      - detects first repeat (mu,end)
      - enforces caps:
          * 3n+1 + FAST_MODE=False -> value caps disabled
          * others -> bit-length cap + adaptive divergence guard
      - optional time budget
    No sequence list is built.
    """

    # --- config --------------------------------------------------------------
    MAX_STEPS = CFG("COLLATZ_LIKE.MAX_STEPS", 200_000)

    # Bit-length cap (preferred for huge integers)
    VALUE_CAP_BITS = _as_opt_number(CFG("COLLATZ_LIKE.VALUE_CAP_BITS", 20000), cast=int)

    # Allow even a/b?
    REQUIRE_ODD_AB = not bool(CFG("COLLATZ_LIKE.ALLOW_EVEN_A_OR_B", False))
    AB_LIMIT = int(CFG("COLLATZ_LIKE.A_B_LIMIT", 9))

    # 3n+1 whitelist when user explicitly allows slow work
    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    IS_3N1 = (a == 3 and b == 1)

    # Optional runtime budget (cheap check every N steps)
    TIME_BUDGET_S = _as_opt_number(CFG("COLLATZ_LIKE.TIME_BUDGET_S", None), cast=float)
    TIME_CHECK_EVERY = 8192  # reduce overhead

    # Divergence guard (macro-steps = odd→(an+b)→…→next odd)
    DRIFT_WINDOW = CFG("COLLATZ_LIKE.DRIFT_WINDOW", 256)
    DRIFT_MIN_BITS = CFG("COLLATZ_LIKE.DRIFT_MIN_BITS", 4096)
    # Average growth in bits per macro-step to consider diverging
    DRIFT_BITS_EPS = CFG("COLLATZ_LIKE.DRIFT_BITS_EPS", 0.25)

    # 3n+1 + FAST_MODE: enable all value/bit caps and divergence guard
    if IS_3N1 and (not fast_mode):
        VALUE_CAP_BITS = None
        # keep divergence guard off to allow giant starts
        drift_enabled = False
    else:
        drift_enabled = True

    # Basic param validation
    if REQUIRE_ODD_AB and ((a & 1) == 0 or (b & 1) == 0):
        return AnbStats(0, 0, 0, n, 0, None, None, True, "require odd a and b")
    if abs(a) > AB_LIMIT or abs(b) > AB_LIMIT:
        return AnbStats(0, 0, 0, n, 0, None, None, True, f"|a| or |b| too large (>{AB_LIMIT})")

    def step(x: int) -> int:
        return (x >> 1) if (x & 1) == 0 else (a * x + b)

    # --- main loop -----------------------------------------------------------
    seen: dict[int, int] = {n: 0}  # value -> index (x0,x1,...)
    x = n
    steps = 0
    evensteps = 0
    oddsteps = 0
    peak = n
    peak_idx = 0
    mu = end = None
    aborted = False
    reason: str | None = None

    # Drift (macro) tracking
    last_odd_idx = 0
    last_odd_val = n
    drift_sum_bits = 0
    drift_count = 0
    drift_ring: list[int] = [0] * DRIFT_WINDOW
    ring_i = 0

    t0 = time.monotonic() if TIME_BUDGET_S else 0.0

    while steps < MAX_STEPS:
        # parity classification *before* stepping
        if (x & 1) == 0:
            evensteps += 1
        else:
            oddsteps += 1

        x_next = step(x)
        steps += 1

        # peak
        if x_next > peak:
            peak = x_next
            peak_idx = steps

        if VALUE_CAP_BITS is not None and x_next.bit_length() > VALUE_CAP_BITS:
            aborted, reason = True, f"bit_length > {VALUE_CAP_BITS}"
            break

        # termination
        if x_next == 1:
            x = x_next
            break

        # cycle check (first repeat)
        prev = seen.get(x_next)
        if prev is not None:
            mu, end = prev, steps
            x = x_next
            break
        seen[x_next] = steps

        # Macro-step boundary: finished the *even* run after an odd?
        # Detect when the *new current* state is odd (i.e., next iteration will count it as odd)
        if drift_enabled and (x_next & 1) == 1:
            # macro from last_odd_val -> x_next
            if last_odd_idx < steps:  # exclude the x0==odd initial degenerate if immediately odd again
                bits_before = last_odd_val.bit_length()
                bits_after = x_next.bit_length()
                delta_bits = bits_after - bits_before

                # sliding window avg
                if DRIFT_WINDOW > 0:
                    drift_sum_bits -= drift_ring[ring_i]
                    drift_sum_bits += delta_bits
                    drift_ring[ring_i] = delta_bits
                    ring_i = (ring_i + 1) % DRIFT_WINDOW
                    if drift_count < DRIFT_WINDOW:
                        drift_count += 1

                    # only consider aborting after we are "large enough"
                    if (
                        bits_after >= DRIFT_MIN_BITS
                        and drift_count >= DRIFT_WINDOW
                        and (drift_sum_bits / drift_count) > DRIFT_BITS_EPS
                    ):
                        aborted = True
                        reason = (f"diverging: avg +{drift_sum_bits/drift_count:.2f} bits "
                                  f"per odd-step over last {drift_count}")
                        x = x_next
                        break

            last_odd_idx = steps
            last_odd_val = x_next

        # time budget (cheap check)
        if TIME_BUDGET_S and (steps & (TIME_CHECK_EVERY - 1)) == 0 and time.monotonic() - t0 > TIME_BUDGET_S:
            aborted, reason = True, f"time > {TIME_BUDGET_S:.0f}s"
            x = x_next
            break

        x = x_next

    if not aborted and mu is None and end is None and steps >= MAX_STEPS:
        aborted, reason = True, f"steps ≥ {MAX_STEPS}"

    return AnbStats(steps, evensteps, oddsteps, peak, peak_idx, mu, end, aborted, reason)


def _reverse_number(n: int) -> int:
    s = str(abs(n))[::-1]
    return int(s)


def _is_palindrome_int(n: int) -> bool:
    s = str(n)
    return s == s[::-1]


def _step_anb(a: int, b: int) -> Callable[[int], int]:
    return lambda x: (x // 2) if (x % 2 == 0) else (a * x + b)


def _ducci_start_from_n(n: int) -> tuple[int, int, int, int]:
    """Start from the last 4 digits of n (left-padded)."""
    ds = f"{abs(n) % 10000:04d}"
    return (int(ds[0]), int(ds[1]), int(ds[2]), int(ds[3]))


def _ducci_next(t: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    a, b, c, d = t
    return (abs(a - b), abs(b - c), abs(c - d), abs(d - a))


def _fmt_ducci_seq(seq, max_items) -> str:
    SEQ_ARROW = CFG("FORMATTING.SEQUENCE_ARROW", " → ")
    items = [f"[{t[0]},{t[1]},{t[2]},{t[3]}]" for t in seq]
    if max_items is None or len(items) <= max_items:
        return SEQ_ARROW.join(items)
    return SEQ_ARROW.join(items[:max_items]) + SEQ_ARROW + "…"


# -------------------------------- classifiers -------------------------------


@classifier(
    label="Generalized Collatz (5n+1)",
    description="Even → n/2; odd → 5n+1; iterate sequence. Rule has a known positive cycle.",
    oeis="A232711",
    category=CATEGORY,
)
def is_seq_5n1(n: int):
    return _detail_for_anb(n, 5, 1)


@classifier(
    label="Generalized Collatz (7n+1)",
    description="Even → n/2; odd → 7n+1; iterate sequence.",
    category=CATEGORY,
)
def is_seq_7n1(n: int):
    return _detail_for_anb(n, 7, 1)


@classifier(
    label="Generalized Collatz (an+b)",
    description="Even → n/2; odd → an+b (a,b defined in active profile); iterate sequence.",
    category=CATEGORY,
)
def is_seq_anb(n: int):
    a = int(CFG("COLLATZ_LIKE.A_VALUE", 3))
    b = int(CFG("COLLATZ_LIKE.B_VALUE", 1))
    return _detail_for_anb(n, a, b)


@classifier(
    label="Collatz (3n+1) sequence",
    description=("Apply n→n/2 if even, else n→3n+1; repeat until 1. "
                 "Conjecture: Every number will reach 1."),
    oeis="A006370",
    category=CATEGORY,
)
def is_seq_collatz(n: int):
    return _detail_for_anb(n, 3, 1)


@classifier(
    label="Happy number sequence",
    description="Repeatedly sum of squares of digits.",
    oeis="A007770",
    category=CATEGORY,
)
def is_seq_happy(n: int) -> tuple[bool, str | None]:
    if n < 1:
        return False, None

    # Build the orbit until we hit 1 (happy) or detect a repeat (unhappy cycle).
    seq: list[int] = [abbr_int_fast(n)]
    seen_pos: dict[int, int] = {n: 0}  # value -> first index in seq
    cycle_idx: tuple[int, int] | None = None

    while True:
        # Happy fixed point reached
        if n == 1:
            break

        # Next iterate: sum of squares of digits
        n = sum((ord(d) - 48) ** 2 for d in str(n))  # slightly faster than int(d)**2
        seq.append(n)

        # Detect first repeat → cycle found; highlight from first occurrence to the repeated value
        if n in seen_pos:
            start = seen_pos[n]
            end = len(seq) - 1
            cycle_idx = (start, end)
            break
        else:
            seen_pos[n] = len(seq) - 1

    # Choose highlight: unhappy → cycle indices; happy → no cycle highlight (-1)
    highlight_idx = cycle_idx if cycle_idx is not None else -1

    # Render using aliquot formatter
    line, status_lines, _ = format_sequence(
        seq, highlight_idx, False, False
    )

    if cycle_idx is None:
        # Happy: reaches 1
        return True, f"happy sequence reaches 1: happy,{Style.RESET_ALL} sequence: {line}"
    else:
        # Unhappy: cycles (classic 4→16→…→4 for base-10)
        return True, f"happy sequence cycles: unhappy,{Style.RESET_ALL} sequence: {line}"


@classifier(
    label="Reverse-and-add sequence",
    description="Reverse digits and add each step; repeat until palindrome or cap. Related to Lychrel.",
    oeis="A033865",
    category=CATEGORY,
)
def is_seq_reverse_and_add(n: int):
    # conventional definition uses nonnegative integers
    if n < 0:
        return False, None
    RANDA_MAX_STEPS = CFG("CLASSIFIER.RANDA.MAX_STEPS", 1000)
    RANDA_PRINT_FULL_IF = CFG("CLASSIFIER.RANDA.PRINT_ALWAYS_IF", 30)
    SEQ_PRINT_LIMIT = CFG("DISPLAY_SETTINGS.SEQUENCE_PRINT_LIMIT", 300)
    x = abs(n)
    steps = 0
    seq = [x]
    cap = int(RANDA_MAX_STEPS)

    # Decide how many terms to store for printing:
    # If SEQ_PRINT_LIMIT is None => no limit; otherwise store up to max(SEQ_PRINT_LIMIT, RANDA_PRINT_FULL_IF)
    store_limit = None if SEQ_PRINT_LIMIT is None else max(SEQ_PRINT_LIMIT, RANDA_PRINT_FULL_IF)

    while steps < cap and not _is_palindrome_int(x):
        r = _reverse_number(x)
        x = x + r
        steps += 1
        if store_limit is None or steps <= store_limit:
            seq.append(x)
        if _is_palindrome_int(x):
            break

    if _is_palindrome_int(x):
        # Include sequence if we have no global cap or if steps is small enough
        include_sequence = (SEQ_PRINT_LIMIT is None) or (steps <= RANDA_PRINT_FULL_IF)
        if include_sequence:
            return True, f"steps {steps}, palindrome {x}, sequence: {format_seq_line(seq, SEQ_PRINT_LIMIT)}"
        else:
            return True, f"steps {steps}, palindrome {x}"
    else:
        # did not reach palindrome within cap; still show partial info
        tail_print_cap = None if SEQ_PRINT_LIMIT is None else RANDA_PRINT_FULL_IF
        return True, f"no palindrome within {cap} steps (last terms: {format_seq_line(seq, tail_print_cap)})"


@classifier(
    label="Kaprekar routine",
    description="Sort digits desc−asc and subtract; repeat. 3-digit→495, 4-digit→6174 (Kaprekar constants).",
    oeis="A099009",
    category=CATEGORY,
)
def is_seq_kaprekar(n: int):
    s = str(abs(n))
    if len(s) not in (3, 4):
        return False, None  # only engage for 3- or 4-digit numbers

    SEQ_PRINT_LIMIT = CFG("DISPLAY_SETTINGS.SEQUENCE_PRINT_LIMIT", 300)
    k = len(s)
    target = 6174 if k == 4 else 495

    # If all digits are identical, it collapses to 0 and stays there
    if len(set(s)) == 1:
        ds = f"{int(s):0{k}d}"
        hi = int(''.join(sorted(ds, reverse=True)))
        lo = int(''.join(sorted(ds)))
        x1 = hi - lo  # this will be 0
        seq = [int(s)]
        # show the fixed point explicitly (two terms minimum)
        seq.append(x1)
        if SEQ_PRINT_LIMIT is None or len(seq) < SEQ_PRINT_LIMIT:
            seq.append(0)  # 0 → 0
        return True, f"steps 1, sequence: {format_seq_line(seq, SEQ_PRINT_LIMIT)}"

    x = int(s)

    # If we start at the Kaprekar constant, show the fixed point loop explicitly
    if x == target:
        # Apply once to demonstrate the self-loop
        ds = f"{x:0{k}d}"
        hi = int(''.join(sorted(ds, reverse=True)))
        lo = int(''.join(sorted(ds)))
        nxt = hi - lo  # will equal target again
        seq = [x, nxt]
        return True, f"steps 1, sequence: {format_seq_line(seq, SEQ_PRINT_LIMIT)}"

    seq = [x]
    cap = 100  # very safe; convergence is fast
    steps = 0

    while steps < cap and x != target:
        ds = f"{x:0{k}d}"              # keep k digits with leading zeros
        hi = int(''.join(sorted(ds, reverse=True)))
        lo = int(''.join(sorted(ds)))
        x = hi - lo
        steps += 1
        if SEQ_PRINT_LIMIT is None or len(seq) < SEQ_PRINT_LIMIT:
            seq.append(x)

    if x == target:
        return True, f"steps {steps}, sequence: {format_seq_line(seq, SEQ_PRINT_LIMIT)}"
    else:
        return True, f"no convergence within {cap} steps (last: {format_seq_line(seq, SEQ_PRINT_LIMIT)})"


@classifier(
    label="Ducci (4-digit) sequence",
    description="From last 4 digits form (a,b,c,d); iterate (|a−b|,|b−c|,|c−d|,|d−a|) until [0,0,0,0].",
    oeis=None,
    category=CATEGORY,
)
def is_seq_ducci4(n: int):
    start = _ducci_start_from_n(n)
    seq = [start]
    steps = 0
    SEQ_PRINT_LIMIT = CFG("DISPLAY_SETTINGS.SEQUENCE_PRINT_LIMIT", 300)
    DUCCI_MAX_STEPS = CFG("CLASSIFIER.DUCCI.MAX_STEPS", 256)
    # If already zero, show explicit self-loop with one step (like Kaprekar fixed point)
    if start == (0, 0, 0, 0):
        seq.append(start)
        return True, f"collapses to zero at step 1; sequence: {_fmt_ducci_seq(seq, SEQ_PRINT_LIMIT)}"

    while steps < DUCCI_MAX_STEPS:
        nxt = _ducci_next(seq[-1])
        seq.append(nxt)
        steps += 1
        if nxt == (0, 0, 0, 0):
            return True, f"collapses to zero at step {steps}; sequence: {_fmt_ducci_seq(seq, SEQ_PRINT_LIMIT)}"

    # Safety cap (shouldn’t trigger for 4-tuples, but included for robustness)
    return True, f"reached step cap {DUCCI_MAX_STEPS} before zero; sequence: {_fmt_ducci_seq(seq, SEQ_PRINT_LIMIT)}"


@classifier(
    label="Fibonacci mod n: Pisano period",
    description=(
        "Iterate F mod n from (0,1) until (0,1) reappears; "
        "the number of steps is the Pisano period π(n), "
        "includes the residues starting at F₀≡0."
    ),
    oeis="A001175",
    category=CATEGORY,
)
def is_seq_pisano(n: int):
    if n <= 0:
        return False, None
    if n == 1:
        return True, "period 1; residues: [0]"

    SEQ_PRINT_LIMIT = CFG("DISPLAY_SETTINGS.SEQUENCE_PRINT_LIMIT", 300)
    a, b = 0, 1
    residues = [0]
    steps = 0
    CAP = min(1_000_000, n*n)  # π(n) ≤ n²

    while steps < CAP:
        a, b = b, (a + b) % n
        residues.append(a)      # a == F_{k+1} mod n
        steps += 1
        if a == 0 and b == 1:   # (F_m, F_{m+1}) == (0,1)
            period = steps
            break
    else:
        return True, f"no period detected within cap {CAP}"

    if SEQ_PRINT_LIMIT is None or len(residues) <= SEQ_PRINT_LIMIT:
        resid_str = "[" + ", ".join(map(str, residues)) + "]"
    else:
        resid_str = "[" + ", ".join(map(str, residues[:SEQ_PRINT_LIMIT])) + ", …]"

    return True, f"period {period}; residues: {resid_str}"
