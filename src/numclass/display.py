# src/numclass/display.py
from __future__ import annotations

import ast
import operator as _op
import os
import sys
import tomllib as _toml
from collections import defaultdict
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numclass.context import NumCtx

from colorama import Fore, Style, init
from sympy import isprime, nextprime, prevprime
from sympy.ntheory import n_order

from numclass import __version__
from numclass.config import list_profiles_with_descriptions, load_settings, read_current_profile
from numclass.fmt import (
    abbr_int_fast,
    format_duration,
    format_sequence,
    format_zeckendorf,
    visible_len,
    wrap_after_label,
    wrap_description_bullet,
)
from numclass.runtime import CFG
from numclass.runtime import current as _rt_current
from numclass.utility import (
    AliquotCacheEntry,
    _stringify_guarded,
    build_ctx,
    carmichael_with_details,
    clear_screen,
    compute_aliquot_sequence,
    ctx_isprime,
    dec_digits,
    digit_product,
    digit_sum,
    digital_root_sequence,
    enumerate_divisors_limited,
    flatten_dotted,
    format_prime_factors,
    get_terminal_height,
    get_terminal_width,
    mobius_and_radical,
    multiplicative_persistence_sequence,
    parity,
    set_aliquot_cache,
    totient_with_details,
)
from numclass.workspace import workspace_dir

# config (profile I/O)
try:
    import numclass.config as CONFIG
except Exception:
    CONFIG = None

# platform-specific imports
if os.name == "nt":
    try:
        import msvcrt  # type: ignore[attr-defined]
    except Exception:
        msvcrt = None  # type: ignore[assignment]
else:
    try:
        import termios  # type: ignore[attr-defined]
        import tty  # type: ignore[attr-defined]
    except Exception:
        termios = None  # type: ignore[assignment]
        tty = None      # type: ignore[assignment]


def _normalize_results_to_grouped(results: Any) -> dict[str, list[dict]]:
    """
    Support both:
      - legacy dict: {"grouped": {cat: [ {label, details?, is_intersection?} ]}, "skipped": [...]}
      - new list[Outcome]: Outcome has .label, .category, .detail, .is_intersection
    Return {"grouped": {...}, "skipped": [...]}.
    """
    if isinstance(results, dict) and "grouped" in results:
        return {"grouped": results.get("grouped", {}) or {}, "skipped": results.get("skipped", []) or []}

    grouped = defaultdict(list)
    skipped = []
    if isinstance(results, list):
        for o in results:
            cat = getattr(o, "category", None) or "Uncategorized"
            grouped[cat].append({
                "label": getattr(o, "label", ""),
                "details": getattr(o, "detail", None),
                "is_intersection": bool(getattr(o, "is_intersection", False)),
            })
    return {"grouped": dict(grouped), "skipped": skipped}


def _print_aliquot_block(
    *,
    label: str,
    entry: AliquotCacheEntry,
    print_limit: int,
    max_steps: int,
    om,
) -> None:

    seq = entry.seq
    highlight_idx = entry.highlight_idx
    aborted = entry.aborted
    skipped = entry.skipped
    stats = entry.stats or {}

    # If the compute phase returned no sequence and provided a reason, print only that
    if (not seq) and isinstance(skipped, str) and skipped:
        # single, concise line
        om.write(f"{label}{skipped}")
        return

    # Use peak_step (if present) for formatting highlights
    peak_idx = stats.get("peak_step")
    if peak_idx is None:
        peak_idx = -1  # format_sequence will just ignore if out of range

    line, status_lines, _ = format_sequence(
        seq,
        highlight_idx,
        aborted,
        skipped,
        print_limit=print_limit,
        max_steps=max_steps,
        peak_idx=peak_idx,
    )

    steps = int(stats.get("steps", 0))
    peak_val = stats.get("peak_value")
    peak_step = stats.get("peak_step")
    elapsed_s = float(stats.get("elapsed_s", 0.0))

    bits: list[str] = [f"steps: {steps}"]

    # Only show a peak when we actually have one and at least one transition happened
    if peak_val is not None and peak_step is not None and steps > 0:
        peak_word = f"{Fore.YELLOW}{Style.BRIGHT}peak{Style.RESET_ALL}"
        bits.append(f"{peak_word} {peak_val} @ [{peak_step + 1}]")

    if CFG("DISPLAY_SETTINGS.SHOW_ALIQUOT_TIME", True):
        bits.append(format_duration(elapsed_s))

    # add an “Abort/Truncated/Skipped” marker if aborted/skipped
    if aborted or skipped:
        if isinstance(skipped, str) and skipped:
            if steps == 0 and not aborted:
                # e.g. start-value digit cap for stats:
                # don't bother repeating the input number on this line
                bits.append(f"{Fore.YELLOW}Skipped: {skipped}{Style.RESET_ALL}")
            else:
                bits.append(f"{Fore.YELLOW}Abort: {skipped}{Style.RESET_ALL}")
        else:
            bits.append(f"{Fore.YELLOW}Truncated{Style.RESET_ALL}")

    # For pure "skipped with 0 steps", we don't need to show the sequence at all
    if steps == 0 and isinstance(skipped, str) and skipped and not aborted:
        line = ""

    text = line + (f" ({' — '.join(bits)})" if bits else "")
    om.write(wrap_after_label(label, text))

    start_col = len(label)
    for sline in status_lines:
        om.write(" " * start_col + sline)


def print_statistics(n: int, user_input: str, show_details: bool = True, om=None):
    """
    Pretty print all statistics for n.
    output_manager: if None, prints to screen.
    """
    rt = _rt_current()
    debug = rt.debug
    quiet = getattr(om, "quiet", False)
    ctx = None

    ALIGN_WIDTH = 22  # label column

    # helpers for safe alignment/formatting
    def _to_str_int(x) -> str:
        try:
            return str(int(x))  # handles SymPy Integer
        except Exception:
            return str(x)

    def _iterate_to_one_via(f_next, start, cap=1000):
        """Apply f_next(n)->int repeatedly until 1 or cap. Returns (seq, steps)."""
        x = start
        seq = [x]
        steps = 0
        while x != 1 and steps < cap:
            x = f_next(x)
            seq.append(x)
            steps += 1
        return seq, steps

    def _next_phi(m: int) -> int:
        # totient_with_details(n, factors) -> (phi, details)
        fac = factors if m == ctx.n else build_ctx(m).fac
        return totient_with_details(m, fac)[0]

    def _next_lambda(m: int) -> int:
        fac = factors if m == ctx.n else build_ctx(m).fac
        return carmichael_with_details(m, fac)[0]

    def _compute_aliquot_for_display(
        n: int,
        kind: str,
        om,
        ctx: NumCtx | None = None,
    ) -> AliquotCacheEntry:
        """
        Lightweight aliquot-like sequence run for print_statistics.

        - Respects FAST_MODE caps (steps, time, peak digits).
        - Optionally reuses a precomputed NumCtx for the starting n (ctx).
        - In fast_mode, optionally skips the sequence entirely when n is
          above a configured digit threshold (ALIQUOT.STATS_MAX_START_DIGITS).
        """
        rt = _rt_current()
        fast_mode = bool(getattr(rt, "fast_mode", False))

        max_steps = CFG("ALIQUOT.MAX_STEPS", 50)
        step_timeout = CFG("ALIQUOT.STEP_TIME_LIMIT", 0.3)

        stats_total = CFG("ALIQUOT.STATS_MAX_TIME_S", 0.5)
        stats_peak_digits = CFG("ALIQUOT.STATS_MAX_PEAK_DIGITS", 2000)
        stats_start_digits = CFG("ALIQUOT.STATS_MAX_START_DIGITS", 0)

        # --- Fast-mode: start-digit guard for "quick stats" -------------------
        # If n is already enormous, don't even attempt the sequence in stats mode.
        if fast_mode:
            try:
                max_start_digits = int(stats_start_digits)
            except Exception:
                max_start_digits = 0

            if max_start_digits > 0 and dec_digits(n) > max_start_digits:
                stats = {
                    "steps": 0,
                    "peak_value": None,
                    "peak_step": None,
                    "elapsed_s": 0.0,
                }
                return AliquotCacheEntry(
                    seq=[n],
                    highlight_idx=None,
                    aborted=False,
                    skipped="start value digit cap exceeded",
                    stats=stats,
                )

        # --- Time + peak caps (depend on fast_mode) ---------------------------
        if fast_mode:
            # Everyday/default profile: enforce quick stats caps
            total_timeout = stats_total
            max_peak_digits = stats_peak_digits
        else:
            # Aliquot-heavy profiles: 0 or missing means “no global cap”
            total_timeout = None if not stats_total or stats_total <= 0 else stats_total
            max_peak_digits = None if not stats_peak_digits or stats_peak_digits <= 0 else stats_peak_digits

        seq, hi, aborted, skipped, stats = compute_aliquot_sequence(
            n,
            kind=kind,
            max_steps=max_steps,
            step_timeout=step_timeout,
            total_timeout=total_timeout,
            max_peak_digits=max_peak_digits,
            om=om,
            ctx0=ctx,  # re-use σ / σ* for the first step if available
        )

        return AliquotCacheEntry(
            seq=seq,
            highlight_idx=hi,
            aborted=aborted,
            skipped=skipped,
            stats=stats,
        )

    def describe_repeating_decimal(
        n: int,
        style: str | None = None,
        max_digits: int | None = None,
    ) -> tuple[str, int, int | None, int]:
        """
        Preview 1/n in base 10 with correct termination/repetition logic.
        Returns:
          s, pre_len, per_len, rep_preview_len
        """
        STYLE = style if style is not None else CFG("FORMATTING.REPEATING_DEC_STYLE", "underline")
        DIG_CAP = max_digits if max_digits is not None else CFG("DISPLAY_SETTINGS.REPEATING_DEC_PRINT_LIMIT", 60)
        REDUCED_CAP = CFG("FORMATTING.UNITFRAC_MAX_REDUCED", 1_000_000)
        SHOW_REP_MIN = CFG("FORMATTING.REPEATING_DEC_MIN_REP_SHOWN", 12)

        if n == 0:
            return "undefined (division by zero)", 0, 0, 0
        if n < 0:
            n = -n

        # strip 2s and 5s: n = 2^a * 5^b * n'
        d = n
        a = b = 0
        while d % 2 == 0:
            d //= 2
            a += 1
        while d % 5 == 0:
            d //= 5
            b += 1
        pre_len = max(a, b)
        nprime = d  # 1 ⇒ terminating

        def _fmt(int_part: str, pre: str, rep: str) -> str:
            if not rep:
                return f"{int_part}.{pre or '0'}"
            if STYLE == "paren":
                return f"{int_part}.{pre}({rep})"
            if STYLE == "overline":
                return f"{int_part}.{pre}" + "".join(ch + "\u0305" for ch in rep)
            if STYLE == "underline":
                UL, OFF = "\x1b[4m", "\x1b[0m"
                return f"{int_part}.{pre}{UL}{rep}{OFF}"
            return f"{int_part}.{pre}({rep})"

        int_part = "0"
        denom = n

        if nprime == 1:
            # show exactly pre_len digits
            rem = 1 % denom
            digs = []
            for _ in range(pre_len):
                rem *= 10
                digs.append(str(rem // denom))
                rem %= denom
            return _fmt(int_part, "".join(digs) or "0", ""), pre_len, 0, 0

        # repeating: preview via capped long division
        rem = 1 % denom
        digits: list[str] = []
        first_idx: dict[int, int] = {}
        loop_start: int | None = None
        idx = 0
        limit = float("inf") if DIG_CAP is None else int(DIG_CAP)

        while rem and idx < limit:
            if rem in first_idx:
                loop_start = first_idx[rem]
                break
            first_idx[rem] = idx
            rem *= 10
            q = rem // denom
            rem %= denom
            digits.append(str(q))
            idx += 1

        rep_preview_len = 0

        if loop_start is not None:
            pre_preview = "".join(digits[:loop_start])
            rep_preview = "".join(digits[loop_start:])
            rep_preview_len = len(rep_preview)
            if DIG_CAP is not None and loop_start + rep_preview_len > DIG_CAP:
                avail = max(0, DIG_CAP - loop_start)
                if avail < rep_preview_len:
                    rep_preview = rep_preview[:avail] + "…"
                    rep_preview_len = avail
            s_preview = _fmt(int_part, pre_preview, rep_preview)
        else:
            # cap hit before loop → compact non-repeating preview…
            if digits and all(ch == "0" for ch in digits):
                pre_preview = "0…"
            else:
                pre_preview = "".join(digits)
                if DIG_CAP is not None and len(pre_preview) >= DIG_CAP:
                    pre_preview += "…"

            # ALWAYS show a sample of the repetend
            rem2 = pow(10, pre_len, denom)
            rep_sample = []
            for _ in range(max(1, SHOW_REP_MIN)):
                rem2 *= 10
                rep_sample.append(str(rem2 // denom))
                rem2 %= denom
            rep_preview = "".join(rep_sample)
            rep_preview_len = len(rep_preview)
            s_preview = _fmt(int_part, pre_preview, rep_preview)

        per_len = n_order(10, nprime) if nprime <= REDUCED_CAP else None

        return s_preview, pre_len, per_len, rep_preview_len

    num_str = _stringify_guarded(n, label="number")

    if not debug and not quiet:
        clear_screen()
    om.write(f"{Fore.CYAN + Style.BRIGHT}Number statistics:{Style.RESET_ALL}")
    if user_input != num_str:
        om.write(f"  Input:                {user_input}")
    label = "  Number:               "
    colored = f"{Fore.YELLOW}{Style.BRIGHT}{num_str}"
    om.write(f"{wrap_after_label(label, colored, wrap_long_tokens=True)}{Style.RESET_ALL}")

    digit_cnt = dec_digits(n)
    om.write(f"  Digits:               Count={digit_cnt}, Sum={digit_sum(n)}, Product={digit_product(n)}")

    om.write(f"  {'Parity:':<{ALIGN_WIDTH}}{parity(n)}")

    if not quiet:
        print("  Prime:                ⏳ Testing primality", end="\r", flush=True)

    label = "  Prime:                "
    prime_flag = ctx_isprime(n, ctx)
    prime_str = "Yes" if prime_flag else "No"

    _SMALLEST_PRIME = 2
    _MAX_DIGITS = 350
    if digit_cnt < _MAX_DIGITS:
        # Only pay for prime neighbors when small enough
        p_lo = prevprime(n) if n > _SMALLEST_PRIME else None
        p_hi = nextprime(n)
        msg = f"{prime_str}, nearest primes: ◀{p_lo} ▶{p_hi}" if p_lo is not None else f"No, nearest prime: ▶{p_hi}"
    else:
        msg = prime_str + "                    "
    om.write(wrap_after_label(label, msg, wrap_long_tokens=True))

    # Digital root and additive persistence
    label = "  Digital root of |n|:  "
    dr_seq = digital_root_sequence(n)
    if CFG("DISPLAY_SETTINGS.SHOW_DIGITAL_ROOT_SEQ", True):
        dr_seq_str = " → ".join(abbr_int_fast(x) for x in dr_seq)
        om.write(wrap_after_label(label, f"{dr_seq[-1]} {Style.DIM}(sequence: {dr_seq_str}, "
                                  f"additive persistence: {len(dr_seq)-1}){Style.RESET_ALL}", wrap_long_tokens=True))
    else:
        om.write(wrap_after_label(label, f"{dr_seq[-1]} {Style.DIM}(additive persistence: {len(dr_seq)-1}){Style.RESET_ALL}"))

    # Multiplicative persistence (value + sequence)
    label = "  Mult. persistence:    "
    mp_seq = multiplicative_persistence_sequence(n)

    if CFG("DISPLAY_SETTINGS.SHOW_MULTIPLICATIVE_SEQ", True):
        mp_seq_str = " → ".join(abbr_int_fast(x) for x in mp_seq)
        om.write(
            wrap_after_label(
                label,
                f"{mp_seq[-1]} {Style.DIM}(sequence: {mp_seq_str}){Style.RESET_ALL}",
                wrap_long_tokens=True,
            )
        )
    else:
        om.write(wrap_after_label(label, f"{mp_seq[-1]}"))

    # Zeckendorf decomposition
    msg = format_zeckendorf(n)
    if msg is not None:
        om.write(wrap_after_label("  Zeckendorf:           ", msg, wrap_long_tokens=True))

    if not quiet:
        print("  Prime factorization:  Analyzing number ⏳", end="\r", flush=True)

    ctx = ctx or build_ctx(abs(n))

    if not quiet:
        print("                                           ", end="\r", flush=True)

    abs_n = ctx.n

    # Prime factorization; skip 0/±1
    factors = ctx.fac

    def smooth_rough_summary(ctx) -> str:
        """
        Return a compact smooth/rough summary using P-(n), P+(n),
        a bucketed smoothness tier, and optional friability u = log n / log P+(n).
        """
        fac = ctx.fac
        if not fac:  # n in {0, ±1} or factorization unavailable
            return "Smooth/Rough: n has no prime factors"

        pmin = min(fac)
        pmax = max(fac)
        tiers = (10, 100, 1_000, 10_000, 100_000, 1_000_000)
        smooth_bucket = next((B for B in tiers if pmax <= B), f">{tiers[-1]}")
        parts = [f"P⁻={pmin}", f"P⁺={pmax}", f"smooth≤{smooth_bucket}"]
        if ctx.n > 1 and pmax > 1:
            u = log(ctx.n, pmax)
            parts.append(f"u≈{u:.2f}")
        return ", ".join(parts)

    if n != 0 and abs_n > 1:
        pf_str, used_factors = format_prime_factors(
            n, factors=factors, return_factors=True
        )

        # Handle huge numbers where factorization was skipped and
        # ctx.fac == {} → no partial factors to display.
        if not used_factors:
            msg = "factorization unavailable (skipped or incomplete)." if ctx.incomplete else "factorization unavailable."

            om.write(
                wrap_after_label(
                    "  Prime factorization:  ",
                    msg,
                    wrap_long_tokens=True,
                )
            )
        else:
            # Normal partial/complete factorization
            parts = []
            for p in sorted(used_factors):
                e = used_factors[p]
                tag = " (composite)" if (p > 1 and not isprime(p)) else ""
                parts.append(f"{p}^{e}{tag}" if e > 1 else f"{p}{tag}")

            pf_annot = " × ".join(parts)
            if n < 0:
                pf_annot = f"-1 × {pf_annot}"

            om.write(
                wrap_after_label(
                    "  Prime factorization:  ",
                    f"{Fore.CYAN + Style.BRIGHT}{pf_annot}{Style.RESET_ALL}",
                    wrap_long_tokens=True,
                )
            )

    # Ω(n), ω(n) for |n| > 1
    if abs_n > 1:
        big_omega = sum(factors.values())
        little_omega = len(factors)
        om.write(
            f"  {'Prime factor counts:':<{ALIGN_WIDTH}}"
            f"Ω(n) = {big_omega}, ω(n) = {little_omega}"
        )

    # φ(n) and λ(n) for n > 0 (values only; no sequences)
    if n > 0:
        if n == 1:
            phi, lam = 1, 1
            phi_details = "φ(1)=1"
            lam_details = "λ(1)=1"
        elif ctx.incomplete:
            om.write("  Euler's totient φ(n): skipped (incomplete factorization)")
            om.write("  Carmichael λ(n):      skipped (incomplete factorization)")
        else:
            # fast: uses the factors you already computed in ctx
            phi, phi_details = totient_with_details(n, factors)
            lam, lam_details = carmichael_with_details(n, factors)

            label_phi = "  Euler's totient φ(n): "
            om.write(
                wrap_after_label(
                    label_phi,
                    f"{_to_str_int(phi)} {Style.DIM}{phi_details}{Style.RESET_ALL}",
                    wrap_long_tokens=True,
                )
            )

            label_lam = "  Carmichael λ(n):      "
            om.write(
                wrap_after_label(
                    label_lam,
                    f"{_to_str_int(lam)} {Style.DIM}{lam_details}{Style.RESET_ALL}",
                    wrap_long_tokens=True,
                )
            )

    # Repetend / decimal expansion of 1/n in base 10
    DIGITCOUNT = 26
    if n != 0 and len(str(n)) < DIGITCOUNT:
        REPEATING_DEC_STYLE = CFG("FORMATTING.REPEATING_DEC_STYLE", "underline")
        REPEATING_DEC_PRINT_LIMIT = CFG("DISPLAY_SETTINGS.REPEATING_DEC_PRINT_LIMIT", 60)
        sign = "-" if n < 0 else ""
        s, pre_len, per_len, rep_preview_len = describe_repeating_decimal(
            abs(n), style=REPEATING_DEC_STYLE, max_digits=REPEATING_DEC_PRINT_LIMIT
        )
        line = f"  1/n (base 10):        1/n = {sign}{s}"
        show_cnt = CFG("DISPLAY_SETTINGS.SHOW_REPEATING_DEC_COUNT ", True)
        if show_cnt and REPEATING_DEC_STYLE != "underline":
            if isinstance(per_len, int) and per_len > 0:
                line += f" [period {per_len}]"
            elif rep_preview_len:
                line += f" [rep≥{rep_preview_len}]"
        if per_len == 0:
            line += f" terminates, {pre_len} {'digit' if pre_len == 1 else 'digits'}"
        elif per_len is None:
            if pre_len == 0:
                line += " repeats (period skipped)"
            else:
                line += f" repeats, preperiod {pre_len} (period skipped)"
        elif pre_len == 0:
            line += f" repeats, period {per_len}"
        else:
            line += f" repeats, period {per_len}, preperiod {pre_len}"
        om.write(line)

    # Möbius μ(n) and radical rad(n), only for |n| > 1
    if abs_n > 1:
        mu, rad, sqf = mobius_and_radical(factors)
        if mu == 0:
            mu_detail = "not squarefree: has squared prime factor"
        elif mu == 1:
            mu_detail = "squarefree, even number of prime factors"
        else:  # -1
            mu_detail = "squarefree, odd number of prime factors"
        om.write(f"  {'Möbius μ(n):':<{ALIGN_WIDTH}}{mu} {Style.DIM}({mu_detail}){Style.RESET_ALL}")
        om.write(
            wrap_after_label(
                "  Radical rad(n):       ",
                f"{rad}{Style.DIM} (product of distinct primes){Style.RESET_ALL}",
                wrap_long_tokens=True,
            )
        )

    # -- Divisor statistics (of |n|)
    if n != 0 and CFG("DISPLAY_SETTINGS.SHOW_DIVISOR_STATS", True):

        om.write(f"\n{Fore.CYAN + Style.BRIGHT}Divisor statistics of |n|:{Style.RESET_ALL}")

        abs_n = ctx.n
        factors = ctx.fac
        sigma_n = ctx.sigma
        tau_n = ctx.tau
        us = ctx.unitary_sigma
        ut = ctx.unitary_tau

        max_divs = CFG("DISPLAY_SETTINGS.DIVISOR_PRINT_LIMIT", 100)
        show_div = CFG("DISPLAY_SETTINGS.SHOW_DIVISORS", True)

        if ctx.incomplete:
            # --- Incomplete factorization: skip divisor-dependent items
            if show_div:
                om.write("  Divisors:             Unavailable (incomplete factorization)")

        else:

            # --- Fully factored case ---
            om.write(f"  {'Divisor count τ(n):':<22}{tau_n}")

            if show_div:
                shown, truncated = enumerate_divisors_limited(factors, limit=max_divs if max_divs is not None else None)
                s = ", ".join(map(str, shown))
                if truncated:
                    s += (f" … {Style.RESET_ALL}{Fore.YELLOW}"
                          f"(truncated: showing first {len(shown)} of {tau_n}){Style.RESET_ALL}")
                label = "  Divisors:             "
                text = f"{Fore.CYAN + Style.BRIGHT}{s}{Style.RESET_ALL}"
                om.write(wrap_after_label(label, text))

            # Classical sums (σ, s, q)
            if sigma_n is not None:
                s_n = sigma_n - abs_n
                q_n = s_n - 1
                label = "  Classical div. sums:  "
                om.write(wrap_after_label(label, f"σ(n)={sigma_n}, s(n)=σ−n={s_n}, q(n)=s−1={q_n}", wrap_long_tokens=True))
            else:
                label = "  Classical div. sums:  "
                om.write(wrap_after_label(label, f"{Fore.YELLOW}unknown{Style.RESET_ALL}"))

            # Unitary sums (σ*, τ*, s*, q*)
            label = "  Unitary div. sums:    "
            if us is not None and ut is not None:
                us_aliquot = us - abs_n
                un_sums = f"σ*(n)={us}, s*(n)=σ*−n={us_aliquot}, τ*(n)={ut}"
                if n > 1:
                    un_sums += f", q*(n)=s*−1={us_aliquot-1}"
                om.write(wrap_after_label(label, un_sums, wrap_long_tokens=True))
            else:
                om.write(wrap_after_label(label, f"{Fore.YELLOW}unknown{Style.RESET_ALL}"))

            if n > 0:
                LIMIT = CFG("DISPLAY_SETTINGS.SEQUENCE_PRINT_LIMIT", 50)
                max_steps = CFG("ALIQUOT.MAX_STEPS", 50)

                # Aliquot sequence (only when fully factored)
                if CFG("DISPLAY_SETTINGS.SHOW_ALIQUOT_SEQUENCE", True):
                    aliquot_entry = _compute_aliquot_for_display(n, "aliquot", om)
                    set_aliquot_cache(
                        n,
                        max_steps,
                        aliquot_entry,
                    )
                    _print_aliquot_block(
                        label="  Aliquot sequence:     ",
                        entry=aliquot_entry,
                        print_limit=LIMIT,
                        max_steps=max_steps,
                        om=om,
                    )

                # Quasi-aliquot sequence
                if CFG("DISPLAY_SETTINGS.SHOW_QUASI_ALIQUOT_SEQUENCE", True):
                    quasi_entry = _compute_aliquot_for_display(n, "quasi", om)
                    _print_aliquot_block(
                        label="  Quasi-aliquot seq.:   ",
                        entry=quasi_entry,
                        print_limit=LIMIT,
                        max_steps=max_steps,
                        om=om,
                    )

                # Unitary aliquot sequence
                if CFG("DISPLAY_SETTINGS.SHOW_UNITARY_ALIQUOT_SEQUENCE", False):
                    unitary_entry = _compute_aliquot_for_display(n, "unitary", om)
                    _print_aliquot_block(
                        label="  Unitary aliquot seq.: ",
                        entry=unitary_entry,
                        print_limit=LIMIT,
                        max_steps=max_steps,
                        om=om,
                    )
                    # Compare against aliquot if we computed it
                    if CFG("DISPLAY_SETTINGS.SHOW_ALIQUOT_SEQUENCE", True) and unitary_entry.seq == aliquot_entry.seq:
                        om.write(
                            "                        "
                            "(unitary equals aliquot: all terms were square-free)"
                        )
    om.write()
    return ctx


def _as_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "on"}:
        return True
    if s in {"false", "0", "no", "off"}:
        return False
    return default


def print_classifications(n: int, results, show_details: bool = True, om=None, index=None):
    """
    Original-style rendering:
      • Group by category
      • '  - <Label>: <Description>' (white)
      • Category headings bright cyan
      • Details in bright green
      • Intersections can carry multiple details:
          Details: Palindromic: ...
                   Harshad: ...
      • Atomic 'Prime number' is hidden from output
      • 'Skipped...' footer honors FEATURE_FLAGS.SHOW_SKIPPED
    """

    outcomes = results.outcomes
    skipped = results.skipped or []

    # Map intersection label -> rule (for requires, description fallback)
    rules_by_label = {r.label: r for r in (getattr(index, "intersections", []) or [])}

    # Token maps (for turning requires -> base labels)
    tokmap = getattr(index, "label_to_token", {}) or {}
    rev_tok = {tok: lbl for (lbl, tok) in tokmap.items()}

    # ---- Build grouped structure and skip atomic 'Prime number' ----
    grouped: dict[str, list[dict]] = {}
    for o in outcomes:
        # Hide raw atomic "Prime number" (keep if it's an intersection label—rare, but safe)
        if (o.label == "Prime number") and (not getattr(o, "is_intersection", False)):
            continue

        grouped.setdefault(o.category or "Uncategorized", []).append({
            "label":          o.label,
            "is_intersection": bool(getattr(o, "is_intersection", False)),
            # detail can be:
            #   - str (atomic)
            #   - list[(prefix:str, body:str)] (intersection aggregated in classify)
            "details":        o.detail,
        })

    # Determine category order from profile; else alpha
    cats_cfg = CFG("CATEGORIES", {}) or {}
    if any(isinstance(v, bool) for v in cats_cfg.values()):
        enabled_categories = [c for c, enabled in cats_cfg.items() if enabled]
    else:
        enabled_categories = []
    cat_order = enabled_categories or sorted(grouped.keys(), key=str.lower)
    width = get_terminal_width()

    def _intersection_description(lbl: str) -> str:
        """ Helper: synthetic description for intersections if missing """
        _BASE_EXCLUDE_LABEL = "Prime number"   # atomic label we omit when synthesizing text
        _COUNT_SINGLE = 1
        _COUNT_DOUBLE = 2
        _DEFAULT_INTERSECTION_DESC = "Intersection"
        _PREFIX_ALSO = "Also "
        _PREFIX_BOTH = "Both "
        _PREFIX_ALL_OF = "All of: "

        rule = rules_by_label.get(lbl)
        if not rule:
            return ""

        # Prefer explicit description if provided in intersections.toml
        desc = getattr(rule, "description", "") or ""
        if desc:
            return desc

        # Else synthesize from base labels (omit specific atom)
        bases: list[str] = []
        for tok in (rule.requires or []):
            base_lbl = rev_tok.get(tok)
            if base_lbl and base_lbl != _BASE_EXCLUDE_LABEL:
                bases.append(base_lbl)

        if not bases:
            return _DEFAULT_INTERSECTION_DESC

        count = len(bases)
        if count == _COUNT_SINGLE:
            return f"{_PREFIX_ALSO}{bases[0]}"
        if count == _COUNT_DOUBLE:
            return f"{_PREFIX_BOTH}{bases[0]} and {bases[1]}"
        return f"{_PREFIX_ALL_OF}{', '.join(bases)}"

    first_block = True
    for cat in cat_order:
        items = grouped.get(cat, [])
        if not items:
            continue

        if not first_block:
            om.write("")
        first_block = False

        # Category heading (bright cyan)
        om.write(f"{Style.BRIGHT}{Fore.CYAN}{cat}{Style.RESET_ALL}")

        # Items within the category
        for item in sorted(items, key=lambda d: d["label"].lower()):
            lbl = item["label"]
            is_ix = item["is_intersection"]

            # Description: from index.descriptions, else synthesize for intersections
            desc = (index.descriptions.get(lbl) if index else "") or ""
            if not desc and is_ix:
                desc = _intersection_description(lbl)

            # Main line
            label_part = f"{Fore.WHITE}  - {lbl}:{Style.RESET_ALL} "
            wrapped = wrap_description_bullet(label_part, desc or "", width=width, indent_cols=4)
            om.write(wrapped)

            # Details (atomic: str; intersection: list[(prefix, body)])
            DETAIL_PAIR_LEN = 2  # (prefix, body)
            if show_details and item.get("details"):
                det = item["details"]
                base_label = "    Details: "
                if isinstance(det, str):
                    wrapped = wrap_after_label(base_label, det, width=width)
                    om.write(f"{Fore.GREEN}{Style.BRIGHT}{wrapped}{Style.RESET_ALL}")
                elif (
                    isinstance(det, (list, tuple))
                    and det
                    and isinstance(det[0], (list, tuple))
                    and len(det[0]) == DETAIL_PAIR_LEN
                ):
                    indent_label = " " * visible_len(base_label)
                    first = True
                    for prefix, body in det:
                        lbl_for_line = base_label if first else indent_label
                        first = False
                        wrapped = wrap_after_label(lbl_for_line, f"{prefix}: {body}", width=width)
                        om.write(f"{Fore.GREEN}{Style.BRIGHT}{wrapped}{Style.RESET_ALL}")
                # else: unknown structure → ignore silently

    # ---- Skipped footer ----
    def _skip_sort_key(s: str) -> str:
        return s.split(":")[0] if ":" in s else s

    show_skipped = CFG("DISPLAY_SETTINGS.SHOW_SKIPPED", True)
    max_lines = int(CFG("DISPLAY_SETTINGS.SKIPPED_MAX_LINES", 0))

    if skipped and show_skipped:
        skipped_sorted = sorted(skipped, key=_skip_sort_key)
        total = len(skipped_sorted)

        # Always show a summary line
        om.write(
            f"\n{Fore.RED}{Style.BRIGHT}"
            f"Skipped classifications: {total} "
            f"(limits, timeouts or errors)"
            f"{Style.RESET_ALL} "
        )

        # Decide whether to list individual items
        if max_lines <= 0 or total <= max_lines:
            # Full (or uncapped) list
            for lbl in skipped_sorted:
                om.write(f"{Fore.WHITE}{Style.DIM}  - {lbl}{Style.RESET_ALL}")
        else:
            # Too many: keep output compact
            om.write(
                f"{Fore.WHITE}{Style.DIM}"
                f"Rerun after issuing the 'debug on' command, or increase "
                f"SKIPPED_MAX_LINES in your profile to view all skipped classifiers."
                f"{Style.RESET_ALL}"
            )


# cross-platform single-key reader (no newline)
def _get_keypress() -> str:
    try:
        if os.name == "nt" and msvcrt is not None:
            ch = msvcrt.getwch()
            return "" if ch == "\r" else ch
        elif os.name != "nt" and termios is not None and tty is not None:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            return ch
    except Exception:
        pass
    # fallback: line input
    try:
        return input().strip()[:1]
    except Exception:
        return ""


def _screen_header() -> str:
    return (f"{Fore.YELLOW}{Style.BRIGHT}"
            f"Number Classifier v{__version__} — Mathematical Classifications & Curiosities"
            f"{Style.RESET_ALL}")


def _get_cat_to_labels_and_total(index) -> tuple[dict[str, list[str]], int]:
    cat: dict[str, list[str]] = {}
    # atomic
    for lbl in index.funcs:
        c = index.categories.get(lbl) or "Uncategorized"
        cat.setdefault(c, []).append(lbl)
    # intersections
    for rule in getattr(index, "intersections", []) or []:
        c = rule.category or "Uncategorized"
        cat.setdefault(c, []).append(f"{rule.label} (intersection)")
    total = sum(len(v) for v in cat.values())
    return cat, total


def show_classifier_list(index, paged: bool = True) -> None:
    groups, _ = _get_cat_to_labels_and_total(index)

    # build static lines
    lines: list[str] = []
    for cat in sorted(groups.keys(), key=str.lower):
        lines.append(f"{Fore.CYAN}{cat}:{Style.RESET_ALL}")
        for lbl in sorted(groups[cat], key=str.lower):
            # handle intersection suffix for description lookup
            suffix = ""
            base = lbl
            if lbl.endswith(" (intersection)"):
                base = lbl[:-15]
                suffix = " (intersection)"
            desc = index.descriptions.get(base, "")
            left = f"  {Fore.GREEN}{base}{suffix}{Style.RESET_ALL}"
            if desc:
                lines.append(f"{left} — {desc}")
            else:
                lines.append(left)
        lines.append("")  # blank line between categories

    atomic = len(index.funcs) - 1
    intersections = len(getattr(index, 'intersections', []) or [])
    title = (f"{Fore.YELLOW}Available classifiers: {atomic} atomic + {intersections} "
             f"intersections = {atomic + intersections}{Style.RESET_ALL}")
    clear_screen()
    if paged:
        _paginate(lines=lines, title=title)
    else:
        print(_screen_header())
        print()
        print(title)
        print()
        for ln in lines:
            print(ln)


def show_example_inputs(om) -> None:
    clear_screen()
    om.write(_screen_header())
    examples = [
        ("42", "Sum of three cubes, Highly abundant, fun number, ..."),
        ("89", "Disarium, Fibonacci, Palindromic prime, Thick prime, ..."),
        ("153", "Narcissistic, Octal interpretable, Harshad triangular, ..."),
        ("163", "Lucky Euler number"),
        ("276", "Aliquot open sequence, Erdős-Woods, Fermat Pseudoprime, ..."),
        ("561", "Carmichael, Fermat en Euler-Jacobi pseudoprimes, Hexagonal, ..."),
        ("1089", "Digit-reversal constant, perfect square, sum of palindromes, ..."),
        ("1260", "Vampire, Super abundant, Pronic, Highly composite, ..."),
        ("1729", "Ramanujan Taxicab number, Sphenic, ..."),
        ("1911", "Boring number... say what?, ..."),
        ("2357", "Smarandache Wellin, Chen prime, Proth prime, Sexy prime, ..."),
        ("2856", "Large repeating Aliquot sequence, Untouchable, ..."),
        ("6174", "Kaprekar constant, Sum of Palindromes, Sum of squares and cubes, ..."),
        ("8128", "Perfect, Happy, Semiprime, Squarefree, ..."),
        ("47176870", "Busy Beaver, ..."),
        ("169808691", "Strobogrammatic, Cyclops, Self, Lychrel candidate, ..."),
        ("193764528", "Pandigital, Odious, ..."),
        ("459818240", "Triperfect, Practical, Evil, Semiperfect, ..."),
        ("74596893730427", "Keith prime, Gaussian prime, Isolated prime, ..."),
        ("52631578947368421", "Cyclic permutation, ..."),
        ("192137918101841817", "Motzkin, ..."),
        ("1000111100101100100010011111001000101", "Binary interpretable, Deficient, ..."),
        ("6086555670238378989670371734243169622657830773351885970528324860512791691264", "Sublime"),
        ("Large number", "Switch to profile collatz and enter 2**10000-1"),
    ]

    om.write(
        f"{Fore.MAGENTA}{Style.BRIGHT}\n"
        "Try these numbers:\n"
        "Numclass can support numbers op to 100000 digits."
        f"{Style.RESET_ALL}"
    )
    for n, note in examples:
        om.write(f"  {Fore.YELLOW}{n}{Style.RESET_ALL} — {note}")
    om.write()


def _read_one_key() -> str:
    """Return a single key as a string; '\n' for Enter."""
    if os.name == 'nt' and msvcrt:
        ch = msvcrt.getwch()
        if ch == '\r':
            ch = '\n'
        return ch
    if termios and tty and sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch
    try:
        s = input()
    except EOFError:
        return 'q'
    return (s[:1] or '\n')


def _paginate(
    lines: list[str],
    page_size: int | None = None,
    title: str | None = None,
) -> None:
    # build stream (title counts toward pagination)
    stream: list[str] = []
    if title:
        stream.append(title)
        stream.append("")  # blank line
    stream.extend(lines)

    if not stream:
        return

    # No paging if stdout isn't a TTY
    if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        for ln in stream:
            print(ln)
        return

    if page_size is None:
        h = get_terminal_height(default=24)
        # reserve exactly ONE line for the prompt
        page_size = max(1, h - 1)

    i, total = 0, len(stream)
    while i < total:
        end = min(i + page_size, total)
        for ln in stream[i:end]:
            print(ln)
        i = end

        if i < total:
            prompt = "\033[94m[Enter] to continue, 'q' to quit help...\033[0m"
            sys.stdout.write(prompt)
            sys.stdout.flush()

            key = _read_one_key()

            # clear the prompt line
            sys.stdout.write('\r\x1b[2K')
            sys.stdout.flush()

            if key.lower() == 'q':
                break


def show_intro_help(index, om=None) -> None:
    """
    Original-style intro with menu:
      l - List all classifications by category
      r - Show OEIS references
      e - Show some example inputs
      q - Quit help
    """
    cat_to_labels, total = _get_cat_to_labels_and_total(index)

    intro_lines = [
        "",
        f"{Fore.GREEN}Welcome to NumClass{Style.RESET_ALL}",
        f"{'-'*90}",
        f"{Fore.LIGHTWHITE_EX}Explore the fascinating world of integer classifications.{Style.RESET_ALL}",
        "",
        "Discover the hidden mathematical properties of any integer — from the most",
        "famous sequences to delightful oddities.",
        "",
        " • Find out if a number is Prime, Perfect, Abundant, Deficient, ... for classic math fans.",
        " • Amicable, Keith, Cake, Untouchable, ... for the curious.",
        " • Enjoy playful categories: Fun number, Cyclops number, Repdigit, Evil,",
        "   and many more from pop culture, computing, science fiction, internet lore, and memes.",
        "",
        f"{Fore.YELLOW}Features:{Style.RESET_ALL}",
        f" • {total-1} classifications available (atomic + intersection).",
        " • Shows details like explanation, decompositions, whitness or calculation.",
        " • Handy links to the On-Line Encyclopedia of Integer Sequences (OEIS).",
        "",
        f"{Fore.MAGENTA + Style.BRIGHT}Usage in interactive mode:{Style.RESET_ALL}",
        " • Enter an integer to show number/divisor statistics and all its classifications.",
        "   Spaces, Undescores, Commas and Periods are allowed as thousand separators.",
        "   Valid prefixes are: 0b (binary), 0o (octal) and 0x (hexadecimal).",
        "",
        " • Enter a mathematical expression to show the classifications of the calculated number.",
        "   Valid operators are: ( ) ! ** + - * // % << >> & ^ | ?  Example: 9**10-1 > 3486784400.",
        "",
        " • Valid commands are:",
        "   debug on|off|status to switch debug mode on, off or show current status.",
        "   fast on|off|status  to switch fast mode on, off or show current status.",
        "   h or help           to enter this help screen.",
        "   hist                to show a history of entered numbers.",
        "   p                   to show a list of available profiles.",
        "   q or quit           to quit NumClass.",
        "",
        " • Enter a profile name to switch to that profile.",
        "",
        " • Press Ctrl-C during evaluation to abort and skip long calculations.",
        "",
        f"{Fore.CYAN}Tips:{Style.RESET_ALL}",
        " • For command-line options, run: numclass -h or --help",
        " • NumClass adapts to your terminal window size — try resizing for the best view!",
        "",
        f"{Fore.GREEN}Next steps:{Style.RESET_ALL}",
        "",
    ]
    help_lines = [
        f"  {Style.BRIGHT}h{Style.RESET_ALL} - Show main help overview",
        f"  {Style.BRIGHT}s{Style.RESET_ALL} - List effective settings",
        f"  {Style.BRIGHT}c{Style.RESET_ALL} - List all classifications by category",
        f"  {Style.BRIGHT}r{Style.RESET_ALL} - Show OEIS references",
        f"  {Style.BRIGHT}e{Style.RESET_ALL} - Show some example numbers you can try",
        f"  {Style.BRIGHT}l{Style.RESET_ALL} - Show symbols and notation legend",
        f"  {Style.BRIGHT}q{Style.RESET_ALL} - Quit help and start classifying integers",
        "",
    ]

    clear_screen()
    if om:
        om.write(_screen_header())
        for line in intro_lines:
            om.write(line)
    else:
        print(_screen_header())
        print("\n".join(intro_lines))

    while True:
        for line in help_lines:
            om.write(line) if om else print(line)
        # prompt
        print(f"{Fore.LIGHTBLUE_EX}Enter h/s/c/r/e/l/q : {Style.RESET_ALL}", end="", flush=True)
        key = _get_keypress().strip().lower()
        if hasattr(om, "clear_line"):
            try:
                om.clear_line()
            except Exception:
                pass
        if key == "h":
            clear_screen()
            if om:
                om.write(_screen_header())
                for line in intro_lines:
                    om.write(line)
            else:
                print(_screen_header())
                print("\n".join(intro_lines))
            continue
        if key == "s":
            show_effective_settings()
        elif key == "c":
            show_classifier_list(index, paged=True)
        elif key == "r":
            show_oeis_references(index, om)
        elif key == "e":
            show_example_inputs(om)
        elif key == 'l':
            show_symbols_help()
        elif key in ("q", ""):
            break
        else:
            (om.write if om else print)(f"{Fore.RED}Invalid choice:{Style.RESET_ALL} Please type S, C, R, E, L, or Q.\n")
    clear_screen()
    # header after exit
    (om.write if om else print)(_screen_header())


def _osc8_link(text: str, url: str) -> str:
    r"""
    Create a terminal hyperlink using OSC 8:

      ESC ] 8 ; ; URL ST TEXT ESC ] 8 ; ; ST

    Where:
      - ESC = \x1b
      - ST  = ESC followed by backslash (ESC \)
    Most modern terminals (incl. Windows Terminal) make this clickable.
    """
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


def _normalize_oeis_codes(value) -> list[str]:
    """
    Accept a single string or list; split on commas/whitespace; ensure each code
    looks like Axxxxxx (uppercase, 6 digits).
    """
    if value is None:
        return []
    if isinstance(value, str):
        raw = [x.strip() for x in value.replace(",", " ").split() if x.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw = []
        for v in value:
            raw.extend(_normalize_oeis_codes(v))
    else:
        raw = [str(value).strip()]

    out = []
    for tok in raw:
        t = tok.upper()
        if t and not t.startswith("A"):
            t = "A" + t  # be forgiving if someone wrote just digits
        # keep simple; do not enforce exact 6 digits (users may add variants)
        out.append(t)
    # dedupe preserving order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def show_oeis_references(index, om, *, paged: bool = True) -> None:
    """
    Show OEIS references for all configured classifiers:
      • Atomic classifiers (index.funcs + index.oeis)
      • Intersection classifiers (index.intersections[*].oeis)

    Keeps OSC-8 links and aligned label padding.
    """
    # --- collect entries (atomic + intersections) ----------------------------
    entries: list[tuple[str, list[str]]] = []

    # atomic
    for lbl in getattr(index, "funcs", {}) or {}:
        codes = _normalize_oeis_codes((getattr(index, "oeis", {}) or {}).get(lbl))
        if codes:
            entries.append((lbl, codes))  # lbl is the display label

    # intersections
    for rule in (getattr(index, "intersections", []) or []):
        codes = _normalize_oeis_codes(getattr(rule, "oeis", None))
        if codes:
            entries.append((rule.label, codes))

    # nothing to show
    if not entries:
        clear_screen()
        if om:
            om.write(f"{Fore.YELLOW}OEIS references (0){Style.RESET_ALL}")
            om.write("")
            om.write(f"{Style.DIM}No OEIS references are configured yet.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}OEIS references (0){Style.RESET_ALL}\n"
                  f"{Style.DIM}No OEIS references are configured yet.{Style.RESET_ALL}")
        return  # done

    # sort by label (case-insensitive) and compute padding
    entries.sort(key=lambda t: t[0].lower())
    pad = max((len(lbl) for lbl, _ in entries), default=0)

    # build lines with OSC-8 hyperlinks
    lines: list[str] = []
    for lbl, codes in entries:
        links = [_osc8_link(code, f"https://oeis.org/{code}") for code in codes]
        label_padded = lbl.ljust(pad)
        colored = f"{Fore.GREEN}{label_padded}{Style.RESET_ALL}"
        lines.append(f"  {colored} : " + ", ".join(links))

    title = (f"{Fore.YELLOW}OEIS references ({len(entries)}), "
             f"Ctrl+Click the number to open its OEIS page.{Style.RESET_ALL}")

    # --- render --------------------------------------------------------------
    clear_screen()
    if paged:
        _paginate(lines=lines, title=title)  # paging UI
        om.write()
    # non-paged: push through om or stdout directly
    elif om:
        om.write(title)
        om.write("")
        for ln in lines:
            om.write(ln)
        om.write()
    else:
        print(title)
        print()
        for ln in lines:
            print(ln)


def show_symbols_help() -> None:
    """
    Colorized Symbols & Notation help page for numclass (paged).
    """
    init(autoreset=True)

    H1 = Fore.YELLOW
    H2 = Fore.CYAN
    SYMC = Fore.YELLOW + Style.BRIGHT
    EXC = Fore.GREEN
    DEF = Fore.WHITE

    def sym(t: str) -> str:
        return SYMC + t + Style.RESET_ALL

    def ex(t: str) -> str:
        return EXC + t + Style.RESET_ALL

    lines: list[str] = []
    lines.append(_screen_header())
    lines.append("")
    lines.append(H1 + "Symbols & Notation" + Style.RESET_ALL)
    lines.append(DEF + "\nThis page explains symbols and shorthands used in numclass output." + Style.RESET_ALL)

    # General
    lines.append("\n" + H2 + "General" + Style.RESET_ALL)
    lines.append("  " + sym("• ") + sym("a|n") + ": " + DEF + "“a divides n”." + Style.RESET_ALL)
    lines.append("  " + sym("• ") + sym("(a, b)") + ": " + DEF + "open interval { x ∈ ℝ : a < x < b }." + Style.RESET_ALL)
    lines.append("  " + sym("• ") + "Prime factorization: " + ex("n = ∏ pᵢ^{eᵢ}"))
    lines.append("  " + sym("• ") + "Arrows: " + ex("n₀ → n₁ → n₂ → ⋯"))
    lines.append("  " + sym("• ") + "“steps” / “peak”: " + DEF + "step count; peak value." + Style.RESET_ALL)

    # Classical divisor functions
    lines.append("\n" + H2 + "Classical divisor functions" + Style.RESET_ALL)
    lines.append("  " + sym("• ") + sym("τ(n)") + " " + ex("= σ₀(n)") + " number of divisors (also d(n)).")
    lines.append("  " + sym("• ") + sym("σ(n)") + " " + ex("= σ₁(n)") + " sum of all divisors.")
    lines.append(
        "  "
        + sym("• ")
        + sym("s(n)")
        + " "
        + ex("= σ(n) − n")
        + " aliquot sum (proper divisors; includes 1, excludes n)."
    )
    lines.append(
        "  "
        + sym("• ")
        + sym("q(n)")
        + " "
        + ex("= s(n) − 1 = σ(n) − n − 1")
        + " quasi-aliquot sum (excludes both 1 and n)."
    )
    lines.append("  " + sym("• ") + sym("σ_k(n)") + " " + ex("= Σ_{d|n} d^k") + " generalized divisor sum.")

    # Unitary divisor functions
    lines.append("\n" + H2 + "Unitary divisor functions" + Style.RESET_ALL)
    lines.append(DEF + "A divisor d of n is unitary if gcd(d, n/d) = 1." + Style.RESET_ALL)
    lines.append("  " + sym("• ") + sym("σ*(n)") + " " + ex("= ∏ (1 + pᵢ^{eᵢ}") + " sum of unitary divisors.")
    lines.append(
        "  "
        + sym("• ")
        + sym("s*(n)")
        + " "
        + ex("= σ*(n) − n")
        + " sum of proper unitary divisors (includes 1, excludes n)."
    )
    lines.append("  " + sym("• ") + sym("τ*(n)") + " " + ex("= 2^ω(n)") + " number of unitary divisors.")
    lines.append("  " + sym("• ") + sym("q*(n)") + " " + ex("= s*(n) − 1 = σ*(n) − n − 1") +
                 " unitary quasi-aliquot sum (sum of unitary divisors excluding 1 and n).")

    # Prime-factor functions & related
    lines.append("\n" + H2 + "Prime-factor functions & related" + Style.RESET_ALL)
    lines.append("  " + sym("• ") + sym("ω(n)") + " number of distinct prime factors.")
    lines.append("  " + sym("• ") + sym("Ω(n)") + " total prime factors with multiplicity.")
    lines.append("  " + sym("• ") + sym("rad(n)") + " " + ex("= ∏ pᵢ") + " radical (product of distinct primes).")
    lines.append(
        "  "
        + sym("• ")
        + sym("μ(n)")
        + " Möbius function; 1 if n=1, (-1)^k if n is the product of k distinct primes, 0 if n is divisible by a square > 1."
    )
    lines.append("  " + sym("• ") + sym("φ(n)") + " Euler’s totient; counts integers ≤ n that are coprime to n.")
    lines.append("  " + sym("• ") + sym("λ(n)") + " Carmichael function; the least m with aᵐ ≡ 1 (mod n).")

    # Prime letters & sets
    lines.append("\n" + H2 + "Prime letters & sets" + Style.RESET_ALL)
    lines.append("  " + sym("• ") + sym("𝙋") + " the set of all prime numbers.")
    lines.append("  " + sym("• ") + sym("P⁺(n)") + " largest prime factor of n (n>1).")
    lines.append("  " + sym("• ") + sym("P⁻(n)") + " smallest prime factor of n (n>1).")
    lines.append("  " + sym("• ") + sym("p, q") + " generic symbols for primes (also used as bases/parameters).")

    # Common analysis symbols
    lines.append("\n" + H2 + "Common analysis symbols" + Style.RESET_ALL)
    lines.append("  " + sym("• ") + sym("∈") + " Element; an element (or member) of a set.")
    lines.append("  " + sym("• ") + sym("n!") + " Factorial; product of 1..n.")
    lines.append("  " + sym("• ") + sym("ε (> 0)") + " arbitrarily small positive constant (e.g., σ(n) / n^{1+ε}).")
    lines.append("  " + sym("• ") + sym("Σ") + " summation sign, e.g., " + ex("Σ_{d|n} f(d)."))
    lines.append("  " + sym("• ") + sym("≡") + " congruence: a ≡ b (mod m).")

    # Iterations & sequences
    lines.append("\n" + H2 + "Iterations & sequences" + Style.RESET_ALL)
    lines.append("  " + sym("• ") + "Aliquot sequence: s-iteration — repeatedly apply s(n).")
    lines.append("  " + sym("• ") + "Quasi-aliquot sequence: q-iteration — repeatedly apply " + ex("q(n)=s(n)−1."))
    lines.append("  " + sym("• ") + "Unitary aliquot sequence: s*-iteration — repeatedly apply s*(n).")

    # Specific sequences used in classifiers
    lines.append("\n" + H2 + "Specific sequences & forms (used in classifiers)" + Style.RESET_ALL)
    lines.append("  " + sym("• ") + sym("Bell numbers") + " " + ex("B(n)") + "; the n-th Bell number.")
    lines.append("  " + sym("• ") + sym("Catalan numbers") + " " + ex("Cₙ") + "; the n-th Catalan number.")
    lines.append("  " + sym("• ") + sym("Fibonacci numbers") + " " + ex("F(n)") + "; the n-th Fibonacci number.")
    lines.append("  " + sym("• ") + sym("Lucas numbers") + " " + ex("L(n)") + "; the n-th Lucas number.")
    lines.append("  " + sym("• ") + sym("Pell numbers") + " " + ex("P(n)") + "; the n-th Pell number.")
    lines.append(Style.RESET_ALL)

    clear_screen()
    _paginate(lines=lines, title=None)


def show_effective_settings() -> None:
    """
    Show all settings (profile + discovered defaults) in a paged view.
      - Yellow: explicitly set in the profile
      - White : discovered default
      - Red   : present in profile but unused in code
      - Pretty-print CATEGORIES.*
    """
    # --- helpers (unchanged in spirit) --------------------------------------

    NON_LITERAL_DEFAULT = object()  # present in code, default not a literal
    _ = CFG("BEHAVIOUR.FAST_MODE")  # Not used in code, needed for discovery of setting

    def _is_cfg_call(node: ast.Call) -> bool:
        f = node.func
        return (isinstance(f, ast.Name) and f.id == "CFG") or (
            isinstance(f, ast.Attribute) and f.attr == "CFG"
        )

    def _iter_py_files(roots: list[Path]) -> list[Path]:
        out: list[Path] = []
        skip = {
            ".venv", "build", "dist", "__pycache__", ".ruff_cache",
            ".pytest_cache", ".mypy_cache", ".git", "site-packages", "egg-info",
        }
        for root in roots:
            if not root or not root.exists():
                continue
            for p in root.rglob("*.py"):
                if any(part in skip for part in p.parts):
                    continue
                out.append(p)
        return out

    def _const_eval(node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = _const_eval(node.operand)
            if isinstance(v, (int, float)):
                return +v if isinstance(node.op, ast.UAdd) else -v
            return NON_LITERAL_DEFAULT
        if isinstance(node, ast.BinOp):
            left = _const_eval(node.left)
            right = _const_eval(node.right)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                return NON_LITERAL_DEFAULT
            ops = {
                ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
                ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod, ast.Pow: _op.pow,
                ast.LShift: _op.lshift, ast.RShift: _op.rshift,
                ast.BitAnd: _op.and_, ast.BitOr: _op.or_, ast.BitXor: _op.xor,
            }
            for t, fn in ops.items():
                if isinstance(node.op, t):
                    return fn(left, right)
            return NON_LITERAL_DEFAULT
        return NON_LITERAL_DEFAULT

    def _value_from_ast(node: ast.AST | None) -> object:
        if node is None:
            return NON_LITERAL_DEFAULT
        v = _const_eval(node)
        if v is not NON_LITERAL_DEFAULT:
            return v
        try:
            return ast.literal_eval(node)
        except Exception:
            return NON_LITERAL_DEFAULT

    def _collect_cfg_defaults(roots: list[Path]) -> dict[str, object]:
        found: dict[str, object] = {}

        def _maybe_record(key_node: ast.AST, val_node: ast.AST | None):
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                return
            key = key_node.value.strip()
            if key in found:
                return
            found[key] = _value_from_ast(val_node)

        for file in _iter_py_files(roots):
            try:
                src = file.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(file))
            except Exception:
                continue

            class V(ast.NodeVisitor):
                def visit_Call(self, node: ast.Call):
                    _ARGS = 2
                    if _is_cfg_call(node):
                        if node.args:
                            val_node = node.args[1] if len(node.args) >= _ARGS else None
                            _maybe_record(node.args[0], val_node)
                        if node.keywords and node.args:
                            kwmap = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                            for param in ("default", "value", "fallback"):
                                if param in kwmap:
                                    _maybe_record(node.args[0], kwmap[param])
                                    break
                    self.generic_visit(node)
            V().visit(tree)

        return found

    def _format_value_for_display(val: object) -> str:
        if val is NON_LITERAL_DEFAULT:
            return "(non-literal default)"
        if val is None:
            return "(unlimited)"
        return repr(val)

    def _pretty_category(label: str) -> str:
        s = label.replace("_AND_", " and ").replace("_", " ").lower()
        keep_lower = {"and", "of", "the", "in", "on"}
        words = [(w.capitalize() if (i == 0 or w not in keep_lower) else w) for i, w in enumerate(s.split())]
        s = " ".join(words)
        return s.replace("Divisor based", "Divisor-based")

    # --- collect data --------------------------------------------------------

    profile_name = read_current_profile() or "default"
    selected = load_settings(profile_name)

    raw: dict[str, object] | None = None
    src_path: Path | None = getattr(selected, "_source", None)

    if not src_path:
        try:
            cand = (workspace_dir() / "profiles" / f"{profile_name}.toml")
            if cand.exists():
                src_path = cand
        except Exception:
            src_path = None

    if src_path:
        try:
            with open(src_path, "rb") as f:
                raw = _toml.load(f)
        except Exception:
            raw = None

    flat_profile = flatten_dotted(raw) if raw else {}

    roots: list[Path] = []
    try:
        ws = workspace_dir()
    except Exception:
        ws = None
    if ws and ws.exists():
        if (ws / "classifiers").exists():
            roots.append(ws / "classifiers")
        roots.append(ws)
    pkg_root = Path(__file__).resolve().parent
    roots.append(pkg_root)

    discovered = _collect_cfg_defaults(roots)
    all_keys = set(discovered.keys()) | set(flat_profile.keys())

    # --- render lines & paginate --------------------------------------------

    lines: list[str] = []
    lines.append(f"Effective settings are auto discovered from Python code and file: {profile_name}.toml")
    lines.append("Value legend: white: default, "
                 f"{Fore.YELLOW}yellow{Style.RESET_ALL}: set in profile, "
                 f"{Fore.RED}red{Style.RESET_ALL}: not used in Python code.")
    lines.append("")

    if not all_keys:
        lines.append("No settings discovered.")
        clear_screen()
        _paginate(lines=lines, title=None)
        return

    for key in sorted(all_keys, key=str.lower):
        effective = flat_profile.get(key, discovered.get(key))
        low = key.lower()
        if low in {"categories", "include_exclude"} and (effective is None or effective == {}):
            continue

        upper = key.upper()
        is_category_ns = upper.startswith("CATEGORIES.") or upper == "CATEGORIES"
        is_classifier_ns = upper.startswith("INCLUDE_EXCLUDE.") or upper == "INCLUDE_EXCLUDE"

        if (key in flat_profile) and (key not in discovered) and not (is_category_ns or is_classifier_ns):
            val_s = f"{Fore.RED}{_format_value_for_display(effective)}{Style.RESET_ALL}"
        elif key in flat_profile:
            val_s = f"{Fore.YELLOW}{_format_value_for_display(effective)}{Style.RESET_ALL}"
        else:
            val_s = _format_value_for_display(effective)

        if upper.startswith("CATEGORIES."):
            tail = key.split(".", 1)[1]
            pretty = f"categories.{_pretty_category(tail)}"
            lines.append(f"{pretty:.<50} {val_s}")
            continue

        suffix = ""
        if upper.startswith("CLASSIFIERS.") and isinstance(effective, bool):
            suffix = " (whitelisted)" if effective else " (blacklisted)"

        lines.append(f"{key.lower():.<50} {val_s}{suffix}")

    clear_screen()
    _paginate(
        lines=lines,
        title=f"{Fore.YELLOW}Effective settings — active profile: '{profile_name}'{Style.RESET_ALL}",
    )
    print()


def print_profiles_with_descriptions() -> None:
    try:
        pairs = list_profiles_with_descriptions()
    except Exception:
        pairs = []

    if not pairs:
        print("\nAvailable profiles: (none)")
        return

    current = None
    try:
        current = read_current_profile()
    except Exception:
        pass

    lines = []
    for name, desc in pairs:
        mark = "🡆" if current and name == current else " "
        lines.append(f"{mark} {name:13} — {desc}")
    print("\nAvailable profiles:\n  " + "\n  ".join(lines))
