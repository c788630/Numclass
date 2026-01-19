# -----------------------------------------------------------------------------
#  Utility functions
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import sys
import tomllib as _toml  # py311+
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib.resources import files as pkg_files
from math import isqrt
from multiprocessing import Process, Queue
from pathlib import Path
from time import perf_counter
from typing import Any

import gmpy2
from colorama import Fore, Style
from sympy import factorint, isprime, lcm

from numclass.context import NumCtx
from numclass.runtime import CFG
from numclass.runtime import current as _rt_current
from numclass.workspace import workspace_dir

# --- Aliquot sequence cache (per run, in-memory) -----------------------------
# Keyed by (n, max_steps). Stores exactly what compute_aliquot_sequence returns.

_ALIQUOT_CACHE: dict[tuple[int, int], AliquotCacheEntry] = {}
_SUM3C_SOURCE: ContextVar[str | None] = ContextVar("_SUM3C_SOURCE", default=None)


class UserInputError(Exception):
    pass


def _factorint_worker(q: Queue[dict[int, int] | Exception], n: int, limit: int | None, use_ecm: bool) -> None:
    """
    Worker run in a subprocess to call sympy.factorint safely.
    Result or exception is put into the queue.
    """
    try:
        part = factorint(
            n,
            limit=limit,
            use_trial=True,
            use_rho=True,
            use_pm1=True,
            use_ecm=use_ecm,
            verbose=False,
        )
    except Exception as e:  # propagate errors to parent
        q.put(e)
    else:
        q.put(part)


def dec_digits(n: int) -> int:
    """Exact decimal digit count without str(); handles n >= 0."""
    n = abs(n)
    if n == 0:
        return 1
    # lower/upper estimates via bit_length
    # floor(log10(n)) ~= floor(bitlen*log10(2))
    bl = n.bit_length()
    # 0.30102999566 ~ log10(2)
    est = int((bl * 30103) // 100000)  # tiny int math, close to log10(n)
    # bring into correct decade with at most a couple of steps
    p10 = 10 ** est
    if n < p10:
        while n < p10:
            est -= 1
            p10 //= 10
    else:
        p10 *= 10
        while n >= p10:
            est += 1
            p10 *= 10
    return est + 1


def _effective_digit_limit() -> int | None:
    """
    Effective decimal-digit limit for stringifying integers.

    - Primary source: profile setting BEHAVIOUR.MAX_DIGITS.
    - Secondary: Python's own guard (sys.get_int_max_str_digits), if available.

    We return the tighter (smaller) of the two when both exist.
    """
    profile_limit = CFG("BEHAVIOUR.MAX_DIGITS", 100_000)
    try:
        py_limit = sys.get_int_max_str_digits()
    except Exception:
        py_limit = None

    # Normalize to int or None
    try:
        profile_limit = int(profile_limit)
    except Exception:
        profile_limit = None

    if profile_limit is None and py_limit is None:
        return None
    if profile_limit is None:
        return py_limit
    if py_limit is None:
        return profile_limit
    # Both set: respect the tighter one
    return min(profile_limit, py_limit)


def _stringify_guarded(n: int, label: str = "number") -> str:
    """Return str(n) or raise a friendly user error if it exceeds the digit guard."""
    limit = _effective_digit_limit()

    # Proactive check based on our own digit counter
    if limit is not None and dec_digits(n) > limit:
        raise UserInputError(
            f"{label} has more than {limit} decimal digits. "
            "Increase the limit in the profile or pass a smaller value."
        )

    try:
        return str(n)
    except ValueError:
        # Belt & suspenders: if Python still refuses (e.g. because its internal
        # guard is stricter than we thought), surface the *same* friendly message.
        # Recompute the current effective limit in case it changed.
        limit = _effective_digit_limit()
        if limit is not None:
            raise UserInputError(
                f"{label} has more than {limit} decimal digits. "
                "Increase the limit in the profile or pass a smaller value."
            ) from None
        # Absolute fallback: no known limit, but conversion failed anyway
        raise UserInputError(
            f"{label} is too large to be converted to a string under "
            "the current settings. Increase the digit limit in the profile "
            "or pass a smaller value."
        ) from None


@dataclass(frozen=True, slots=True)
class AliquotCacheEntry:
    seq: list[int]
    highlight_idx: tuple[int, int] | None
    aborted: bool
    skipped: str | bool
    stats: dict[str, Any]  # steps, peak_value, peak_step, elapsed_s


def _token(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).upper().strip("_")


def set_aliquot_cache(n: int, max_steps: int, entry: AliquotCacheEntry) -> None:
    _ALIQUOT_CACHE[(int(n), int(max_steps))] = entry


def get_aliquot_cache(n: int, max_steps: int) -> AliquotCacheEntry | None:
    return _ALIQUOT_CACHE.get((int(n), int(max_steps)))


def clear_screen(keep_scrollback: bool = False) -> None:
    """
    Clear the terminal screen.
    - On Windows: uses 'cls'
    - On POSIX: ANSI sequences; optionally clear scrollback
    """
    try:
        if os.name == "nt":
            os.system("cls")
        else:
            seq = "\033[H\033[2J" if keep_scrollback else "\033[3J\033[H\033[2J"
            sys.stdout.write(seq)
            sys.stdout.flush()
    except Exception:
        try:
            os.system("cls" if os.name == "nt" else "clear")
        except Exception:
            pass


def _read_text_from_workspace_or_package(name: str) -> str | None:
    """Return text of a data file from workspace/data first, else from numclass.data, else None."""
    name = Path(name).name
    wp = workspace_dir() / "data" / name
    if wp.is_file():
        return wp.read_text(encoding="utf-8-sig")
    try:
        return pkg_files("numclass.data").joinpath(name).read_text(encoding="utf-8")
    except Exception:
        return None


def _parse_sum3c_toml(text: str) -> dict[int, tuple[int, int, int]]:
    NUM_CUBES = 3
    data = _toml.loads(text)
    out: dict[int, tuple[int, int, int]] = {}
    sol = data.get("solutions")
    if isinstance(sol, dict):
        for k, v in sol.items():
            try:
                n = int(k)
            except Exception:
                continue
            if isinstance(v, (list, tuple)) and len(v) == NUM_CUBES:
                try:
                    a, b, c = int(v[0]), int(v[1]), int(v[2])
                except Exception:
                    continue
                out[n] = (a, b, c)
    return out


HARD_FACTORS_FILE = "hard_factors.txt"


@cache
def _load_hard_factors() -> dict[int, dict[int, int]]:
    """
    Load a table of 'hard' factorizations from workspace or package data.

    File format (hard_factors.txt):

        # comment
        n  p1^e1  p2^e2  ...

    where n, p1, p2,... are decimal integers (underscores allowed),
    and exponents are optional (default 1).
    """

    name = HARD_FACTORS_FILE
    text: str | None = None

    # Workspace data overrides packaged data
    up = Path(workspace_dir()) / "data" / name
    if up.is_file():
        text = up.read_text(encoding="utf-8")
    else:
        try:
            text = pkg_files("numclass.data").joinpath(name).read_text(encoding="utf-8")
        except FileNotFoundError:
            _HARD_FACTORS = {}
            return _HARD_FACTORS

    table: dict[int, dict[int, int]] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        _EXPECTED_COLS = 2
        parts = re.split(r"\s+", line)
        if len(parts) < _EXPECTED_COLS:
            continue

        # Parse n, stripping underscores
        try:
            n_str = parts[0].replace("_", "")
            n_val = int(n_str)
        except ValueError:
            continue

        fac: dict[int, int] = {}
        for token in parts[1:]:
            tok = token.strip()
            if not tok or tok.startswith("#"):
                break

            # p or p^e
            if "^" in tok:
                base_str, exp_str = tok.split("^", 1)
            else:
                base_str, exp_str = tok, "1"

            try:
                base = int(base_str.replace("_", ""))
                exp = int(exp_str.replace("_", ""))
            except ValueError:
                continue

            if exp <= 0:
                continue
            fac[base] = fac.get(base, 0) + exp

        if fac:
            table[n_val] = fac

    return table


def _lookup_hard_factor(n: int) -> dict[int, int] | None:
    """
    Return a precomputed factorization map for n if present in hard_factors.txt,
    else None. The dict maps prime -> exponent, just like sympy.factorint.
    """
    table = _load_hard_factors()
    return table.get(n)


def is_square(x: int) -> bool:
    if x < 0:
        return False
    r = isqrt(x)
    return r * r == x


def load_sum_of_three_cubes_toml(name: str = "sum_of_three_cubes.toml") -> dict[int, tuple[int, int, int]]:
    """
    Return { n: (a,b,c) }.
    Resolution order:
      1) <workspace>/data/<name> (UTF-8-SIG)
      2) packaged: numclass.data/<name> (UTF-8)

    Side effect:
      - Records the source string in a ContextVar retrievable via last_sum3c_source().
    """
    name = Path(name).name

    # 1) Workspace override
    wp = workspace_dir() / "data" / name
    if wp.is_file():
        try:
            text = wp.read_text(encoding="utf-8-sig")
            _SUM3C_SOURCE.set(str(wp))
            return _parse_sum3c_toml(text)
        except (OSError, UnicodeError, ValueError) as e:
            _SUM3C_SOURCE.set(f"{wp} (read failed: {e.__class__.__name__})")
            return {}

    # 2) Packaged fallback
    try:
        text = (pkg_files("numclass.data") / name).read_text(encoding="utf-8")
        _SUM3C_SOURCE.set(f"package:numclass.data/{name}")
        return _parse_sum3c_toml(text)
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as e:
        _SUM3C_SOURCE.set(f"package:numclass.data/{name} (read failed: {e.__class__.__name__})")
        return {}


def _divisors_from_factorization(f: dict[int, int]) -> list[int]:
    """Return all positive divisors from a prime-power factorization dict."""
    ds = [1]
    for p, e in f.items():
        cur = []
        pe = 1
        for _ in range(e + 1):
            for d in ds:
                cur.append(d * pe)
            pe *= p
        ds = cur
    return sorted(set(ds))


def _fmt_pairs_vdw(pairs: list[tuple[int, int]]) -> str:
    """Format Van der Waerden witnesses as 'W(r,k), W(r,k), ...'."""
    pairs = sorted(pairs)
    return ", ".join([f"W({r},{k})" for (r, k) in pairs])


def _fmt_pairs_ramsey(pairs: list[tuple[int, int]]) -> str:
    """Format Ramsey witnesses as 'R(s,t), R(s,t), ...' (canonicalizing (s,t) as (min,max))."""
    pairs = sorted({(min(s, t), max(s, t)) for (s, t) in pairs})
    return ", ".join([f"R({s},{t})" for (s, t) in pairs])


def _merge_fac(dst: dict[int, int], src: dict[int, int]) -> None:
    for p, e in src.items():
        dst[p] = dst.get(p, 0) + e


def _prime_part_value(fac: dict[int, int]) -> int:
    # multiply only prime bases (ignore composite cofactors)
    v = 1
    for p, e in fac.items():
        if p > 1 and isprime(p):
            v *= p**e
    return v


def _cofactor_from_map(n: int, fac: dict[int, int]) -> int:
    return n // _prime_part_value(fac)


def _too_big(x: int) -> bool:
    """
    Return True if x exceeds configured size caps.

    Uses bit_length for a fast upper bound and dec_digits for digit caps, so we
    avoid expensive str(x) on huge integers.
    """
    bit_cap = int(CFG("FACTORING.BITLEN_CAP", 10000))  # ~3k decimal digits
    digit_cap = int(CFG("FACTORING.DIGIT_CAP", 0))     # 0 = disabled

    # Hard cap on bit length
    if x.bit_length() > bit_cap:
        return True

    # Optional decimal-digit cap
    if digit_cap:
        return dec_digits(x) > digit_cap

    return False


def _factorint_bounded(n: int, *, limit: int | None, use_ecm: bool, max_time_s: float) -> dict[int, int] | None:
    """
    Run factorint(n, limit=..., use_ecm=...) in a subprocess and enforce
    a hard wall-clock timeout. Returns:

        dict  -> successful factorization result
        None  -> timeout (no reliable result)

    Any exception raised inside factorint is re-raised in the parent.
    """
    q: Queue[dict[int, int] | Exception] = Queue()
    p = Process(target=_factorint_worker, args=(q, n, limit, use_ecm))
    p.start()
    p.join(max_time_s)

    if p.is_alive():
        # Hard timeout: kill the worker and treat remaining as composite.
        p.terminate()
        p.join()
        return None

    if q.empty():
        # Shouldn't happen, but be defensive: treat as timeout.
        return None

    res = q.get()
    if isinstance(res, Exception):
        # Re-raise so _factor can handle OverflowError/ValueError as before.
        raise res
    return res


def _factor(n: int, *, max_time_s: float | None = None) -> dict[int, int]:
    """
    Return the (possibly partial) factorization of ``n`` as a {prime: exponent} map.

    This function provides all of NumClass’s general-purpose factoring logic.
    It merges several layers:

    • **Hard-factors override**
      If ``n`` (or a divisor of ``n``) appears in ``hard_factors.txt``,
      return those factors immediately.  This is used for large RSA numbers,
      Cunningham numbers, Fermat numbers with known parts, etc.

    • **Small/medium-n fast path**
      If ``n`` has at most ``FACTORING.SMALL_DIGIT_THRESHOLD`` decimal digits
      (default: 80) and is not too large according to ``_too_big(n)``, we call
      SymPy’s ``factorint`` directly (in-process, no time budget).  This is
      faster and avoids the overhead of subprocesses for the common cases.

    • **Big-n general path with a strict wall-clock budget**
      For larger integers, we enforce a *true* time limit.  The total allowed
      wall-clock time is:

          ``max_time_s``  (explicit parameter), or
          ``FACTORING.MAX_TIME_S`` from the profile (default: 2.0 seconds).

      A sequence of increasing ``limit`` values (``FACTORING.STEP_LIMITS``)
      is tried, but **every attempt uses the bounded subprocess wrapper
      ``_factorint_bounded``**, never plain ``factorint``.  Each subprocess
      gets a ``max_time_s`` equal to the remaining budget.  When the time is
      exhausted, the remaining composite is returned as a single unfactored
      cofactor.

      This guarantees that the total time spent by `_factor` will not exceed
      the configured budget, even for extremely hard inputs such as
      ``2**10000 - 1``.

    • **DIGIT_CAP / BITLEN_CAP safety gates**
      The early helper ``_too_big(n)`` applies:
          - ``FACTORING.BITLEN_CAP`` (default: 10M bits), and
          - ``FACTORING.DIGIT_CAP`` (default: 0 = disabled).
      If either limit is exceeded, we immediately return ``{n: 1}`` without
      attempting any factoring.

    • **Composite-base detection**
      After collecting partial factors, any remaining composite cofactor
      is included with exponent 1.  The caller (e.g. ``build_ctx``) can use
      ``has_composite_base(fac)`` to detect incomplete factorization.

    • **Debug progress reporting**
      When ``runtime.debug`` is enabled, periodic status lines are printed to
      ``stderr`` while factoring, showing:
          - elapsed time,
          - time left in the budget,
          - current ``limit`` value,
          - approximate bit-length of the remaining composite.

      Output is throttled to once every 0.5 seconds.

    Parameters
    ----------
    n : int
        The integer to factor (n ≥ 0).

    max_time_s : float | None
        Optional explicit wall-clock budget in seconds.
        If ``None``, the profile value ``FACTORING.MAX_TIME_S`` is used.

    Returns
    -------
    dict[int, int]
        A mapping ``{prime: exponent}``.  If factoring is incomplete
        (due to timeouts, size caps, or composite leftovers),
        the remaining cofactor appears as a single composite key.

    Notes
    -----
    This function is designed for *general* use (classification, σ/τ, etc.).
    For repeated aliquot work, the separate fast-sigma / fast-usigma machinery
    should be used instead of calling ``_factor`` in a loop.

    The function is safe for very large inputs:
    - Hard-factors are used when available,
    - Impossible-size numbers return ``{n: 1}`` immediately,
    - Big numbers always use bounded subprocesses,
    - The configured time budget is strictly respected.

    Examples
    --------
    >>> _factor(5040)
    {2: 4, 3: 2, 5: 1, 7: 1}

    >>> _factor(2**10000 - 1)
    {<huge_composite>: 1}    # returned quickly (caps/timeout)
    """
    rt = _rt_current()
    debug = rt.debug

    # 1) Hard-factors override
    hard = _lookup_hard_factor(n)
    if hard is not None:
        return dict(hard)

    # 2) Small/medium-n fast path: direct factorint, no time logic
    SMALL_DIGIT_THRESHOLD = CFG("FACTORING.SMALL_DIGIT_THRESHOLD", 80)
    if dec_digits(n) <= SMALL_DIGIT_THRESHOLD and not _too_big(n):
        return factorint(
            n,
            use_trial=True,
            use_rho=True,
            use_pm1=True,
            use_ecm=bool(CFG("FACTORING.USE_ECM", False)),
            verbose=False,
        )

    # 3) General big-n path — *always bounded*
    max_time = float(
        max_time_s if max_time_s is not None else CFG("FACTORING.MAX_TIME_S", 2.0)
    )
    use_ecm = bool(CFG("FACTORING.USE_ECM", False))
    step_limits = list(CFG("FACTORING.STEP_LIMITS", [1000, 10000, 100000, 1000000]))

    t0 = perf_counter()
    last_report = 0.0
    remaining = n
    acc: dict[int, int] = {}

    # Early size cap check
    if _too_big(remaining):
        acc[remaining] = acc.get(remaining, 0) + 1
        return acc

    _INTERVAL = 0.5
    for lim in [*step_limits, None]:
        if remaining == 1:
            break

        elapsed = perf_counter() - t0
        time_left = max_time - elapsed

        if debug and elapsed - last_report >= _INTERVAL:
            bits = remaining.bit_length()
            print(
                f"[factor] t={elapsed:5.2f}s left={max(time_left, 0):5.2f}s "
                f"lim={lim!r} remaining≈{bits} bits",
                file=sys.stderr,
            )
            last_report = elapsed

        if time_left <= 0:
            acc[remaining] = acc.get(remaining, 0) + 1
            break

        if _too_big(remaining):
            acc[remaining] = acc.get(remaining, 0) + 1
            break

        # Hard limit for big numbers: always use bounded variant
        try:
            part = _factorint_bounded(
                remaining,
                limit=lim,
                use_ecm=use_ecm,
                max_time_s=max(time_left, 0.05),  # tiny positive safeguard
            )
            if part is None:
                acc[remaining] = acc.get(remaining, 0) + 1
                break
        except (OverflowError, ValueError):
            acc[remaining] = acc.get(remaining, 0) + 1
            break

        _merge_fac(acc, {p: e for p, e in part.items() if p > 1 and isprime(p)})
        remaining = _cofactor_from_map(n, acc)

    if remaining > 1:
        acc[remaining] = acc.get(remaining, 0) + 1

    return acc


@lru_cache(maxsize=200_000)
def sigma_ctx(n: int) -> int | None:
    """
    σ(n) via build_ctx, reusing the same factoring pipeline (hard_factors,
    time caps, composite-base detection). Returns None if factorization is
    incomplete.
    """
    ctx = build_ctx(int(n))
    if ctx.incomplete or ctx.sigma is None:
        return None
    return ctx.sigma


def proper_sum_ctx(n: int) -> int | None:
    """
    s(n) = σ(n) - n via ctx. Returns None if σ(n) could not be computed.
    """
    sig = sigma_ctx(int(n))
    if sig is None:
        return None
    return sig - int(n)


def _sigma_from_fac_once(fac: dict[int, int]) -> int:
    s = 1
    for p, e in fac.items():
        s *= (p**(e+1)-1)//(p-1)
    return s


def _tau_from_fac(fac: dict[int, int]) -> int:
    t = 1
    for _, e in fac.items():
        t *= (e+1)
    return t


def _unitary_sigma_from_fac(fac: dict[int, int]) -> int:
    u = 1
    for p, e in fac.items():
        u *= (1 + p**e)
    return u


def has_composite_base(fac: dict[int, int]) -> bool:
    # factorint may leave composite cofactors as bases when limit is hit
    return any(p > 1 and not isprime(p) for p in fac)


def _composite_bases(fac: dict[int, int]) -> tuple[int, ...]:
    return tuple(p for p in fac if p > 1 and not isprime(p))


@cache
def _known_primes_from_bfile(filename: str) -> set[int]:
    found, idx_file, series, idx_set = check_oeis_bfile(filename, 0)  # target unused
    return set(series or ())


_MERSENNE_KEY_RE = re.compile(r"^2\*\*(\d+)-1$")


@cache
def _known_mersenne_prime_exponents() -> set[int]:
    """
    Return a cached set of exponents p for which 2^p − 1 is a *known* Mersenne prime.

    Sources:
      - b000668.txt  (OEIS b-file for Mersenne primes as integers; typically only the small ones)
      - special_inputs.tsv (tab-delimited; contains exact expressions like 2**p-1 for huge known primes)
    """
    exps: set[int] = set()

    # --- from b000668.txt ---------------------------------------------------
    # If present, parse the listed Mersenne primes and convert each value to its exponent p.
    try:
        for v in _known_primes_from_bfile("b000668.txt"):
            p = _mersenne_exponent_if_exact(v)
            if p is not None:
                exps.add(p)
    except Exception:
        # Missing/corrupt file → just skip this source
        pass

    # --- from special_inputs.tsv --------------------------------------------
    # file is tab-delimited and has a commented header starting with '#'.
    try:
        txt = _read_text_from_workspace_or_package("special_inputs.tsv")
    except Exception:
        txt = ""
    _EXPECTED_COLS = 2
    if txt:
        lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if lines:
            reader = csv.reader(io.StringIO("\n".join(lines)), delimiter="\t")
            for row in reader:
                if len(row) < _EXPECTED_COLS:
                    continue
                key = (row[0] or "").strip()
                handler = (row[1] or "").strip()

                # Only treat exact-known-prime handlers as proofs of primality.
                # (Adjust if you really use other handler names.)
                if handler not in ("mersenne_exact", "mersenne"):
                    continue

                m = _MERSENNE_KEY_RE.match(key)  # e.g. "2**82589933-1"
                if not m:
                    continue

                exps.add(int(m.group(1)))

    return exps


def _prime_shortcut(n: int) -> bool | None:
    p = _mersenne_exponent_if_exact(n)
    if p is not None and p in _known_mersenne_prime_exponents():
        return True
    # Ramanujan primes from b104272.txt (safe, but smaller win)
    if n in _known_primes_from_bfile("b104272.txt"):
        return True

    return None


def build_ctx(n: int) -> NumCtx:
    sigma = tau = unitary_sigma = unitary_tau = omega = None
    fac: dict[int, int] = {}
    comps: tuple[int, ...] = ()
    incomplete = False

    if n != 0:
        decdigits = dec_digits(n)

        stats_digit_limit = int(CFG("FACTORING.DIGIT_CAP", 5_000))

        # Too big for general σ/τ in stats: don't even try factoring.
        if decdigits > stats_digit_limit:
            incomplete = True
        else:
            # Let the factoring pipeline apply BITLEN_CAP, DIGIT_CAP, timeouts, etc.
            fac = _factor(n)
            # If factorization left composite bases, treat as incomplete
            if has_composite_base(fac):
                incomplete = True
                comps = _composite_bases(fac)

    if n == 0:
        pass
    elif n == 1:
        sigma = tau = unitary_sigma = unitary_tau = 1
        omega = 0
    elif not incomplete:
        tau = _tau_from_fac(fac)
        sigma = sigma_from_fac(fac)
        unitary_sigma = _unitary_sigma_from_fac(fac)
        omega = len(fac)
        unitary_tau = 1 << omega
    # else: leave divisor-derived fields as None

    return NumCtx(
        n=n,
        fac=fac,
        sigma=sigma,
        tau=tau,
        unitary_sigma=unitary_sigma,
        unitary_tau=unitary_tau,
        omega=omega,
        incomplete=incomplete,
        composite_bases=comps,
        is_prime=None
    )


@lru_cache(maxsize=128)
def _isprime_lru(n: int) -> bool:
    """Process-wide cache for primality of n."""
    return isprime(int(n))


def ctx_isprime(n: int, ctx: NumCtx | None = None) -> bool:
    """
    Prime test that:
      - reuses ctx.is_prime if present,
      - otherwise tries cheap proven shortcuts (e.g. known Mersenne primes),
      - otherwise uses a small LRU cache (SymPy isprime),
      - stores result back into ctx.
    """
    if ctx is not None and ctx.n == n and ctx.is_prime is not None:
        return ctx.is_prime

    # --- Cheap shortcut layer (proven primes only) ---
    sc = _prime_shortcut(int(n))
    if sc is True:
        if ctx is not None and ctx.n == n:
            object.__setattr__(ctx, "is_prime", True)
        return True

    # Fall back to SymPy primality test
    print("  Prime:                ⏳ Testing primality", end="\r", flush=True)
    result = _isprime_lru(n)

    if ctx is not None and ctx.n == n:
        object.__setattr__(ctx, "is_prime", result)

    return result


def cycle_bounds_from_seq(seq: list[int]) -> tuple[int, int] | None:
    """Return (start, end) indices [a, b) of first detected cycle in seq, else None."""
    pos = {}
    for i, v in enumerate(seq):
        if v in pos:
            return pos[v], i
        pos[v] = i
    return None


def enumerate_divisors_limited(fac: dict[int, int], limit: int | None = None) -> tuple[list[int], bool]:
    """
    Enumerate divisors from a factorization; stop once `limit` items collected.
    Returns (divisors_sorted, truncated_flag).
    """
    ds = [1]
    for p, e in fac.items():
        new = []
        pe = 1
        for _ in range(e + 1):
            for d in ds:
                new.append(d * pe)
                if limit is not None and len(new) >= limit * len(ds):
                    # not exact, but keeps work bounded; final sort handles order
                    pass
            pe *= p
        ds = new
        if limit is not None and len(ds) > limit * 4:
            # keep list from exploding; we'll trim/sort at the end
            ds = ds[:limit * 4]
    ds = sorted(set(ds))
    truncated = False
    if limit is not None and len(ds) > limit:
        ds = ds[:limit]
        truncated = True
    return ds, truncated


def _mersenne_exponent_if_exact(n: int) -> int | None:
    m = n + 1
    if m <= 0:
        return None
    if m & (m - 1):
        return None
    return m.bit_length() - 1


def base_name(a: int) -> str:
    """Return string like 'binary (base 2)' for a given base integer."""
    NAMES = {
        2: "binary",
        3: "ternary",
        4: "quaternary",
        5: "quinary",
        6: "senary",
        7: "septenary",
        8: "octal",
        9: "nonary",
        10: "decimal",
    }
    return f"{NAMES.get(a, f'base {a}')}" + f" (base {a})"


def _lambda_prime_power(p: int, a: int) -> int:
    """Carmichael λ for a prime power p^a."""
    # --- Carmichael λ constants (prime-power rules) ---
    _PRIME_TWO = 2           # special-case prime
    _EXPONENT_1 = 1          # p^1
    _EXPONENT_2 = 2          # p^2
    _SHIFT_BASE = 2          # 2 ** (a - 2) for p = 2, a >= 3

    if p == _PRIME_TWO:
        if a == _EXPONENT_1:
            return 1
        if a == _EXPONENT_2:
            return 2
        # a >= 3
        return _SHIFT_BASE ** (a - _EXPONENT_2)
    # odd prime: λ(p^a) = φ(p^a) = (p-1) * p**(a-1)
    return (p - 1) * (p ** (a - 1))


def carmichael_with_details(n: int, factors: dict[int, int]) -> tuple[int, str]:
    """
    Return Carmichael's λ(n) and a compact explanation string.

    Example: λ(2^2)=2, λ(3)=2, λ(5)=4 ⇒ lcm(2,2,4)=4
    """
    term_values: list[int] = []
    term_texts: list[str] = []

    lam = 1
    for p, a in sorted(factors.items()):
        lam_pa = _lambda_prime_power(p, a)
        term_values.append(lam_pa)
        term_texts.append(f"λ({p}^{a})={lam_pa}" if a > 1 else f"λ({p})={lam_pa}")
        lam = lcm(lam, lam_pa)

    details = f"{', '.join(term_texts)} ⇒ lcm({', '.join(map(str, term_values))})={lam}"
    return lam, details


def compute_aliquot_sequence(  # noqa: PLR0913
    n: int,
    *,
    kind: str = "aliquot",             # 'aliquot' | 'quasi' | 'unitary'
    max_steps: int = 200,
    step_timeout: float | None = 0.3,
    total_timeout: float | None = None,
    max_peak_digits: int | None = None,
    om=None,
    ctx0: NumCtx | None = None,        # NEW: optional context for starting n
):
    """
    Unified engine for aliquot-like sequences.

    ctx0 : NumCtx | None, default None
        Optional NumCtx for the starting value n. When provided, the engine
        reuses ctx0.sigma / ctx0.unitary_sigma for the *first* step instead
        of recomputing σ(n) or σ*(n). This avoids a second expensive divisor
        computation in print_statistics, where ctx has already been built.
    """
    kind = str(kind).lower().strip()
    if kind not in {"aliquot", "quasi", "unitary"}:
        raise ValueError(f"unknown kind: {kind}")

    # Determine label/value tag for progress line
    if kind == "aliquot":
        prefix, value_tag = "  Aliquot sequence:     ", "aliquot sum"
    elif kind == "quasi":
        prefix, value_tag = "  Quasi-aliquot seq.:   ", "q(n)"
    else:
        prefix, value_tag = "  Unitary aliquot seq.: ", "s*(n)"

    ui_enabled = CFG("DISPLAY_SETTINGS.SHOW_ALIQUOT_PROGRESS", True) and (om is not None)

    rt = _rt_current()
    fast_mode = bool(getattr(rt, "fast_mode", False))

    # Effective per-step timeout (only in fast_mode)
    eff_step_timeout = float(step_timeout) if fast_mode and step_timeout is not None else None

    # Per-sequence hard timeout (0 or None => disabled)
    eff_total_timeout = float(total_timeout) if total_timeout is not None and total_timeout > 0 else None

    # Peak-digit cap (0 or None => disabled)
    eff_max_peak_digits = int(max_peak_digits) if max_peak_digits is not None and max_peak_digits > 0 else None

    t_start = perf_counter()
    cur = int(n)

    seq = [cur]
    seen_pos = {cur: 0}
    highlight_idx = None
    skipped: str | bool = False
    aborted = False

    peak_value = abs(cur)
    peak_step = 0  # 0-based index

    try:
        for step in range(max_steps):
            # Global per-sequence time guard
            elapsed_seq = perf_counter() - t_start
            if (eff_total_timeout is not None) and (elapsed_seq > eff_total_timeout):
                aborted = True
                skipped = "sequence time exceeded guard"
                if ui_enabled:
                    om.clear_line()
                break

            # Global peak-digit guard
            if (eff_max_peak_digits is not None) and (dec_digits(cur) > eff_max_peak_digits):
                aborted = True
                skipped = "peak digit cap exceeded"
                if ui_enabled:
                    om.clear_line()
                break

            t0 = perf_counter()

            # --- NEXT TERM ------------------------------------------------
            # For the *first* step and matching ctx0, reuse precomputed σ / σ*.
            use_ctx0 = (
                step == 0
                and ctx0 is not None
                and getattr(ctx0, "n", None) == cur
            )

            if kind in {"aliquot", "quasi"}:
                sig_val = ctx0.sigma if use_ctx0 and ctx0.sigma is not None and not ctx0.incomplete else fast_sigma(cur)
                nxt = sig_val - abs(cur)
                if kind == "quasi":
                    nxt -= 1
            else:  # unitary
                if use_ctx0 and ctx0.unitary_sigma is not None and not ctx0.incomplete:
                    usig_val = ctx0.unitary_sigma
                else:
                    usig_val = fast_usigma(cur)
                nxt = usig_val - abs(cur)

            t1 = perf_counter()

            # ▲/▼ indicator relative to current term
            arrow = "▲" if nxt > cur else ("▼" if nxt < cur else "")

            if ui_enabled:
                len_str = len(str(nxt))
                msg = (
                    f"\r{prefix}{Fore.CYAN}"
                    f"Step {step+1}/{max_steps}, {value_tag} "
                    f"{Fore.YELLOW}{Style.BRIGHT}{arrow} ({len_str})"
                    f"{Fore.CYAN}{Style.NORMAL} {nxt}          "
                    f"{Style.RESET_ALL}"
                )
                om.write_screen(msg, end="", flush=True)

            # Per-step time guard (disabled when fast_mode=False)
            if (eff_step_timeout is not None) and ((t1 - t0) > eff_step_timeout):
                skipped = "step time exceeded guard"
                if ui_enabled:
                    om.clear_line()
                break

            # Record term and track peak
            nxt = int(nxt)
            seq.append(nxt)
            if abs(nxt) > peak_value:
                peak_value = abs(nxt)
                peak_step = len(seq) - 1

            # Terminals / cycle detection
            if nxt == 0:
                break
            if nxt in seen_pos:
                start = seen_pos[nxt]
                end = len(seq) - 1
                highlight_idx = (start, end)
                break

            seen_pos[nxt] = len(seq) - 1
            cur = nxt
        else:
            # Exhausted max_steps without hitting a terminal or cycle
            aborted = True
            if not skipped:
                skipped = "max steps reached"
            if ui_enabled:
                om.clear_line()

    except KeyboardInterrupt:
        if ui_enabled:
            om.clear_line()
        elapsed_s = perf_counter() - t_start
        stats = {
            "steps": 0,
            "peak_value": seq[0] if seq else n,
            "peak_step": 0,
            "elapsed_s": elapsed_s,
        }
        return [], None, False, "Aborted by user (Ctrl-C)", stats

    except OverflowError:
        if ui_enabled:
            om.clear_line()
        elapsed_s = perf_counter() - t_start
        stats = {
            "steps": max(0, len(seq) - 1),
            "peak_value": peak_value,
            "peak_step": peak_step,
            "elapsed_s": elapsed_s,
        }
        return seq, highlight_idx, aborted, "overflow / value too large in σ(n)", stats

    if ui_enabled:
        om.clear_line()
    elapsed_s = perf_counter() - t_start
    stats = {
        "steps": max(0, len(seq) - 1),
        "peak_value": peak_value,
        "peak_step": peak_step,
        "elapsed_s": elapsed_s,
    }

    return seq, highlight_idx, aborted, skipped, stats


def digit_sum(n: int) -> int:
    """
    Calculate the sum of digits of n.
    Args: n (int): The number.

    Returns: int: The sum of the absolute value digits.
    """
    return sum(int(d) for d in str(abs(n)))


def digit_product(n: int) -> int:
    """
    Return the product of the digits of n (absolute value).
    """
    digits = [int(d) for d in str(abs(n))]
    prod = 1
    for d in digits:
        prod *= d
    return prod


def digital_root_sequence(n: int):
    """
    Yield the digital root sequence for n, repeatedly summing its decimal
    digits until a single-digit is reached, including the starting value.
    """
    _ONE_DIGIT = 9
    seq = [abs(n)]
    while seq[-1] > _ONE_DIGIT:
        seq.append(sum(int(d) for d in str(seq[-1])))
    return seq


def format_prime_factors(n: int, *,
                         factors: dict[int, int] | None = None,
                         return_factors: bool = False):
    """
    Format the prime factorization of n as a string.
    If `factors` is provided, it is used (no refactor). If `return_factors`
    is True, also return the factor dict that was used.

    Examples:
    >>> format_prime_factors(60)
    '2^2 × 3 × 5'
    >>> s, facs = format_prime_factors(60, return_factors=True)
    >>> s, facs = ('2^2 × 3 × 5', {2: 2, 3: 1, 5: 1})
    """
    if n == 0:
        return ("0", {}) if return_factors else "0"
    if abs(n) == 1:
        return (str(n), {}) if return_factors else str(n)

    if factors is None:
        factors = factorint(abs(n))

    parts = []
    for p in sorted(factors):
        exp = factors[p]
        parts.append(f"{p}^{exp}" if exp > 1 else f"{p}")

    result = " × ".join(parts)
    if n < 0:
        result = f"-1 × {result}"

    return (result, factors) if return_factors else result


def get_terminal_width(default=80):
    """
    Return the terminal's character width if detected, else the default
    value (80 by default).
    """
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


def get_terminal_height(default=24):
    """
    Get the number of terminal lines.
    """
    try:
        return shutil.get_terminal_size().lines
    except Exception:
        return default


def get_ordinal_suffix(n: int) -> str:
    """
    Return the ordinal representation of an integer (e.g., 1st, 2nd, 3rd, 4th...).
    """
    # Ordinal suffix rules
    _MOD_100 = 100
    _MOD_10 = 10
    _TEENS_START = 11
    _TEENS_END = 13
    _DEFAULT_SUFFIX = "th"
    _LAST_DIGIT_SUFFIX = {1: "st", 2: "nd", 3: "rd"}

    last_two = n % _MOD_100
    suffix = (
        _DEFAULT_SUFFIX
        if _TEENS_START <= last_two <= _TEENS_END
        else _LAST_DIGIT_SUFFIX.get(n % _MOD_10, _DEFAULT_SUFFIX)
    )
    return f"{n}{suffix}"


@cache
def _load_oeis_series(name: str) -> list[int]:
    """
    Cached loader for OEIS b-files.
    Returns the list of values (series) in order.
    """
    name = Path(name).name
    up = Path(workspace_dir()) / "data" / name
    if up.is_file():
        text = up.read_text(encoding="utf-8")
    else:
        try:
            text = pkg_files("numclass.data").joinpath(name).read_text(encoding="utf-8")
        except FileNotFoundError:
            return []

    values: list[int] = []
    _EXPECTED_COLS = 2
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
        parts = re.split(r"[,\s]+", line)
        if len(parts) < _EXPECTED_COLS:
            continue
        try:
            val = int(parts[1])
        except ValueError:
            continue
        values.append(val)
    return values


def check_oeis_bfile(name: str, target: int) -> tuple[bool, int | None, list[int], int | None]:
    """
    Returns: (found, index_file, series, index_set)
      - index_file: the FIRST COLUMN from the b-file (real OEIS term index, may be 0-based for Bell!)
      - index_set : 0-based position in 'series' (only for neighbors/fallback)
    """
    name = Path(name).name
    # load text from workspace or package (same as you already do)...
    text = None
    up = Path(workspace_dir()) / "data" / name
    if up.is_file():
        text = up.read_text(encoding="utf-8")
    else:
        try:
            text = pkg_files("numclass.data").joinpath(name).read_text(encoding="utf-8")
        except FileNotFoundError:
            return False, None, [], None

    series: list[int] = []
    found = False
    index_file: int | None = None
    index_set: int | None = None
    _EXPECTED_COLS = 2
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

        parts = re.split(r"[,\s]+", line)
        if len(parts) < _EXPECTED_COLS:
            continue

        # Parse both columns; column 1 is the true OEIS index
        try:
            i = int(parts[0])            # <-- file index (may be 0-based)
            v = int(parts[1])            # <-- value
        except ValueError:
            continue

        series.append(v)

        if not found and v == target:
            found = True
            index_file = i               # <-- keep the FILE INDEX, not list pos
            index_set = len(series) - 1  # 0-based position in series

    return found, index_file, series, index_set


def int_to_words(n: int) -> str:
    """
    Convert an integer to British English words.
    Examples:
      42 -> "forty-two"
      101 -> "one hundred and one"
      1205 -> "one thousand two hundred and five"
      -17 -> "minus seventeen"
    """
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + int_to_words(-n)

    ones = [
        "", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    ]
    tens = [
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety"
    ]
    scales = [
        "", "thousand", "million", "billion", "trillion",
        "quadrillion", "quintillion", "sextillion", "septillion",
        "octillion", "nonillion", "decillion", "undecillion",
        "duodecillion", "tredecillion", "quattuordecillion",
        "quindecillion", "sexdecillion", "septendecillion",
        "octodecillion", "novemdecillion", "vigintillion"
    ]

    def chunk_to_words(num: int) -> str:
        """Convert a number less than 1000 to words."""
        words = []
        hundreds, remainder = divmod(num, 100)
        _TEENS_LIMIT = 20
        if hundreds:
            words.append(ones[hundreds] + " hundred")
            if remainder:
                words.append("and")
        if remainder:
            if remainder < _TEENS_LIMIT:
                words.append(ones[remainder])
            else:
                t, o = divmod(remainder, 10)
                words.append(tens[t])
                if o:
                    words[-1] += "-" + ones[o]
        return " ".join(w for w in words if w)

    # Split number into groups of three digits
    words = []
    group_index = 0
    while n > 0:
        n, chunk = divmod(n, 1000)
        if chunk:
            chunk_words = chunk_to_words(chunk)
            scale = scales[group_index]
            if scale:
                words.insert(0, f"{chunk_words} {scale}")
            else:
                words.insert(0, chunk_words)
        group_index += 1

    return " ".join(words).strip()


def mobius_and_radical(factors: dict[int, int]) -> tuple[int, int, bool]:
    """
    Return (μ(n), rad(n), squarefree) from a prime factor dict.
    μ(n)=0 if any exponent>1; else (-1)^ω(n).  rad(n)=∏p.
    """
    rad = 1
    squarefree = True
    for p, e in factors.items():
        rad *= p
        if e > 1:
            squarefree = False
    mu = 0 if not squarefree else (-1 if (len(factors) % 2) else 1)
    return mu, rad, squarefree


def multiplicative_persistence_sequence(n: int) -> list[int]:
    """
    Return the multiplicative persistence sequence for |n|.
    Repeatedly replace x with the product of its decimal digits until x < 10.
    Example: 39 → [39, 27, 14, 4] (persistence = 3)
    """
    x = abs(n)
    seq = [x]
    _ONE_DIGIT = 9
    while x > _ONE_DIGIT:
        prod = 1
        for ch in str(x):
            prod *= int(ch)
        x = prod
        seq.append(x)
    return seq


def parity(n: int):
    """
    Returns the parity (even/odd of a number)
    """
    if n % 2 == 0:
        return "Even"
    return "Odd"


# --- Number-stat helpers (factorization-aware) ---


def sigma_term(p: int, e: int) -> int:
    return (p**(e + 1) - 1) // (p - 1)


def sigma_from_fac(fac: dict[int, int]) -> int:
    s = 1
    for p, e in fac.items():
        s *= sigma_term(p, e)
    return s


def sigma_with_parts(fac: dict[int, int]) -> tuple[int, list[tuple[int, int, int]]]:
    sigma = 1
    parts = []
    for p, e in sorted(fac.items()):
        t = sigma_term(p, e)
        parts.append((p, e, t))
        sigma *= t
    return sigma, parts


def _sigma_from_factors(fac: dict[int, int]) -> int:
    """σ(n) = ∏ (p^(a+1) − 1)/(p − 1) using gmpy2 bigints."""
    acc = gmpy2.mpz(1)
    for p, a in fac.items():
        pz = gmpy2.mpz(p)
        num = pow(pz, a + 1) - 1        # p^(a+1) - 1
        den = pz - 1
        acc *= num // den
    return int(acc)


def _usigma_from_factors(fac: dict[int, int]) -> int:
    """σ*(n) = ∏ (1 + p^a) using gmpy2 bigints."""
    acc = gmpy2.mpz(1)
    for p, a in fac.items():
        acc *= 1 + pow(gmpy2.mpz(p), a)  # p^a
    return int(acc)


@lru_cache(maxsize=100_000)
def fast_sigma(n: int) -> int:
    """
    Fast σ(n) for aliquot sequences:

    - Uses hard_factors.txt if n is listed (instant σ),
    - Otherwise falls back to SymPy factorint,
    - Results are cached to avoid refactoring the same n.
    """
    n = abs(int(n))
    if n <= 1:
        return 1

    # Try hard_factors.txt first
    hard = _lookup_hard_factor(n)
    if hard is not None:
        return _sigma_from_factors(hard)

    # Fall back to plain factorint (no multiprocessing here)
    fac = factorint(n)  # faster with gmpy2 present
    return _sigma_from_factors(fac)


def fast_usigma(n: int) -> int:
    """
    Fast unitary sigma σ*(n):

    - check hard_factors.txt first (instant),
    - fallback to factorint for general case,
    - results cached via lru_cache.
    """
    n = abs(int(n))
    if n <= 1:
        return 1

    # 1) Check hard_factors first
    hard = _lookup_hard_factor(n)
    if hard is not None:
        return _usigma_from_factors(hard)

    # 2) Fallback: factor normally
    fac = factorint(n)
    return _usigma_from_factors(fac)


def totient_with_details(n: int, factors: dict[int, int]):
    """
    Return φ(n) and a compact explanation string.
    Example: φ(2^2)=2, φ(3)=2, φ(5)=4 ⇒ 2*2*4=16
    """
    if n == 1:
        return 1, "φ(1)=1"

    parts = []
    phi = 1
    mults = []  # for the "2*2*4" bit

    for p, a in sorted(factors.items()):
        # φ(p^a) = p^(a-1) * (p-1)
        phi_pa = (p - 1) * (p ** (a - 1))
        parts.append(f"φ({p}^{a})={phi_pa}" if a > 1 else f"φ({p})={phi_pa}")
        mults.append(str(phi_pa))
        phi *= phi_pa

    return phi, f"{', '.join(parts)} ⇒ {'*'.join(mults)}={phi}"


def validate_output_setting(output_file: str | None) -> str | None:
    """
    Validate output setting.
    - None / "" => ok (screen only)
    - "." / "./" / trailing "/" => ok (per-number directory mode)
    - path/to/file => must not be in forbidden base names or extensions
    Returns the (possibly normalized) output_file, or raises ValueError.
    """
    FORBIDDEN_FILENAMES = {
        ".gitignore",
        "b002093.txt",
        "b004394.txt",
        "b005114.txt",
        "b104272.txt",
        "LICENSE",
        "requirements.txt",
        # Windows reserved device names (case-insensitive on Windows)
        "con", "prn", "aux", "nul",
        "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
        "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
    }

    FORBIDDEN_EXTENSIONS = {".py", ".md"}

    if not output_file:
        return output_file  # screen only

    # directory/per-number modes are allowed as-is
    if output_file in (".", "./") or output_file.endswith("/"):
        return output_file

    # single file mode: check basename & extension
    basename = os.path.basename(output_file)
    name_no_ext, ext = os.path.splitext(basename)
    ext = ext.lower()

    # On Windows, device names are forbidden regardless of extension
    base_lower = basename.lower()
    name_lower = name_no_ext.lower()
    if base_lower in FORBIDDEN_FILENAMES or name_lower in FORBIDDEN_FILENAMES:
        raise ValueError(f"Forbidden output filename: {basename}")

    if ext in FORBIDDEN_EXTENSIONS:
        raise ValueError(f"Forbidden output file extension: {ext}")

    return output_file


def typename(v: object) -> str:
    return type(v).__name__


def flatten_dotted(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dotted(v, key))
        else:
            out[key] = v
    return out


# --- Fibonacci / Zeckendorf -------------------------------------------------

@lru_cache(maxsize=50_000)
def fib_upto(n: int, *, f0: int = 0, f1: int = 1) -> list[int]:
    """
    Return Fibonacci numbers [F0, F1, ...] up to the largest <= n.
    Defaults to (0,1). Assumes n >= 0.
    """
    if n <= 0:
        return [f0] if n == 0 else [f0]
    a, b = f0, f1
    out = [a, b]
    while b <= n:
        a, b = b, a + b
        out.append(b)
    # last appended may be > n (because loop condition is <=), so trim
    if out and out[-1] > n:
        out.pop()
    return out


def is_fibonacci_value(n: int) -> bool:
    # valid for n >= 0
    t = 5 * n * n
    return is_square(t + 4) or is_square(t - 4)


def fibonacci_index_if_member(n: int) -> int | None:
    """
    Return k such that F(k)=n (with F0=0,F1=1), else None.
    Uses the fast membership test first; then a tiny loop to find k.
    """
    if n < 0:
        return None
    if n == 0:
        return 0
    if n == 1:
        return 1  # (also F2=1, but pick the smallest index)

    if not is_fibonacci_value(n):
        return None

    a, b = 0, 1
    k = 1
    while b < n:
        a, b = b, a + b
        k += 1
    return k if b == n else None


def zeckendorf_decomposition(n: int) -> tuple[list[int], list[int], str]:
    """
    Zeckendorf decomposition for n >= 0 using Fibonacci numbers with F0=0,F1=1.
    Returns (terms, indices, bitstring) where:
      - terms are Fibonacci values used, descending
      - indices are their Fibonacci indices (matching F0=0,F1=1)
      - bitstring covers [F_k .. F_2] (common convention; excludes F0,F1)
    """
    if n < 0:
        raise ValueError("Zeckendorf requires n >= 0")
    if n == 0:
        return ([], [], "0")

    fibs = fib_upto(n)  # [F0..F_k] up to <= n
    # We will use indices >= 2 (Zeckendorf uniqueness normally stated with F2=1,F3=2,...)
    # but allowing F1=1 in generation doesn't hurt; we simply never need consecutive terms.
    used_terms: list[int] = []
    used_idx: list[int] = []

    rem = n
    i = len(fibs) - 1
    while rem > 0 and i >= 0:
        f = fibs[i]
        if f <= rem:
            used_terms.append(f)
            used_idx.append(i)
            rem -= f
            i -= 2  # skip the next Fibonacci down to avoid consecutive terms
        else:
            i -= 1

    # Build a compact fib-base bitstring from max used index down to 2
    k = max(used_idx)
    if k < 2:
        # n==1 edge case; represent as "1" (still fine for display)
        return (used_terms, used_idx, "1")

    used_set = set(used_idx)
    bits = []
    for j in range(k, 1, -1):  # k,k-1,...,2
        bits.append("1" if j in used_set else "0")
    return (used_terms, used_idx, "".join(bits))
