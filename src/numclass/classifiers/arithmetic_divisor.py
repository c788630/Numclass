# -----------------------------------------------------------------------------
#  Arithmetic_divisor.py
#  Arithmetic and Divisor-based test functions
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from array import array
from fractions import Fraction
from functools import cache, reduce
from itertools import pairwise
from math import gcd, isqrt

from sympy import factorint, integer_nthroot
from sympy.ntheory.primetest import is_square

from numclass.context import NumCtx
from numclass.fmt import (
    abbr_int_fast,
    format_factorization,
    format_sigma_terms,
)
from numclass.registry import classifier
from numclass.runtime import CFG
from numclass.runtime import current as _rt_current
from numclass.utility import (
    _divisors_from_factorization,
    _mersenne_exponent_if_exact,
    build_ctx,
    check_oeis_bfile,
    compute_aliquot_sequence,
    cycle_bounds_from_seq,
    dec_digits,
    enumerate_divisors_limited,
    fast_sigma,
    get_aliquot_cache,
    get_ordinal_suffix,
    proper_sum_ctx,
    sigma_ctx,
    sigma_from_fac,
    sigma_with_parts,
)

CATEGORY = "Arithmetic and Divisor-based"
ARROW = CFG("FORMATTING.SEQUENCE_ARROW", " → ")
MAX_DIVISORS_FOR_EXACT = 6000          # don't enumerate more than this many divisors


def _abundancy_frac(ctx: NumCtx) -> tuple[int, int]:
    """Return σ(n)/n in lowest terms as (num, den)."""
    a, b = ctx.sigma, ctx.n
    g = gcd(a, b)
    return a // g, b // g


def _sigma_parts(fac):
    """
    Return (sigma, parts_for_fmt) where parts_for_fmt is a list of
    (symbolic_term, numeric_value) pairs suitable for fmt.format_sigma_terms().
    Accepts:
      - (sigma, [(p,e,t), ...])            # from utility.sigma_with_parts()
      - (sigma, [(symbolic_term,t), ...])  # already normalized
      - sigma only                         # then parts_for_fmt=None
    """
    res = sigma_with_parts(fac)
    if isinstance(res, tuple):
        if len(res) == 2:
            sigma, parts = res
            parts_for_fmt = []
            for item in parts or []:
                if isinstance(item, tuple) and len(item) == 3:
                    p, e, t = item
                    # Build "(1 + p + p^2 + ... + p^e)"
                    if e >= 1:
                        syms = ["1"] + [f"{p}^{k}" if k > 1 else f"{p}" for k in range(1, e + 1)]
                        sym = "(" + "+".join(syms) + ")"
                    else:
                        sym = "1"
                    parts_for_fmt.append((sym, t))
                elif isinstance(item, tuple) and len(item) == 2:
                    parts_for_fmt.append(item)
            return sigma, parts_for_fmt if parts_for_fmt else None
        elif len(res) == 1:
            return res[0], None

    # Fallback: compute sigma and omit parts
    try:
        sig = sigma_from_fac(fac)
    except Exception:
        sig = 1
        for p, e in fac.items():
            sig *= (p ** (e + 1) - 1) // (p - 1)
    return sig, None


def _is_k_free(n: int, k: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    k-free ⇔ no prime exponent ≥ k. Returns (True, detail) or (False, None).
    """
    if n in (0, 1):
        # by convention: 0 is not k-free; 1 has empty factorization and is k-free
        details = "n = 1 has no prime factors ⇒ k-free" if n == 1 else None
        return (n == 1, details)
    if n < 0:
        n = -n  # use |n|
    ctx = ctx or build_ctx(n)
    if not ctx.fac:
        # prime n ⇒ all exponents = 1 < k
        details = f"n is prime ⇒ all exponents = 1 < {k}"
        return True, details
    e_max = max(ctx.fac.values())
    if e_max < k:
        # concise, crash-proof detail
        facts = " × ".join(f"{abbr_int_fast(p)}^{e}" if e > 1 else f"{abbr_int_fast(p)}" for p, e in sorted(ctx.fac.items()))
        details = f"n = {facts}; max exponent = {e_max} < {k}"
        return True, details
    return False, None


def _is_k_full(n: int, k: int, ctx: NumCtx | None) -> tuple[bool, str | None]:
    if n == 0:
        return False, None  # undefined / excluded

    ctx = ctx or build_ctx(n)

    # NEW: honour incomplete factorisation like other ctx-based classifiers
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    if ctx.n == 1:
        # 1 has empty factorization; vacuously k-full for all k
        return True, f"n=1; no prime factors ⇒ exponents ≥ {k} vacuously true"

    if not ctx.fac:
        # Shouldn't really happen if ctx.incomplete handled, but be safe:
        # treat as "can't certify", so just say not k-full.
        return False, None

    exps = [e for _, e in sorted(ctx.fac.items())]
    e_min = min(exps)
    fac_str = format_factorization(ctx.fac)
    if e_min >= k:
        return True, f"n = {fac_str}; min exponent = {e_min} ≥ {k}"
    return False, None


def _is_k_perfect(n: int, k: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n <= 0:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.sigma != k * n:
        return False, None

    sigma, parts = _sigma_parts(ctx.fac)
    fac_str = format_factorization(ctx.fac)
    if parts is not None:
        syms, vals = format_sigma_terms(parts)
        return True, f"n = {fac_str};  σ(n) = {syms} = {vals} = {sigma} = {k}×{n}"
    else:
        return True, f"n = {fac_str};  σ(n) = {sigma} = {k}×{n}"


def _is_weird_quick(n: int, ctx: NumCtx | None = None) -> bool | None:
    """Abundant and not semiperfect; returns None if undecided (too big)."""
    MAX_DIVISORS_LIST = 2000   # enumerate at most this many for checks
    MAX_TAU_CHECK = 8000       # skip if too many divisors
    ctx = ctx or build_ctx(n)

    if ctx.incomplete or ctx.sigma is None or ctx.tau is None:
        return None

    # abundant gate
    if ctx.sigma <= 2 * n:
        return False
    # need proper divisors for semiperfect test
    if ctx.tau > MAX_TAU_CHECK:
        return None
    # enumerate divisors (bounded)
    divs, trunc = enumerate_divisors_limited(ctx.fac, limit=MAX_DIVISORS_LIST)
    if trunc:
        return None
    proper = [d for d in divs if d < n]
    # subset-sum to n (bitset)
    target = n
    bits = 1
    for d in proper:
        if d > target:
            continue
        bits |= (bits << d)
        if (bits >> target) & 1:
            return False  # semiperfect
        if bits.bit_length() > target + 1:
            bits &= (1 << (target + 1)) - 1
    return True


def _phi_from_fac(n: int) -> int:
    ctx = build_ctx(n)
    phi = n
    for p in ctx.fac:
        phi = (phi // p) * (p - 1)
    return phi


def _tau_from_factors(fac: dict[int, int]) -> tuple[int, str]:
    """Return (tau, pretty_terms) where tau = ∏(e_i+1) and pretty_terms is '(e1+1)×(e2+1)…'."""
    tau = 1
    terms = []
    for _p, e in sorted(fac.items()):
        tau *= (e + 1)
        terms.append(f"({e}+1)")
    return tau, "×".join(terms) if terms else "1"  # τ(1)=1


# --- tiny sigma/proper-divisor-sum helpers (factorization-based, cached) ---

@cache
def _fac(n: int):
    return factorint(abs(n))


def _proper_sum(n: int) -> int:
    if n <= 1:
        return 0
    fac = _fac(n)
    return sigma_from_fac(fac) - n


def q_next(n: int) -> int:
    """Quasi-aliquot map q(n) = s(n) - 1."""
    return _proper_sum(n) - 1


def is_perfect(m: int) -> bool:
    return m > 1 and _proper_sum(m) == m


def q_sequence(n: int, max_steps: int = 2000):
    """Return (seq list, cycle_start_index or None). Ends if repeats or hits <=0."""
    seen = {}
    seq = [n]
    cur = n
    for _step in range(max_steps):
        if cur in seen:
            return seq, seen[cur]  # cycle starts here
        seen[cur] = len(seq) - 1
        nxt = q_next(cur)
        seq.append(nxt)
        if nxt <= 0:
            return seq, None
        cur = nxt
    return seq, None  # gave up (budget)


# --- classifiers ---


@classifier(
    label="Abundant number",
    description="Proper divisors sum > the given number.",
    oeis="A005101",
    category=CATEGORY
)
def is_abundant_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check if n is an abundant number.
    An abundant number is a number for which the sum of its proper divisors
    exceeds the number itself.
    """
    if n < 12:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    aliquot = ctx.sigma - n
    if aliquot > n:
        return True, f"Sum of proper divisors: {abbr_int_fast(aliquot)} > {abbr_int_fast(n)}."
    return False, None


@classifier(
    label="Achilles number",
    description="A powerful number that is not a perfect power.",
    oeis="A052486",
    category=CATEGORY
)
def is_achilles_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Returns True if n is an Achilles number, else False.
    An Achilles number is powerful but not a perfect power.
    """
    if n < 72:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")
    # Factorization: {prime: exponent}
    factors = ctx.fac
    exponents = list(factors.values())

    # Must be powerful: all exponents >= 2
    if any(exp < 2 for exp in exponents):
        return False, None

    # Not a perfect power: GCD of exponents == 1 means not a perfect power
    ex_gcd = reduce(gcd, exponents)
    if ex_gcd > 1:
        return False, None  # It's a perfect power

    # Build details string for explanation
    prime_factors = ' × '.join([f"{p}^{e}" for p, e in factors.items()])
    detail = f"{n} = {prime_factors} (gcd of exponents = {ex_gcd})"
    return True, detail


@classifier(
    label="Almost perfect number",
    description="σ(n)=2n−1. Known examples: 1 and all powers of 2; none others known.",
    oeis="A000079",
    category="Arithmetic and Divisor-based",
)
def is_almost_perfect(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    if n <= 0:
        return False, None

    # n=1: trivial almost-perfect
    if n == 1:
        return True, "σ(1)=1=2×1−1"

    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    # Fast path: power of two
    if n & (n - 1) == 0:
        # n = 2^k
        k = n.bit_length() - 1
        # σ(2^k) = 2^{k+1}−1 = 2n−1
        return True, f"n = 2^{k}: σ(2^{k}) = 2^{k+1}−1 = {2*n-1} = 2n−1"

    if ctx.sigma == 2 * n - 1:
        return True, f"σ({n}) = {ctx.sigma} = 2n−1"

    return False, None


@classifier(
    label="Amicable number",
    description="Proper divisors sum to m, whose divisors sum to n.",
    oeis="A063990",
    category=CATEGORY,
    limit=10**7-1
)
def is_amicable_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n < 220:
        return False, None

    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    a = ctx.sigma - n  # s(n)

    # parity quick-check
    if (n ^ a) & 1:
        return False, None
    if a == n:
        return False, None  # exclude perfect numbers

    s_a = proper_sum_ctx(a)
    if s_a is None:
        return False, None  # couldn’t certify within limits

    if s_a == n:
        return True, (
            f"{n} and {a} are an amicable pair: "
            f"σ({n})−{n}={a}, and σ({a})−{a}={n}."
        )
    return False, None


@classifier(
    label="Aspiring number",
    description="Aliquot sequence hits a perfect number (n itself not perfect).",
    oeis="A063769",
    category="Arithmetic and Divisor-based",
)
def is_aspiring_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    if n <= 0:
        return False, None

    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    max_steps = CFG("ALIQUOT.MAX_STEPS", 250)
    tlimit = CFG("ALIQUOT.STEP_TIME_LIMIT", 0.30)
    vlimit = CFG("ALIQUOT.VALUE_LIMIT", 10**7)

    entry = get_aliquot_cache(n, max_steps)

    if entry:
        seq = entry.seq or []
        # Perfect hit appears as ... → m → m
        for i, (a, b) in enumerate(pairwise(seq), start=0):
            if a == b and a > 0 and seq[0] != a:  # exclude perfect n itself
                return True, f"hits perfect {a} in {i + 1} steps"
        return False, None

    # 2) Fallback: minimal quiet computation (no printing)
    t0 = time.perf_counter()
    x = int(n)
    seen = {x}
    for step in range(1, int(max_steps) + 1):
        s = fast_sigma(x) - x
        if s > vlimit or (time.perf_counter() - t0) > tlimit:
            raise TimeoutError("aliquot guard exceeded")
        if s == x and s > 0:              # fixed point → perfect number
            if s != n:                    # exclude perfect n itself
                return True, f"hits perfect {s} in {step} steps"
            return False, None
        if s == 0 or s in seen:           # zero or cycle → not aspiring
            return False, None
        seen.add(s)
        x = s

    return False, None


@classifier(
    label="Deficient number",
    description="Proper divisors sum < the given number.",
    oeis="A005100",
    category=CATEGORY
)
def is_deficient_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check if n is a deficient number.
    A deficient number is a number for which the sum of its proper divisors
    is less than the number itself.
    """
    if n < 1:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")
    aliquot = ctx.sigma - n
    if aliquot < n:
        details = f"Sum of proper divisors: {abbr_int_fast(aliquot)} < {abbr_int_fast(n)}."
        return True, details
    return False, None


@classifier(
    label="Descartes number",
    description="Odd n with a coprime split n = m×q, q composite, and 2n = σ(m)(q+1).",
    oeis="A174292",
    category="Arithmetic and Divisor-based",
    limit=10**12-1,
)
def is_descartes_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """Detect a Descartes (spoof odd perfect) number with block-wise σ(m)."""
    if n <= 0 or (n & 1) == 0:
        return False, None  # must be odd

    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        # cannot certify
        return False, None

    fn = ctx.fac  # full prime-power factorization of n
    if not fn or n == 1:
        return False, None

    # Pre-extract whole blocks: (prime, p^e, e)
    blocks = [(p, p**e, e) for p, e in fn.items()]

    # Helper: given a list of blocks for m, compute σ(m) directly.
    def sigma_from_blocks(m_blocks):
        total = 1
        for p, _pe, e in m_blocks:
            # sigma(p^e) = (p^(e+1) - 1) // (p - 1)
            total *= (p**(e + 1) - 1) // (p - 1)
        return total

    best = None

    def dfs(i: int, q_val: int, m_blocks: list, blocks_used: int, exp_ge2: bool):
        nonlocal best
        if i == len(blocks):
            if q_val in (1, n):
                return

            # q composite?
            is_composite = (blocks_used >= 2) or exp_ge2
            if not is_composite:
                return

            m_val = n // q_val             # guaranteed integer
            two_n = 2 * n
            q1 = q_val + 1
            if two_n % q1 != 0:
                return
            target_sigma_m = two_n // q1

            sigma_m = sigma_from_blocks(m_blocks)
            if sigma_m == target_sigma_m:
                best = (
                    f"n = m×q with m={m_val}, q={q_val} (composite), "
                    f"and 2n = σ(m)×(q+1)"
                )
            return

        p, block, e = blocks[i]

        # Add block to q
        dfs(i + 1, q_val * block, m_blocks, blocks_used + 1, exp_ge2 or (e >= 2))

        # Add block to m
        dfs(i + 1, q_val, [*m_blocks, (p, block, e)], blocks_used, exp_ge2)

    dfs(0, 1, [], 0, False)

    return (True, best) if best else (False, None)


@classifier(
    label="Highly abundant number",
    description="An integer whose sum of divisors is greater than that of any smaller positive integer.",
    oeis="A002093",
    category=CATEGORY,
    limit=149918408641,
)
def is_highly_abundant_number(n: int) -> tuple[bool, str | None]:
    """
    Highly abundant numbers: σ(n) > σ(k) for all k < n.
    We rely on √OEIS b002093 for membership, and σ-values from ctx
    """
    if n < 1:
        return False, None

    # Check if n is in the OEIS list
    found, idx_file, series, idx_set = check_oeis_bfile("b002093.txt", n)
    if not found:
        return False, None

    # σ(n) via ctx
    sigma_n = sigma_ctx(n)
    if sigma_n is None:
        return False, None   # incomplete factorization or timeout

    # If this is NOT the first element, show the previous record
    if idx_set is not None and idx_set > 0:
        prev = series[idx_set - 1]
        sigma_prev = sigma_ctx(prev)
        if sigma_prev is None:
            return True, f"σ({n}) = {sigma_n}. Previous record: {prev} (σ unknown)."
        details = (
            f"σ({n}) = {sigma_n}. "
            f"Previous record: {prev} with σ({prev}) = {sigma_prev}."
        )
    else:
        details = f"{n} is the first highly abundant; σ({n}) = {sigma_n}."

    return True, details


@classifier(
    label="Highly composite number",
    description="(a.k.a. anti-prime) Has more divisors than any smaller positive integer.",
    oeis="A002182",
    category=CATEGORY,
    limit=6_385_128_751
)
def is_highly_composite_number(n: int) -> tuple[bool, str | None]:
    """
    A highly composite number has τ(n) > τ(m) for all 1 ≤ m < n.
    We recognize them via a finite list (fast), and provide a witness:
    τ(n) with factor-exponent formula, and the previous record-holder.
    """
    KNOWN = [
        1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260,
        1680, 2520, 5040, 7560, 10080, 15120, 20160, 27720, 45360, 50400,
        55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960,
        554400, 665280, 720720, 1081080, 1441440, 2162160, 2882880,
        3603600, 4324320, 6486480, 73513440, 147026880, 294053760,
        367567200, 698377680, 1396755360, 2095133040, 4190266080, 6385128750
    ]
    # Fast membership check
    if n not in KNOWN:
        return False, None

    # Find previous record-holder from the same list
    # (Sorting here keeps the code robust even if KNOWN turns into a set by accident.)
    ordered = sorted(KNOWN)
    idx = ordered.index(n)
    prev = ordered[idx - 1] if idx > 0 else None

    # Build a clean witness using the tau formula from the factorization
    if n == 1:
        return True, "τ(1)=1 (base case). No smaller positive integer."
    tau_n, terms_n = _tau_from_factors(factorint(n))
    tau_prev, _ = _tau_from_factors(factorint(prev))  # type: ignore[arg-type]

    details = (
        f"τ({n}) = {terms_n} = {tau_n}; "
        f"previous record τ({prev}) = {tau_prev}."
    )
    # (Optionally assert tau_n > tau_prev to catch list mistakes in tests)
    return True, details


@classifier(
    label="Mersenne number",
    description="Number of the form 2^p − 1 for some integer p ≥ 1.",
    oeis="A000225",
    category=CATEGORY,
)
def is_mersenne_number(n: int) -> tuple[bool, str | None]:
    if n <= 0:
        return False, None

    p = _mersenne_exponent_if_exact(n)
    if p is None:
        return False, None

    return True, f"{abbr_int_fast(n)} = 2^{p} − 1"


@classifier(
    label="Perfect cube",
    description="n = k^3 for some integer k.",
    oeis="A000578",
    category=CATEGORY
)
def is_perfect_cube(n: int) -> tuple[bool, str | None]:
    """
    Check if n is a perfect cube using exact integer cube root.
    Works for arbitrarily large |n| and negative cubes.
    """
    # 0 is a perfect cube (0^3)
    if n == 0:
        return True, "0 = 0³ (perfect cube)."

    a = abs(n)
    r, exact = integer_nthroot(a, 3)  # r = floor(a^(1/3)), exact if r**3 == a
    if exact:
        k = r if n >= 0 else -r
        return True, f"{n} = {k}³ (perfect cube)."
    return False, None


@classifier(
    label="Perfect number",
    description="A positive integer equal to the sum of its proper divisors.",
    oeis="A000396",
    category=CATEGORY,
)
def is_perfect_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Check if n is a perfect number.
    Example: 28 = 1 + 2 + 4 + 7 + 14
    """
    if n < 2:
        return False, None

    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    # Fast decision using σ(n)
    aliquot = ctx.sigma - n
    if aliquot != n:
        return False, None

    # At this point we know n is perfect.
    # For small n, give a nice explicit sum-of-divisors proof.
    if dec_digits(n) <= 8 and ctx.fac:
        divs = _divisors_from_factorization(ctx.fac)
        proper = sorted(d for d in divs if d != n)
        proof = " + ".join(map(str, proper)) + f" = {n}"
        details = f"Sum of proper divisors: {proof} ✨."
    else:
        # Large perfect numbers (or future ones): avoid enumerating divisors.
        details = f"Sum of proper divisors equals n (σ(n) = 2n); {n} is a perfect number."

    return True, details


@classifier(
    label="Semiperfect number",
    description="A positive integer equal to the sum of a subset of its proper divisors.",
    oeis="A005835",
    category=CATEGORY,
)
def is_semiperfect_number(
    n: int,
    ctx: NumCtx | None = None,
    time_limit: float = 5.0,
) -> tuple[bool | None, str | None]:

    # 1. Trivial small cases
    if n < 6:
        return False, None

    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    # 2. Quick sigma test
    aliquot = ctx.sigma - n
    if aliquot < n:
        return False, None          # cannot be semiperfect
    if aliquot == n:
        return False, None          # treat perfect numbers as NOT semiperfect (by convention)

    # 3. Reject huge n — divisor enumeration will explode
    #    (semiperfect tests are only meaningful for small–moderate n)
    if dec_digits(n) > 12:
        return None, "n too large for safe semiperfect test"

    # 4. Enumerate divisors from ctx.fac, never via sympy.divisors()
    divs, _ = enumerate_divisors_limited(ctx.fac, limit=None)
    P = sorted([d for d in divs if d < n], reverse=True)
    if not P:
        return False, None

    start = time.time()

    # 5. Greedy: fast success path for many cases
    total = 0
    used = []
    for d in P:
        if total + d <= n:
            total += d
            used.append(d)
        if total == n:
            subset = " + ".join(str(x) for x in sorted(used))
            return True, f"{n} = {subset}"

        if time.time() - start > time_limit:
            return None, "Timeout (greedy check)"

    # 6. DP (subset sum) if safe
    #    Hard limit: about 150 divisors + n ≤ 10 million
    if len(P) <= 150 and n <= 10_000_000:
        dp = [False] * (n + 1)
        prev = [None] * (n + 1)
        dp[0] = True

        for d in P:
            for i in range(n, d - 1, -1):
                if dp[i - d] and not dp[i]:
                    dp[i] = True
                    prev[i] = d
            if time.time() - start > time_limit:
                return None, "Timeout (DP check)"

        if dp[n]:
            subset = []
            x = n
            while x > 0 and prev[x] is not None:
                subset.append(prev[x])
                x -= prev[x]
            subset_str = " + ".join(str(x) for x in sorted(subset))
            return True, f"{n} = {subset_str} (DP)"
        else:
            return False, None

    # 7. Too large for DP
    return None, f"Too many divisors ({len(P)}) or n too large for fast check"


@classifier(
    label="Perfect power",
    description="An integer of the form m^k with m>1, k>1 (square, cube, …).",
    oeis="A001597",
    category=CATEGORY,
)
def is_perfect_power(n: int) -> tuple[bool, str | None]:
    """
    Bigint-safe: no floats. Uses integer n-th roots; short-circuits powers of two.
    Returns (True, 'n = m^k') if n is a perfect power with m>1, k>1; else (False, None).
    """
    # By convention we usually restrict to n > 1; adapt if you want to allow negatives.
    if n <= 1:
        return False, None

    # --- quick path: powers of two (fast bit trick) -------------------------
    # If n is a power of two, bit_count()==1 and exponent is bit_length-1
    if n & (n - 1) == 0:
        k = n.bit_length() - 1
        if k > 1:
            return True, f"{abbr_int_fast(n)} = 2^{abbr_int_fast(k)}"

    a = n  # positive

    # --- iterate only PRIME exponents k (classic trick) ---------------------
    # If n = m^k with k composite, then n is also m^(p) for some prime p|k.
    # So it suffices to test prime exponents.
    def primes_upto(L: int):
        if L < 2:
            return
        sieve = bytearray(b"\x01") * (L + 1)
        sieve[0:2] = b"\x00\x00"
        p = 2
        while p * p <= L:
            if sieve[p]:
                step = p
                sieve[p*p: L+1: step] = b"\x00" * (((L - p*p) // step) + 1)
            p += 1
        for q in range(2, L + 1):
            if sieve[q]:
                yield q

    # The largest possible exponent k satisfies 2^k <= n  →  k_max = floor(log2(n))
    k_max = a.bit_length() - 1  # exact, integer, no floats

    # Small optimization: if k_max <= 64 the loop is tiny; if k_max is large,
    # the integer_nthroot calls are still fast (they run in O(M(log n)) time).
    for k in primes_upto(k_max):
        # Negative bases are not allowed here (definition m>1); if you *do*
        # want to classify negative perfect powers like (-2)^3, allow only odd k.
        r, exact = integer_nthroot(a, k)  # r = floor(a^(1/k)), exact if r**k == a
        if exact and r > 1:
            return True, f"{abbr_int_fast(n)} = {abbr_int_fast(r)}^{abbr_int_fast(k)}"

    return False, None


@classifier(
    label="Perfect square",
    description="n = k² for some integer k.",
    oeis="A000290",
    category=CATEGORY
)
def is_perfect_square(n: int) -> tuple[bool, str]:
    """
    Check if n is a perfect square using SymPy.
    A perfect square is an integer that is the square
    of another integer; n = k².
    """
    if n < 1:
        return False, None
    if is_square(n):
        k = isqrt(n)
        return True, f"{abbr_int_fast(n)} = {abbr_int_fast(k)}² (perfect square)."
    return False, None


@classifier(
    label="Polite number",
    description="Can be expressed as a sum of two or more consecutive positive integers (equivalently: not a power of two).",
    oeis="A138591",
    category=CATEGORY
)
def is_polite_number(n: int) -> tuple[bool, str]:
    if n <= 1:
        return False, None
    # impolite if power of two
    if (n & (n - 1)) == 0:
        return False, None
    # detail via largest odd divisor > 1 ⇒ length of a consecutive-sum representation
    odd = n // (n & -n)  # strip 2-adic valuation
    details = f"Has odd divisor {odd} > 1 ⇒ sum of {odd} consecutive integers."
    return True, details


@classifier(
    label="Powerful number",
    description="Every prime factor appears with exponent at least 2.",
    oeis="A001694",
    category=CATEGORY,
)
def is_powerful_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Returns (True, details) if n is a powerful number, else (False, None).
    A number is powerful if every prime factor appears with an exponent ≥ 2.
    Requires a complete factorization; skips if ctx.incomplete.
    """
    if n < 1:
        return False, None
    if n == 1:
        return True, "1 is powerful by definition."

    ctx = ctx or build_ctx(n)

    # If factorization is incomplete, we cannot decide ⇒ SKIP
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    factors = ctx.fac

    # Safety: if for some reason we have no factor data for n>1, also SKIP
    if not factors and n > 1:
        raise TimeoutError("incomplete factorization")

    # Check exponents: all must be ≥ 2
    if any(exp < 2 for exp in factors.values()):
        return False, None

    detail = "Prime factorization: " + " × ".join(
        f"{p}^{exp}" for p, exp in sorted(factors.items())
    ) + "."
    return True, detail


@classifier(
    label="Practical number",
    description="All smaller positive integers can be written as sums of distinct divisors of the number.",
    oeis="A005153",
    category=CATEGORY
)
def is_practical_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Returns (True, details) for whether n is a practical number.
    Details show each check, aligned for further formatting.
    """
    if not isinstance(n, int) or n < 1:
        return False, None
    if n == 1:
        return True, "1 is practical by definition."
    if n % 2:
        return False, None

    ctx = ctx or build_ctx(n)
    prime_factors = ctx.fac
    # Defensive: Ensure all keys are int (not string)
    primes = sorted(int(p) for p in prime_factors)
    exponents = [prime_factors[p] for p in primes]

    if primes[0] != 2:
        return False, None

    lines = []
    # For pure power of 2
    if len(primes) == 1:
        details = f"The only prime factor is 2; condition is satisfied, {n} is practical."
        return True, details

    for i in range(1, len(primes)):
        p = primes[i]
        prev_prod = 1
        for j in range(i):
            prev_prod *= primes[j] ** exponents[j]
        # Compute σ(prev_prod) multiplicatively from the subset of (p,e)
        prev_fac = {primes[j]: exponents[j] for j in range(i)}
        prev_sigma = 1
        for p, e in prev_fac.items():
            prev_sigma *= (p**(e+1) - 1) // (p - 1)
        check_str = f"{p} ≤ 1 + σ({abbr_int_fast(prev_prod)}) = {abbr_int_fast(1 + prev_sigma)}"
        if p > 1 + prev_sigma:
            return False, None
        lines.append(check_str)

    lines.append(f"All conditions satisfied, {abbr_int_fast(n)} is practical.")
    details = ", ".join(lines)
    return True, details


@classifier(
    label="Sociable number",
    description="Aliquot sums cycle back to the start (k≥2).",
    oeis="A003416",
    category=CATEGORY,
    limit=10**6-1
)
def is_sociable_number(n: int, max_cycle_len: int = 28, min_k: int = 2) -> tuple[bool, str | None]:
    """
    True if n lies on an aliquot cycle of length k>=min_k (k=2 covers amicable).
    Uses 'steps' in messaging; internally counts distinct cycle nodes.
    """
    if n < 2:
        return False, None

    tlimit = CFG("ALIQUOT.STEP_TIME_LIMIT", 0.30)
    vlimit = CFG("ALIQUOT.VALUE_LIMIT", 10**7)

    t0 = time.perf_counter()
    seq: list[int] = [n]
    seen: dict[int, int] = {n: 0}  # value -> index in seq
    current = n

    # We attempt at most max_cycle_len transitions
    for _steps in range(1, max_cycle_len + 1):
        # Aliquot sum s(x) = sigma(x) - x
        if (time.perf_counter() - t0) > tlimit:
            raise TimeoutError("aliquot guard exceeded (time)")
        s_current = fast_sigma(current) - current
        if s_current > vlimit:
            raise TimeoutError("aliquot guard exceeded (value)")
        if s_current == 0:
            return False, None  # sequence died; primes send to 1 then 0, etc.

        next_n = s_current

        if next_n in seen:
            # We have a loop; cycle is seq[cycle_start : ]
            cycle_start = seen[next_n]
            cycle = seq[cycle_start:]
            # If closure was by revisiting n itself, we’re sociable with k=len(cycle)
            if next_n == n:
                k = len(cycle)
                if k >= min_k:
                    # Pretty print the cycle: append n at end for closure display only
                    disp = [*cycle, n]
                    cycle_str = (ARROW).join(map(str, disp))
                    details = f"Sociable cycle (k={k}): {cycle_str}"
                    return True, details
                else:
                    # Trivial cycle (k=1) or below min_k: not sociable
                    return False, None
            else:
                # Loop does not close at n -> n is preperiodic (quasi-sociable under s), not sociable
                return False, None

        # advance
        seen[next_n] = len(seq)
        seq.append(next_n)
        current = next_n

    # No closure within the allowed steps
    return False, None


@classifier(
    label="Socially aspiring number",
    description="Aliquot sequence reaches an amicable/sociable cycle (len≥2), but n isn’t in the cycle.",
    oeis="A121508",
    category="Arithmetic and Divisor-based",
    limit=10**11-1,
)
def is_socially_aspiring(n: int) -> tuple[bool, str]:
    if n <= 0:
        return False, None

    max_steps = CFG("ALIQUOT.MAX_STEPS", 250)
    tlimit = CFG("ALIQUOT.STEP_TIME_LIMIT", 0.30)
    vlimit = CFG("ALIQUOT.VALUE_LIMIT", 10**7)

    t0 = time.perf_counter()

    # --- 1) Try cached sequence first
    seq = None
    highlight = None
    entry = get_aliquot_cache(n, max_steps)
    if entry:
        seq = entry.seq
        highlight = entry.highlight_idx
        if highlight is None and seq:
            highlight = cycle_bounds_from_seq(seq)

    # --- 2) Fallback: minimal, quiet computation (fast sigma)
    if seq is None:
        seen_pos = {int(n): 0}
        seq = [int(n)]
        x = int(n)
        highlight = None
        for _ in range(int(max_steps)):
            s = fast_sigma(x) - x  # aliquot sum
            if s > vlimit or (time.perf_counter() - t0) > tlimit:
                raise TimeoutError("aliquot guard exceeded")

            seq.append(s)

            if s == 0:
                return False, None                # terminates → not socially aspiring
            if s == x:
                return False, None                # perfect fixed point → not socially aspiring

            if s in seen_pos:
                highlight = (seen_pos[s], len(seq) - 1)
                break

            seen_pos[s] = len(seq) - 1
            x = s

        if highlight is None:
            return False, None                    # no cycle within cap

    # --- 3) Analyze the cycle safely
    if highlight is None:
        return False, None

    a, b = highlight
    cycle_len = b - a
    if cycle_len < 2:
        return False, None                        # len 1 would be perfect; filtered above

    steps_to_cycle = a
    if steps_to_cycle == 0:
        return False, None                        # n itself in the cycle → not "aspiring"

    first = seq[a]
    return True, f"enters cycle (len {cycle_len}) after {steps_to_cycle} steps; starts at {first}"


@classifier(
    label="Sphenic number",
    description="A positive integer that is the product of exactly three distinct primes.",
    oeis="A007304",
    category=CATEGORY
)
def is_sphenic_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Returns (True, details) if n is sphenic (product of 3 distinct primes), else False.
    """
    if n < 30:  # smallest sphenic number is 2*3*5=30
        return False, None
    ctx = ctx or build_ctx(n)
    f = ctx.fac
    if len(f) == 3 and all(exp == 1 for exp in f.values()):
        primes = list(f.keys())
        return True, f"{n} = {primes[0]} × {primes[1]} × {primes[2]}"
    return False, None


@classifier(
    label="Squarefree number",
    description="No prime factor appears more than once in its factorization.",
    oeis="A005117",
    category=CATEGORY
)
def is_squarefree_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    return _is_k_free(n, 2, ctx)


@classifier(
    label="Sublime number",
    description="A number with a perfect number of divisors, and the sum of those divisors is also perfect.",
    oeis="A081357",
    category=CATEGORY
)
def is_sublime_number(n: int) -> tuple[bool, str]:
    KNOWN = [
        12,
        6086555670238378989670371734243169622657830773351885970528324860512791691264,
    ]

    if n in KNOWN:
        index = KNOWN.index(n) + 1  # +1 for 1-based ordinal
        ordinal = "first" if index == 1 else "second"
        details = f"This is the {ordinal} known sublime number."
        return True, details

    return False, None


@classifier(
    label="Superabundant number",
    description="The ratio σ(n)/n exceeds that of any smaller number.",
    oeis="A004394",
    category=CATEGORY,
    limit=25484247877474623694559469201315033045359474150161923076850486576760360768000,
)
def is_superabundant_number(n: int) -> tuple[bool, str | None]:
    """
    Check if n is a superabundant number (OEIS A004394).
    Uses the OEIS b-file and compares σ(n)/n to the previous record,
    with σ computed via the ctx-based pipeline.
    """
    if n < 1:
        return False, None

    found, idx_file, series, idx_set = check_oeis_bfile("b004394.txt", n)
    if not found:
        return False, None

    sigma_n = sigma_ctx(n)
    if sigma_n is None:
        # Factorization incomplete or timed out; cannot certify
        return False, None

    ratio_n = sigma_n / n

    if idx_set is not None and idx_set > 0:
        prev = series[idx_set - 1]
        sigma_prev = sigma_ctx(prev)
        if sigma_prev is None:
            details = (
                f"σ({n})={sigma_n}, σ({n})/{n}≈{ratio_n:.5f}; "
                f"previous record at {prev}, but σ({prev}) is unavailable."
            )
        else:
            ratio_prev = sigma_prev / prev
            details = (
                f"σ({n})={sigma_n}, σ({n})/{n}≈{ratio_n:.5f}; "
                f"previous record at {prev}: σ({prev})/{prev}≈{ratio_prev:.5f}"
            )
    else:
        details = (
            f"{n} is the first superabundant number; "
            f"σ({n})={sigma_n}, σ({n})/{n}≈{ratio_n:.5f}"
        )

    return True, details


@classifier(
    label="Triperfect number",
    description="Sum of divisors equals three times the number σ(n)=3n.",
    oeis="A005820",
    category=CATEGORY,
)
def is_triperfect_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    # Note (context, not used for logic): the six known examples are
    # 120, 672, 523776, 459818240, 1476304896, 51001180160,
    # and none others exist below e^350 ≈ 1.2×10^152.
    return _is_k_perfect(n, 3, ctx=ctx)


@classifier(
    label="Quadriperfect number",
    description="Sum of proper divisors equals four times the number σ(n)=4n.",
    oeis="A027687",
    category=CATEGORY,
)
def is_quadriperfect_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    return _is_k_perfect(n, 4, ctx=ctx)


@classifier(
    label="Pentaperfect number",
    description="Sum of divisors equals five times the number σ(n)=5n.",
    oeis="A046060",
    category=CATEGORY
)
def is_pentaperfect_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    return _is_k_perfect(n, 5, ctx=ctx)


@classifier(
    label="Hexaperfect number",
    description="Sum of divisors equals six times the number σ(n)=6n.",
    oeis="A046061",
    category=CATEGORY
)
def is_hexaperfect_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    return _is_k_perfect(n, 6, ctx=ctx)


@classifier(
    label="Unitary perfect number",
    description="Sum of unitary divisors equals 2n (σ*(n)=2n).",
    oeis="A002827",
    category="Arithmetic and Divisor-based",
)
def is_unitary_perfect(n: int, ctx: NumCtx = None) -> tuple[bool, str | None]:
    if n <= 0:
        return False, None
    ctx = ctx or build_ctx(n)
    us = ctx.unitary_sigma
    if us == 2 * ctx.n:
        # proofy detail with product of (1+p^e)
        parts = " × ".join(f"(1+{p}^{e})" for p, e in sorted(ctx.fac.items())) or "1"
        return True, f"σ*(n) = {parts} = {us} = 2×{ctx.n}"
    return False, None


@classifier(
    label="Untouchable number",
    description="Cannot be expressed as the sum of the proper divisors of any number.",
    oeis="A005114",
    category=CATEGORY,
    limit=100_006,
)
def is_untouchable_number(n: int) -> tuple[bool, str | None]:
    """
    Check if n is untouchable (OEIS A005114).
    Uses OEIS b-file for fast membership. Beyond precomputed range, optionally
    performs a slow search using ctx-based proper divisor sums.
    """
    if n < 2:
        return False, None

    rt = _rt_current()

    # --- Fast path: OEIS b-file lookup ---
    found, idx_file, series, idx_set = check_oeis_bfile("b005114.txt", n)
    precomp_limit = (max(series) + 1) if series else 2

    if n < precomp_limit:
        if found:
            # Determine ordinal position
            ordinal = (
                idx_file if idx_file is not None
                else (idx_set + 1 if idx_set is not None
                      else series.index(n) + 1)
            )
            details = f"{n} is the {get_ordinal_suffix(ordinal)} untouchable number."
            if n == 5:
                details += " The only known odd untouchable number."
            return True, details
        else:
            return False, None

    # --- Slow fallback beyond b-file coverage ---
    if rt.fast_mode:
        return False, None

    limit = n * 10
    for k in range(2, limit):
        s_k = proper_sum_ctx(k)    # <-- modern ctx-based aliquot sum
        if s_k is None:
            continue               # factoring of k incomplete → skip this k
        if s_k == n:
            return False, None     # n is touchable

    details = f"No integer k < {limit} has aliquot sum = {n}. {n} is untouchable."
    if n == 5:
        details += " The only known odd untouchable number."
    return True, details


@classifier(
    label="Weird number",
    description="Abundant but not semiperfect.",
    oeis="A006037",
    category=CATEGORY,
    limit=10**6-1,
)
def is_weird_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Weird number: abundant but not semiperfect.
    """
    if n < 70:
        return False, None

    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    # ---- 1) Abundance test (fast: uses σ from ctx) -------------------------
    aliquot = ctx.sigma - n
    if aliquot <= n:
        return False, None

    # ---- 2) Semiperfect test (pass real context) ---------------------------
    semip, reason = is_semiperfect_number(n, ctx=ctx)

    # semip == True  → NOT WEIRD
    if semip is True:
        return False, None

    # semip == None (timeout / too large / DP avoided)
    # For weird number classification we can *only* accept TRUE when we
    # *prove* non-semi-perfect. If semiperfect is undetermined, we cannot claim weirdness.
    if semip is None:
        return False, None

    # ---- 3) Abundant AND provably not semiperfect → Weird ------------------
    return True, (
        f"Sum of proper divisors: {aliquot} > {n}; "
        f"no subset of divisors sums to {n}."
    )


@classifier(
    label="Betrothed number",
    description="Aka Quasi Amicable number, There exists m≠n with s(n)=m+1 and s(m)=n+1 (equivalently q(n)=m and q(m)=n).",
    oeis="A005276",
    category=CATEGORY,
    limit=10**45-1
)
def is_betrothed(n: int) -> tuple[bool, str]:
    m = q_next(n)
    if m <= 0 or m == n:
        return False, None
    if q_next(m) == n:
        return True, f"partner m={m} (q(n)={m}, q(m)={n})"
    return False, None


@classifier(
    label="q-sociable number",
    description="n lies on a q-cycle under q(n)=s(n)−1 (k≥2).",
    oeis="A309227",
    category=CATEGORY,
    limit=10**45-1
)
def is_q_sociable(n: int) -> tuple[bool, str | None]:
    seq, cyc = q_sequence(n, max_steps=2000)
    if cyc is None:
        return False, None

    cycle = seq[cyc:]
    # If seq closes the loop by repeating the first cycle node, drop the duplicate
    if cycle and len(cycle) >= 2 and cycle[-1] == cycle[0]:
        cycle = cycle[:-1]

    # Must be in the cycle (membership); otherwise it's preperiodic -> q-socially aspiring
    if n not in cycle:
        return False, None

    k = len(cycle)  # number of distinct nodes
    if k < 2:
        return False, None  # exclude trivial k=1

    # Pretty/compact rendering
    preview = ", ".join(map(str, cycle[:12])) + ("…" if k > 12 else "")
    return True, f"q-cycle k={k}: {preview}"


@classifier(
    label="q-aspiring number",
    description="Under q-iteration, the sequence hits a perfect number.",
    oeis=None,
    category=CATEGORY,
    limit=10**45-1
)
def is_q_aspiring(n: int) -> tuple[bool, str]:
    seq, cyc = q_sequence(n, max_steps=2000)
    for x in seq:
        if is_perfect(x):
            return True, f"hits perfect number {x} at step {seq.index(x)}"
    return False, None


@classifier(
    label="q-socially aspiring number",
    description="Under q-iteration, reaches a q-cycle (len≥2) that does not include n.",
    oeis=None,
    category=CATEGORY,
    limit=10**45-1
)
def is_q_socially_aspiring(n: int) -> tuple[bool, str]:
    seq, cyc = q_sequence(n, max_steps=2000)
    if cyc is None:
        return False, None
    cycle = seq[cyc:]
    if n not in cycle and len(cycle) >= 2:
        return True, f"reaches q-cycle length {len(cycle)} at step {cyc}"
    return False, None


@classifier(
    label="Quasi-sociable number",
    description="Under s-iteration, the aliquot sequence enters a sociable cycle (len≥2) not containing n.",
    oeis="A002025",
    category=CATEGORY,
    limit=150000,
)
def is_quasi_sociable(n: int) -> tuple[bool, str | None]:
    # Run the s-iteration using the shared engine
    seq, highlight, aborted, skipped, _stats = compute_aliquot_sequence(
        n, kind="aliquot", max_steps=2000
    )  # seq includes starting n

    if not highlight:
        return False, None

    a, b = highlight              # cycle is seq[a:b], with distinct nodes
    cycle = seq[a:b]
    k = len(cycle)                # number of distinct cycle nodes (k ≥ 2 for sociable)

    if k < 2:
        return False, None        # length 1 would be perfect; not a sociable cycle

    if n in cycle:
        return False, None        # n is itself on the cycle ⇒ sociable, not quasi-sociable

    entry_step = a                # steps from n to first cycle node
    return True, f"reaches sociable cycle (k={k}) at step {entry_step}"


@classifier(
    label="Hemi perfect",
    description="Half-integer abundancy with even n: σ(n)/n = k/2 (k odd) and n is even.",
    oeis="A159907",
    category=CATEGORY,
)
def is_hemiperfect(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n <= 0:
        return False, None

    ctx = ctx or build_ctx(n)
    if ctx.incomplete or ctx.sigma is None:
        raise TimeoutError("incomplete factorization")

    twice = 2 * ctx.sigma  # 2×σ(n)
    if twice % ctx.n != 0:
        return False, None

    k = twice // ctx.n  # k = 2×σ(n)/n
    if k % 2 == 1:
        num, den = _abundancy_frac(ctx)  # reduced σ(n)/n
        abbrnum = abbr_int_fast(num)
        abbrden = abbr_int_fast(den)
        abbrk = abbr_int_fast(k)
        # Example detail for 24: "σ(n)/n = 5/2 = 5/2 (half-integer abundancy)"
        return True, f"σ(n)/n = {abbrnum}/{abbrden} = {abbrk}/2 (half-integer abundancy)"
    return False, None


@classifier(
    label="Solitary number",
    description="Shares its abundancy with no other m≠n; proven if gcd(n, σ(n)) = 1.",
    oeis="",  # standard notion; no single OEIS for all solitary numbers
    category=CATEGORY,
)
def is_solitary(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    # provable case only, *only* assert True when gcd(n,σ(n))=1 (a theorem).
    if n <= 0:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")
    if gcd(ctx.n, ctx.sigma) == 1:
        num, den = _abundancy_frac(ctx)
        abbrnum = abbr_int_fast(num)
        abbrden = abbr_int_fast(den)
        return True, f"gcd({abbrnum},σ({abbrnum}))=1 ⇒ solitary; σ(n)/n = {abbrnum}/{abbrden}"
    return False, None


@classifier(
    label="Friendly number",
    description="Shares its abundancy σ(n)/n with another integer (we show one).",
    oeis="A074902",
    category=CATEGORY,
)
def is_friendly(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    # Only claim True when we can *exhibit* a friend.
    if n <= 0:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")
    num, den = _abundancy_frac(ctx)
    abbrnum = abbr_int_fast(num)
    abbrden = abbr_int_fast(den)

    # If solitary by the gcd test, it's not friendly.
    if gcd(ctx.n, ctx.sigma) == 1:
        return False, None

    # Easy wins: k-perfect families (k in {2,3,4,5,6}) have multiple members.
    k, rem = divmod(ctx.sigma, ctx.n)
    if rem == 0 and k in {2, 3, 4, 5, 6}:
        examples = {
            2: [6, 28, 496, 8128],
            3: [120, 672, 523_776, 459_818_240, 1_476_304_896, 51_001_180_160],
            4: [30_240, 32_760],
            5: [14_182_439_040],
            6: [154_345_556_085_770_649_600],
        }
        for m in examples[k]:
            if m != ctx.n:
                return True, f"σ(n)/n = {abbrnum}/{abbrden}; friend m={m} (σ(m)={k}×m)"
        # If we get here, n equals the only example in the list — fall through.

    # Classic friendly “club”: 30 ↔ 140 share σ/n = 12/5.
    if (num, den) == (12, 5):
        m = 140 if n == 30 else 30
        return True, f"σ(n)/n = {abbrnum}/{abbrden}; friend m={m}"

    # No exhibited friend quickly — don’t assert
    return False, None


@classifier(
    label="Near-perfect number",
    description="Sum of proper divisors equals n plus exactly one omitted divisor (redundant d).",
    oeis="A181595",
    category=CATEGORY,
)
def is_near_perfect(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    # Sum of proper divisors except one “redundant” divisor d.
    # Condition: s(n) = σ(n) − n = n + d with d|n  ⇒ d = s(n) − n = σ(n) − 2n ∈ Div(n), d>0.
    if n <= 0:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")
    d = ctx.sigma - 2 * ctx.n
    if d > 0 and (ctx.n % d == 0):
        return True, f"σ(n)−2n = d = {d} divides n (redundant divisor)"
    return False, None


@classifier(
    label="Superperfect number",
    description="Sum of divisors, when summed again, equals twice the original number: σ(σ(n)) = 2nσ(σ(n)) = 2n.",
    oeis="A019279",
    category=CATEGORY,
    limit=10**20-1
)
def is_superperfect(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    # --- Superperfect: σ(σ(n)) = 2n ---
    if n <= 0:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")
    sigma_n = ctx.sigma
    sigma_sigma_n = build_ctx(sigma_n).sigma  # cached & cheap for moderate sizes
    if sigma_sigma_n == 2 * ctx.n:
        return True, f"σ(n)={sigma_n}, σ(σ(n))={sigma_sigma_n} = 2×{ctx.n}"
    return False, None


@classifier(
    label="k-Hyperperfect number",
    description="There exists an integer k≥1 with n = 1 + k(σ(n)−n−1).",
    oeis="A034897",  # k-hyperperfect index
    category=CATEGORY,
)
def is_k_hyperperfect(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    # --- Hyperperfect: n = 1 + k(σ(n)−n−1) ---
    if n <= 0:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")
    denom = ctx.sigma - ctx.n - 1  # = s(n) - 1
    if denom <= 0:
        return False, None
    num = ctx.n - 1
    if num % denom == 0:
        k = num // denom
        if k >= 1:
            return True, (
                f"k={k} hyperperfect: n = 1 + {k}(σ−n−1); "
                f"equivalently σ(n) = (({k}+1)n−1)/{k} = {ctx.sigma}"
            )
    return False, None


@classifier(
    label="Ore (harmonic divisor) number",
    description="Harmonic mean of divisors is integer ⇔ σ(n) | τ(n)×n.",
    oeis="A001599",
    category=CATEGORY,
)
def is_ore_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    # --- Ore (harmonic divisor) numbers: τ(n)×n / σ(n) ∈ ℕ ---
    if n <= 0:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")
    num = ctx.tau * ctx.n
    den = ctx.sigma
    if den != 0 and num % den == 0:
        H = num // den  # harmonic mean
        return True, f"τ(n)×n/σ(n) = {ctx.tau}×{ctx.n}/{ctx.sigma} = {H}"
    return False, None


@classifier(
    label="Primary pseudoperfect number",
    description="Satisfies ∑_{p|n} 1/p + 1/n = 1 (sum over distinct prime divisors).",
    oeis="A054377",
    category=CATEGORY,
)
def is_primary_pseudoperfect(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n <= 1:
        # Conventionally 1 is excluded; sequence starts at 2.
        return False, None

    ctx = ctx or build_ctx(n)
    primes = sorted(ctx.fac.keys())
    if not primes:
        return False, None

    # Exact arithmetic (no FP): sum_{p|n} 1/p + 1/n == 1 ?
    s = sum(Fraction(1, p) for p in primes) + Fraction(1, ctx.n)
    if s == 1:
        # Nice human-readable witness: "1/2 + 1/3 + 1/7 + 1/42 = 1"
        frac_terms = " + ".join([f"1/{p}" for p in primes] + [f"1/{ctx.n}"])
        # Also show the integer form: sum(n/p) + 1 = n
        int_sum = sum(ctx.n // p for p in primes) + 1
        details = (
            f"primes(n) = {'×'.join(str(p) for p in primes)}; "
            f"{frac_terms} = 1 (and ∑ n/p + 1 = {int_sum} = n)"
        )
        return True, details

    return False, None


@classifier(
    label="Giuga number",
    description=("Composite n with p | (n/p − 1) for every prime p|n; "
                 "equivalently ∑(n/p) ≡ 1 (mod n)."),
    oeis="A007850",
    category=CATEGORY,
)
def is_giuga_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n <= 1:
        return False, None
    ctx = ctx or build_ctx(n)
    if len(ctx.fac) < 2:            # must be composite
        return False, None

    primes = sorted(ctx.fac.keys())
    # per-prime test (works even if someone passes a non-squarefree n)
    if not all(((ctx.n // p) - 1) % p == 0 for p in primes):
        return False, None

    # congruence witness
    S = sum(ctx.n // p for p in primes)
    details = (f"primes(n)={'×'.join(map(str, primes))}; "
               f"∑(n/p)={S} ≡ 1 (mod {ctx.n})  "
               f"(i.e. ∑1/p − 1/n = 1)")
    return True, details


@classifier(
    label="Cube-full number",
    description="Every prime in the factorization appears with exponent ≥ 3.",
    oeis="A036966",
    category=CATEGORY,
)
def is_cube_full(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    return _is_k_full(n, 3, ctx)


DEFAULT_K_FULL = CFG("CLASSIFIER.K_FULL.K_VALUE", 4)


@classifier(
    label=f"k-full number (k={DEFAULT_K_FULL})",
    description="Every prime exponent ≥ k.",
    oeis="A036967",  # for the default value k=4
    category=CATEGORY,
)
def is_k_full(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    k = DEFAULT_K_FULL
    return _is_k_full(n, int(k), ctx)


@classifier(
    label="Cube-free number",
    description="No prime cube divides n (all prime exponents ≤ 2).",
    oeis="A004709",
    category="Arithmetic and Divisor-based",
)
def is_cube_free(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    return _is_k_free(n, 3, ctx)


@classifier(
    label="Perfect totient number",
    description="Sum of iterated totients equals n: φ(n)+φ²(n)+…+1 = n.",
    oeis="A082897",
    category=CATEGORY,
    limit=10**20-1
)
def is_perfect_totient(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n <= 1:
        return False, None
    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")
    s, seq, m = 0, [], n
    while m > 1 and s <= n:
        m = _phi_from_fac(m)
        s += m
        seq.append(m)
    if s == n:
        return True, f"φ-iterates: {' → '.join(map(str, seq))} (sum={s})"
    return False, None


@classifier(
    label="Primitive weird number",
    description="Weird and not divisible by any smaller weird number.",
    oeis="A006037",
    category=CATEGORY,
    limit=1_700_000_000
)
def is_primitive_weird(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None | None]:
    MAX_DIVISORS_LIST = 2000   # enumerate at most this many for checks
    MAX_TAU_CHECK = 8000       # skip if too many divisors
    if n <= 0:
        return False, None
    w = _is_weird_quick(n)
    if w is False:
        return False, None
    if w is None:
        return None  # skip: undecided cheaply
    # now verify no proper divisor is weird (bounded)
    ctx = ctx or build_ctx(n)
    if ctx.tau > MAX_TAU_CHECK:
        return None
    divs, trunc = enumerate_divisors_limited(ctx.fac, limit=MAX_DIVISORS_LIST)
    if trunc:
        return None
    for d in divs:
        if 1 < d < n:
            wd = _is_weird_quick(d)
            if wd:
                return False, None
            if wd is None:
                return None
    return True, "Weird, and no proper divisor is weird (checked under bounds)"


# Known colossally abundant numbers up to a moderate range (extend from literature)
CA_KNOWN = {
    2, 6, 12, 60, 120, 360, 2520, 5040, 55440, 720720, 1441440, 4324320, 21621600,
    367567200, 6983776800, 160626866400, 321253732800, 9316358251200, 288807105787200,
    2021649740510400, 6064949221531200, 224403121196654400, 9200527969062830400,
    395622702669701707200, 791245405339403414400, 37188534050951960476800, 1970992304700453905270400,
    116288545977326780410953600, 581442729886633902054768000, 35468006523084668025340848000,
    2376356437046672757697836816000, 168721307030313765796546413936000, 12316655413212904903147888217328000,
    135483209545341953934626770390608000, 10703173554082014360835514860858032000,
    21406347108164028721671029721716064000, 1776726809977614383898695466902433312000,
    5330180429932843151696086400707299936000, 474386058264023040500951689662949694304000,
    46015447651610234928592313897306120347488000, 598200819470933054071700080664979564517344000,
    60418282766564238461241708147162936016251744000, 6223083124956116561507895939157782409673929632000,
    665869894370304472081344865489882717835110470624000, 72579818486363187456866590338397216244027041298016000,
    8201519488959040182625924708238885435575055666675808000,
    1041592975097798103193492437946338450318032069667827616000,
    136448679737811551518347509370970336991662201126485417696000,
    18693469124080182558013608783822936167857721554328502224352000,
    2598392208247145375563891620951388127332223296051661809184928000,
}


@classifier(
    label="Colossally abundant number",
    description="Maximizes σ(n)/n^{1+ε} for some ε>0 (recognized from a curated list).",
    oeis="A004490",
    category=CATEGORY,
    limit=2598392208247145375563891620951388127332223296051661809184928001
)
def is_colossally_abundant(n: int) -> tuple[bool, str | None]:
    if n in CA_KNOWN:
        return True, "Listed among known colossally abundant numbers"
    return False, None


@classifier(
    label="Zumkeller number",
    description="Divisors can be partitioned into two equal-sum sets.",
    oeis="A083207",
    category=CATEGORY,
    limit=10**15 - 1
)
def is_zumkeller_number(n: int, ctx: NumCtx | None = None, *, with_witness: bool = True) -> tuple[bool, str | None]:
    # Settings / caps

    if n <= 0:
        return False, None

    limit_tau = CFG("CLASSIFIER.ZUMKELLER.DIVISOR_LIMIT", 5000)
    tau_small = CFG("CLASSIFIER.ZUMKELLER.SMALL_TAU_MITM", 32)
    cap_target = CFG("CLASSIFIER.ZUMKELLER.BITSET_TARGET_CAP", 10**7)
    witness_cap = CFG("CLASSIFIER.ZUMKELLER.WITNESS_TARGET_CAP", 200_000)

    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    # Early guard: use multiplicative τ(n) from ctx
    if ctx.tau is not None and ctx.tau > limit_tau:
        raise TimeoutError(f"zumkeller guard: τ(n) = {ctx.tau} > {limit_tau}")

    S = ctx.sigma
    if S & 1:
        return False, "σ(n) is odd ⇒ impossible"
    T = S // 2

    # Short-circuits
    if 2 * n > S:
        return False, "Deficient ⇒ not Zumkeller"
    if 2 * n == S:
        msg = f"Perfect ⇒ Zumkeller; σ(n)={S}, target={T}"
        if with_witness:
            return True, f"{msg}; witness: [{n}] (sum {T})"
        return True, msg

    # --- NEW: remember practical reason as a prefix, but still build a witness ---
    prefix = None
    try:
        ok, _ = is_practical_number(n, ctx=ctx)
        if ok:
            prefix = f"Practical & even σ(n) ⇒ Zumkeller; σ(n)={S}, target={T}"
            if not with_witness:
                return True, prefix
    except Exception:
        pass

    def _join(msg: str) -> str:
        return f"{prefix}; {msg}" if prefix else msg

    # Divisors (≤ T)
    divs = _divisors_from_factorization(ctx.fac)
    divs = [d for d in divs if d <= T]
    tau = len(divs)

    if tau > limit_tau:
        raise TimeoutError(f"zumkeller guard: divisor count > {limit_tau}")

    if sum(divs) < T:
        return False, "Sum(divisors ≤ T) < target"

    # Path A: MITM (with witness)
    if tau <= tau_small:
        A = divs[: tau // 2]
        B = divs[tau // 2:]

        sumsA = [(0, 0)]
        for i, a in enumerate(A):
            base = sumsA[:]
            sumsA += [(s + a, m | (1 << i)) for (s, m) in base if s + a <= T]

        sumsB = [(0, 0)]
        for j, b in enumerate(B):
            base = sumsB[:]
            sumsB += [(s + b, m | (1 << j)) for (s, m) in base if s + b <= T]

        dictB = {}
        for s, m in sumsB:
            if s not in dictB:
                dictB[s] = m

        for sA, mA in sumsA:
            need = T - sA
            mB = dictB.get(need)
            if mB is not None:
                if with_witness:
                    subset = [A[i] for i in range(len(A)) if (mA >> i) & 1] \
                           + [B[j] for j in range(len(B)) if (mB >> j) & 1]
                    subset.sort()
                    return True, _join(f"partition found (MITM); witness: {subset} (sum {sum(subset)})")
                else:
                    return True, _join("partition found (MITM)")
        return False, None

    # Path B: DP with predecessor (witness) when T is small
    if cap_target < T:
        raise TimeoutError("zumkeller guard: target too large")

    if with_witness and witness_cap >= T:
        reach = bytearray(T + 1)
        reach[0] = 1
        prev = array('i', [-1] * (T + 1))
        for d in divs:
            for s in range(T, d - 1, -1):
                if not reach[s] and reach[s - d]:
                    reach[s] = 1
                    prev[s] = d
            if reach[T]:
                subset = []
                s = T
                while s > 0:
                    dd = prev[s]
                    subset.append(dd)
                    s -= dd
                subset.sort()
                return True, _join(f"partition found (DP); witness: {subset} (sum {T})")
        return False, None

    # Fast bitset (no witness)
    bits = 1
    mask = (1 << (T + 1)) - 1
    for d in divs:
        bits |= (bits << d)
        if bits.bit_length() > T + 1:
            bits &= mask
        if (bits >> T) & 1:
            return True, _join("partition found (bitset)")

    return False, None
