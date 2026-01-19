# -----------------------------------------------------------------------------
#  conjectures.py
#  Conjectures and Equation-based
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from math import ceil, prod

from sympy import isprime, nextprime, primerange

from numclass.registry import classifier
from numclass.runtime import CFG
from numclass.runtime import current as _rt_current
from numclass.utility import _divisors_from_factorization, build_ctx

CATEGORY = "Conjectures and Equation-based"


@classifier(
    label="Goldbach conjecture",
    description="Every even positive integer > 2 can be written as the sum of two primes.",
    oeis="A045917",
    category=CATEGORY,
    limit=10**7-1,
)
def is_goldbach_sum_of_two_primes(n: int, ctx=None):
    # Only even n >= 4 are relevant
    if (n & 1) or n < 4:
        return False, None

    # Settings
    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    show_all = bool(CFG("CLASSIFIER.GOLDBACH.SHOW_ALL", False))
    max_list = int(CFG("CLASSIFIER.GOLDBACH.MAX_LIST", 10))
    show_count = bool(CFG("CLASSIFIER.GOLDBACH.SHOW_COUNT", True))
    PROBE_CAP = int(CFG("CLASSIFIER.GOLDBACH.PROBE_CAP", 500_000))  # how many p to try in quick path
    SIEVE_MAXN = int(CFG("CLASSIFIER.GOLDBACH.SIEVE_MAXN", 10_000_000))  # absolute cap for sieve use

    half = n // 2

    # ---------- Quick path (no sieve): find *one* pair fast ----------
    # For huge n this is the only sane approach.
    if fast_mode and not show_all:
        # try p=2 first (common quick win)
        q = n - 2
        if isprime(q):
            return True, f"{n} = 2 + {q}"

        # then try odd p up to a probe cap
        p = 3
        probes = 0
        while p <= half and probes < PROBE_CAP:
            if isprime(p) and isprime(n - p):
                return True, f"{n} = {p} + {n - p}"
            p += 2
            probes += 1

        # Couldn’t find quickly (don’t claim false—just no example within cap)
        return False, None

    # ---------- Slow/explicit enumeration with sieve (only for small n) ----------
    if n > SIEVE_MAXN:
        # Don’t even attempt a sieve this large.
        return False, None  # or: (False, f"Skipped: n>{SIEVE_MAXN} not enumerated")

    sieve = _sieve_isprime_upto(n)  # MUST return a plain sequence; no unclosed memoryview!

    pairs = []
    total = 0
    for p in range(2, half + 1):
        if not sieve[p]:
            continue
        q = n - p
        if q < p:
            break
        if sieve[q]:
            total += 1
            if len(pairs) < max_list or show_all:
                pairs.append((p, q))

    if total == 0:
        return False, None

    if show_all:
        shown = ", ".join(f"{p}+{q}" for p, q in pairs[:max_list])
        more = "" if total <= max_list else f" … (+{total - max_list} more)"
        return True, f"{n} has {total} Goldbach pairs: {shown}{more}"

    if show_count:
        shown = ", ".join(f"{p}+{q}" for p, q in pairs[:max_list])
        more = "" if total <= max_list else f" … (+{total - max_list} more)"
        p0, q0 = pairs[0]
        return True, f"{n} = {p0} + {q0}  ({total} pairs; examples: {shown}{more})"

    p0, q0 = pairs[0]
    shown = ", ".join(f"{p}+{q}" for p, q in pairs[:max_list])
    return True, f"{n} = {p0} + {q0}  (examples: {shown})"


def _sieve_isprime_upto(N: int):
    """
    Simple bytearray sieve up to N (inclusive). True means 'prime'.
    O(N log log N) time, O(N) bytes. Fine up to ~10M (see decorator limit).
    """
    if N < 2:
        return bytearray(2)
    sieve = bytearray(b"\x01") * (N + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(N ** 0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:N + 1:step] = b"\x00" * (((N - start) // step) + 1)
    return sieve


@classifier(
    label="Weak Goldbach (ternary)",
    description="Odd n ≥ 7 as p + q + r (primes)",
    oeis="A068307",
    category=CATEGORY,
    limit=500_000,
)
def is_weak_goldbach(n: int, ctx=None):
    if n < 7 or (n % 2 == 0):
        return False, None

    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    show_all = bool(CFG("CLASSIFIER.WEAK_GOLDBACH.SHOW_ALL", False))
    max_list = int(CFG("CLASSIFIER.WEAK_GOLDBACH.MAX_LIST", 5))
    show_count = bool(CFG("CLASSIFIER.WEAK_GOLDBACH.SHOW_COUNT", True))

    sieve = _sieve_isprime_upto(n)
    primes = [i for i in range(2, n + 1) if sieve[i]]

    triples = []
    total = 0

    for p in primes:
        if p >= n - 4:  # need at least 2+2 for the remaining even
            break
        m = n - p  # even
        # Find any q with q ≤ m/2 and sieve[m-q] true
        # (q and r = m-q are primes; order them q ≤ r)
        for q in primes:
            if q > m // 2:
                break
            r = m - q
            if sieve[r]:
                total += 1
                if show_all or len(triples) < max_list:
                    triples.append((p, q, r))

                if fast_mode and not show_all:
                    # fast: return the first triple
                    return True, f"{n} = {p} + {q} + {r}"
                # else continue to count/collect more
        # small optimization: if we found at least one for this p and fast mode is allowed,
        # we would have returned. Otherwise keep scanning.

    if total == 0:
        # The theorem says this won't happen for n ≥ 7, but be safe
        return False, None

    if show_all or (not fast_mode):
        shown = ", ".join(f"{p}+{q}+{r}" for p, q, r in triples[:max_list])
        more = "" if total <= max_list else f" … (+{total - max_list} more)"
        if show_count:
            return True, f"{n} has {total} ternary Goldbach triples: {shown}{more}"
        else:
            return True, f"{n} examples: {shown}{more}"

    # Fallback (shouldn’t reach if fast-return used)
    p, q, r = triples[0]
    return True, f"{n} = {p} + {q} + {r}"


@classifier(
    label="Lemoine's (Levy's)",
    description="Odd n > 5 as p + 2q with primes p,q",
    oeis="A194828",
    category=CATEGORY,
    limit=10**7-1,
)
def is_lemoine(n: int, ctx=None):
    if n <= 5 or (n % 2 == 0):
        return False, None

    # Settings
    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    show_all = bool(CFG("CLASSIFIER.LEMOINE.SHOW_ALL", False))
    max_list = int(CFG("CLASSIFIER.LEMOINE.MAX_LIST", 10))
    show_count = bool(CFG("CLASSIFIER.LEMOINE.SHOW_COUNT", True))

    sieve = _sieve_isprime_upto(n)
    primes = [i for i in range(2, n + 1) if sieve[i]]

    pairs = []
    total = 0
    half = n // 2

    for q in primes:
        if q > half:
            break
        p = n - 2 * q
        if p >= 2 and sieve[p]:
            total += 1
            if show_all or len(pairs) < max_list:
                pairs.append((p, q))

            if fast_mode and not show_all:
                # fast mode: return first hit
                return True, f"{n} = {p} + 2×{q}"

    if total == 0:
        return False, None

    # Compose detail
    if show_all or (not fast_mode):
        shown = ", ".join(f"{p}+2×{q}" for p, q in pairs[:max_list])
        more = "" if total <= max_list else f" … (+{total - max_list} more)"
        if show_count:
            return True, f"{n} has {total} Lemoine representations: {shown}{more}"
        else:
            return True, f"{n} examples: {shown}{more}"

    # Fallback (shouldn’t reach if fast-return above fired)
    p, q = pairs[0]
    return True, f"{n} = {p} + 2×{q}"


@classifier(
    label="Legendre prime interval",
    description="For n>0 there is always a prime between n² and (n+1)².",
    oeis="A014085",
    category=CATEGORY,
    limit=575_000
)
def is_legendre_between_squares(n: int, ctx=None):
    """
    Legendre witness for the square interval containing n.
    Always uses m=floor(sqrt(n)) so it works for every n>0.
    """
    if n < 1:
        return False, None

    # Settings
    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    show_all = bool(CFG("CLASSIFIER.LEGENDRE.SHOW_ALL", False))
    max_list = int(CFG("CLASSIFIER.LEGENDRE.MAX_LIST", 20))
    show_count = bool(CFG("CLASSIFIER.LEGENDRE.SHOW_COUNT", True))

    lo = n * n
    hi = (n + 1) * (n + 1)  # exclusive upper bound

    # --- Fast witness: least prime in (n², (n+1)²)
    p = nextprime(lo)
    if p >= hi:
        # Defensive: would contradict Legendre for m>=1 ever hit.
        return False, None

    # If we just need a witness, return immediately
    if fast_mode and not show_all and max_list <= 1:
        return True, f"least prime {p} in ({lo}, {hi})"

    # Expanded: count & (optionally) list up to max_list primes
    total = 0
    primes = []
    for q in primerange(lo + 1, hi):  # q runs over primes with lo < q < hi
        total += 1
        if show_all or len(primes) < max_list:
            primes.append(q)

    shown = ", ".join(map(str, primes[:max_list]))
    more = "" if total <= max_list else f" … (+{total - max_list} more)"
    if show_count:
        return True, f"{total} prime(s) in ({lo}, {hi}): {shown}{more}"
    else:
        return True, f"primes in ({lo}, {hi}): {shown}{more}"


# --- Egyptian fractions & conjectures ----------------------------------------

# ------------------------------ utilities ------------------------------------

def _two_term_examples(m: int, n: int, distinct: bool, ctx) -> Iterator[tuple[int, int]]:
    """
    Generate all (x,y) with 1/x + 1/y = m/n using (m x - n)(m y - n) = n^2.

    x = (n + u)/m,  y = (n + n^2/u)/m for divisors u|n^2.
    """
    if m <= 0 or n <= 0:
        return
    # Try to get factorization of n to enumerate divisors of n^2 quickly.
    if getattr(ctx, "fac", None):
        fac2 = {p: 2 * a for p, a in ctx.fac.items()}
        divs = _divisors_from_factorization(fac2)
    else:
        # Fallback: minimal set keeps latency bounded; may miss solutions.
        divs = [1, n * n]

    seen = set()
    for u in divs:
        if (n * n) % u != 0:
            continue
        num_x = n + u
        if num_x % m != 0:
            continue
        x = num_x // m

        v = (n * n) // u
        num_y = n + v
        if num_y % m != 0:
            continue
        y = num_y // m

        if x <= 0 or y <= 0:
            continue
        a, b = (x, y) if x <= y else (y, x)
        if distinct and a == b:
            continue
        # Verify exactly
        if n * (a + b) != m * a * b:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        yield (a, b)


def _format_unit_sum(m: int, n: int, k: int, denoms: Iterable[int]) -> str:
    parts = [f"1/{d}" for d in denoms]
    details = f"{m}/{n} = " + " + ".join(parts) if k == 0 else f"(m={m}, k={k}), {m}/{n} = " + " + ".join(parts)
    return details


def _egyptian_search(  # noqa: PLR0913
    m: int,
    n: int,
    k: int,
    *,
    distinct: bool,
    max_denom: int,
    time_budget_s: float,
    max_examples: int,
) -> list[tuple[int, ...]]:
    """
    Greedy + backtrack with tight bounds, returning up to max_examples k-tuples.

    We search strictly increasing denominators to avoid duplicates and to enforce
    'distinct' naturally. (If distinct=False ever needed, you can adapt the start
    bound logic; by default Numclass uses distinct=True.)
    """
    t0 = time.perf_counter()
    target_num, target_den = m, n
    solutions: list[tuple[int, ...]] = []

    def time_exceeded() -> bool:
        return (time.perf_counter() - t0) >= time_budget_s

    def backtrack(cur: list[int], num: int, den: int, start_from: int, r: int) -> None:
        # Stop conditions
        if time_exceeded() or len(solutions) >= max_examples:
            return
        if r == 0:
            # success if residue is exactly 0
            if num == 0:
                solutions.append(tuple(cur))
            return

        # Lower bound for next denominator by Engel/Sylvester bound:
        # 1/d <= num/den  =>  d >= ceil(den/num)
        lb = max(start_from, ceil(den / num))
        if lb > max_denom:
            return

        # Upper bound/pruning: with r terms left, max possible sum if we take r copies of 1/lb
        # (strictly increasing means next denominators will be >= lb, so sum <= r/lb)
        # If r/lb < num/den => impossible
        if r * den < lb * num:
            return

        # Iterate candidates
        d = lb
        while d <= max_denom:
            # Try taking 1/d
            # new residue: num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            if new_num >= 0:
                # Greedy feasibility: remaining r-1 terms from next d+1
                # Quick prune: even taking (r-1) copies of 1/d won't reach target if (r-1)/d < new_num/(den*d)
                if r > 1 and (r - 1) * den < new_num:
                    # since denominator would be >= d+1, this is a generous prune
                    pass
                # Only proceed if residue non-negative and achievable
                if new_num == 0 and r == 1:
                    solutions.append((*cur, d))
                    if len(solutions) >= max_examples or time_exceeded():
                        return
                elif new_num > 0 and r > 1:
                    # enforce strictly increasing denominators
                    backtrack([*cur, d], new_num, den * d, d + 1, r - 1)
                    if len(solutions) >= max_examples or time_exceeded():
                        return
            # Advance d; basic heuristic: cap d growth slightly by residue
            d += 1

    # Kick off search
    backtrack([], target_num, target_den, start_from=1, r=k)
    return solutions


# --------------------------- template hooks ----------------------------------

def _template_erdos_straus(n: int, distinct: bool, max_examples: int) -> list[tuple[int, int, int]]:
    """
    Safe template hook for 4/n = 1/x + 1/y + 1/z.
    (Conservative version: currently returns no parametric families to avoid incorrect identities.)
    Add proven families here later; keep return verified.
    """
    # Placeholder: no general parametric families enabled yet.
    return []


def _template_sierpinski_via_lift(
    n: int,
    distinct: bool,
    time_budget_s: float,
    max_denom: int,
    max_examples: int,
) -> list[tuple[int, int, int, int]]:
    """
    Guaranteed 'lift': 5/n = 1/n + (4/n decomposition).
    We first try to find 4/n as a 3-term Egyptian fraction (quick bounded search),
    then add 1/n to obtain a 4-term witness for 5/n.
    """
    # Small sub-budget for the inner call to keep things snappy
    sub_budget = min(0.10, max(0.02, 0.5 * time_budget_s))
    triples = egyptian_resolve(4, n, 3,
                               distinct=distinct,
                               time_budget_s=sub_budget,
                               max_denom=max_denom,
                               max_examples=max_examples)
    quads: list[tuple[int, int, int, int]] = []
    for t in triples:
        cand = tuple(sorted((n, *t)))
        if distinct and len(set(cand)) != 4:
            # If you insist on distinct denominators, we may need to find another 4/n triple
            # or fall back to general 5/n search. For now, only accept if distinct.
            continue
        quads.append(cand)
        if len(quads) >= max_examples:
            break
    return quads


# ---------------------------- public resolver --------------------------------

def egyptian_resolve(  # noqa: PLR0913
    m: int,
    n: int,
    k: int,
    *,
    distinct: bool,
    time_budget_s: float,
    max_denom: int,
    max_examples: int,
) -> list[tuple[int, ...]]:
    """
    Generic Egyptian fraction resolver:
      returns up to max_examples sorted k-tuples (d1<...<dk) with  m/n = sum 1/di.
      Empty list => none found within caps.
    """
    if n <= 0 or m <= 0 or k <= 0:
        return []

    # 1) Templates always enabled (non-configurable)
    if m == 4 and k == 3:
        sols = _template_erdos_straus(n, distinct=distinct, max_examples=max_examples)
        if sols:
            return sols[:max_examples]

    if m == 5 and k == 4:
        # First try the guaranteed lift; if distinct is True this may return empty for some n
        sols5 = _template_sierpinski_via_lift(
            n, distinct=distinct, time_budget_s=time_budget_s,
            max_denom=max_denom, max_examples=max_examples
        )
        if sols5:
            return sols5[:max_examples]
        # Fall through to general search if lift didn't yield distinct denominators

    # 2) Fast exact solver for k = 2
    if k == 2:
        ctx = build_ctx(n)
        out = []
        for x, y in _two_term_examples(m, n, distinct=distinct, ctx=ctx):
            out.append((x, y))
            if len(out) >= max_examples:
                break
        return out

    # 3) Generic bounded search for k >= 3
    return _egyptian_search(
        m, n, k,
        distinct=distinct,
        max_denom=max_denom,
        time_budget_s=time_budget_s,
        max_examples=max_examples,
    )


# ---------------------------- Classifiers ------------------------------------

@classifier(
    label="Egyptian m/n (profile)",
    description="Searches m/n as a sum of k unit fractions (profile-driven).",
    oeis="A192881",
    category=CATEGORY,
)
def is_egyptian_profile(n: int) -> tuple[bool, str | None]:
    if n <= 1:
        return False, None
    m = int(CFG("CLASSIFIER.EGYPTIAN.M_VALUE", 4))
    k = int(CFG("CLASSIFIER.EGYPTIAN.K_VALUE", 2))
    distinct = bool(CFG("CLASSIFIER.EGYPTIAN.DISTINCT", True))
    time_budget_s = float(CFG("CLASSIFIER.EGYPTIAN.TIME_BUDGET_S", 0.20))
    max_denom = int(CFG("CLASSIFIER.EGYPTIAN.MAX_DENOM", 10_000_000))
    max_examples = int(CFG("CLASSIFIER.EGYPTIAN.MAX_EXAMPLES", 1))

    sols = egyptian_resolve(
        m, n, k,
        distinct=distinct,
        time_budget_s=time_budget_s,
        max_denom=max_denom,
        max_examples=max_examples,
    )
    if not sols:
        return False, None
    # Print first witness only
    return True, _format_unit_sum(m, n, k, sols[0])


@classifier(
    label="Erdős–Straus",
    description="Bounded check for 4/n = 1/x + 1/y + 1/z.",
    oeis="A073101",
    category=CATEGORY,
)
def is_erdos_straus_k3(n: int) -> tuple[bool, str | None]:
    if n <= 1:
        return False, None
    distinct = bool(CFG("CLASSIFIER.EGYPTIAN.DISTINCT", True))
    time_budget_s = float(CFG("CLASSIFIER.EGYPTIAN.TIME_BUDGET_S", 0.20))
    max_denom = int(CFG("CLASSIFIER.EGYPTIAN.MAX_DENOM", 10_000_000))
    max_examples = int(CFG("CLASSIFIER.EGYPTIAN.MAX_EXAMPLES", 1))

    sols = egyptian_resolve(
        4, n, 3,
        distinct=distinct,
        time_budget_s=time_budget_s,
        max_denom=max_denom,
        max_examples=max_examples,
    )
    if not sols:
        return False, None
    return True, _format_unit_sum(4, n, 0, sols[0])


@classifier(
    label="Sierpiński",
    description="Bounded check for 5/n as a sum of four unit fractions.",
    oeis=None,
    category=CATEGORY,
)
def is_sierpinski_k4(n: int) -> tuple[bool, str | None]:
    if n <= 1:
        return False, None
    distinct = bool(CFG("CLASSIFIER.EGYPTIAN.DISTINCT", True))
    time_budget_s = float(CFG("CLASSIFIER.EGYPTIAN.TIME_BUDGET_S", 0.20))
    max_denom = int(CFG("CLASSIFIER.EGYPTIAN.MAX_DENOM", 10_000_000))
    max_examples = int(CFG("CLASSIFIER.EGYPTIAN.MAX_EXAMPLES", 1))

    sols = egyptian_resolve(
        5, n, 4,
        distinct=distinct,
        time_budget_s=time_budget_s,
        max_denom=max_denom,
        max_examples=max_examples,
    )
    if not sols:
        return False, None
    return True, _format_unit_sum(5, n, 0, sols[0])


# --- Znám helpers ---


def _znam_find_sets(  # noqa: PLR0913
    start: int,
    k: int,
    *,
    odd_only: bool,
    proper: bool,
    time_budget_s: float,
    factor_time_budget_s: float,
    max_product_bits: int,
    max_examples: int,
) -> list[tuple[int, ...]]:
    """
    Enumerate up to max_examples proper Znám sets of length k starting at 'start'.

    Improvements over the plain backtracker:
      • Smarter candidate generation at non-final levels:
          - maintain modulus target m_S = product(S / {2}) (pairwise coprime),
          - score candidates that satisfy d ≡ 1 (mod m_S) highest,
            then d ≡ 1 (mod any factor in S / {2}),
            then (optionally) primality bonus,
          - always require gcd(d, P) = 1 and d > last.
      • Adaptive window: widen only when remaining == 1 (the level right before CRT).
      • Same final step: close using Chinese remainder theorem (CRT) to compute residue for d_k, then scan divisors of (P+1).
      • Uses your caps (time, factor time, product bits, branch cap).
      • Progress line emitted via _progress(...) when available.
    """

    t0 = time.perf_counter()

    def timed_out() -> bool:
        return (time.perf_counter() - t0) >= time_budget_s

    # ---- tiny MR primality (heuristic for scoring; correctness does not depend on it)
    def _mr_is_probable_prime(n: int) -> bool:
        if n < 2:
            return False
        small_primes = (2, 3, 5, 7, 11, 13, 17)
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # write n-1 = d * 2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Deterministic bases good for 64-bit; here it's only a hint for ordering.
        for a in (2, 3, 5, 7, 11):
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Base sanity: {start} as seed is acceptable; full Znám will be verified at hits.

    solutions: list[tuple[int, ...]] = []

    def next_d_candidates(last_d: int, P: int, S: list[int], remaining: int) -> list[int]:
        """
        Assemble a scored, ordered list of next candidates:
          - window: BEHAVIOUR.ZNAM_PREFIX_WINDOW (default 200), widened to max(base,2000) if remaining==1
          - enforce gcd(P,d)==1, d>last, odd-only if requested
          - score by congruence preference:
              +2 if d ≡ 1 (mod m_S) with m_S = product(S / {2});
              +1 if d ≡ 1 (mod any q in S / {2});
              +1 if probable prime (cheap MR).
          - return the best up to BEHAVIOUR.ZNAM_MAX_BRANCH_PER_LEVEL
        """
        base = CFG("CLASSIFIER.ZNAM.PREFIX_WINDOW", 200)
        max_window = CFG("CLASSIFIER.ZNAM.MAX_WINDOW", 2000)
        window = max_window if remaining == 1 else base

        # modulus target m_S
        ms_factors = [d for d in S if d != 2]
        mS = 1
        for q in ms_factors:
            mS *= q  # S is kept pairwise-coprime

        # range
        if odd_only:
            start_d = last_d + 1 + (1 if (last_d + 1) % 2 == 0 else 0)
            step = 2
        else:
            start_d = last_d + 1
            step = 1
        end_d = last_d + window

        scored: list[tuple[int, int]] = []  # (neg_score, d)
        d = start_d
        while d <= end_d:
            # coprime to current product
            if _gcd(P, d) == 1:
                score = 0
                if mS > 1 and d % mS == 1:
                    score += 2
                else:
                    # 1 mod any factor (other than 2)
                    for q in ms_factors:
                        if d % q == 1:
                            score += 1
                            break
                # small primality bonus
                try:
                    if _mr_is_probable_prime(d):
                        score += 1
                except Exception:
                    pass
                scored.append((-score, d))
            d += step

        scored.sort()  # by score desc, then by d asc
        cap = CFG("CLASSIFIER.ZNAM.MAX_BRANCH_PER_LEVEL", 800)
        return [d for _, d in scored[:cap]]

    def backtrack_prefix(S: list[int]) -> None:
        if timed_out() or len(solutions) >= max_examples:
            return

        depth = len(S)

        P = prod(S)
        if P.bit_length() > max_product_bits:
            return

        if depth == k - 1:
            # Final step: compute dk via CRT constraints + divisors of P+1
            # Build congruences: dk ≡ ai (mod mi=di), where ai = - (P/di)^(-1) (mod di)
            a_mod_m: list[tuple[int, int]] = []
            for di in S:
                mi = di
                Pi = P // di
                inv = _inv_mod(Pi % mi, mi)
                if inv is None:
                    return  # should not happen (pairwise coprime)
                ai = (-inv) % mi
                a_mod_m.append((ai, mi))

            res = _crt_list(a_mod_m)
            if res is None:
                return
            R, M = res  # want dk ≡ R (mod M=P)

            P1 = P + 1
            fac = _small_factorization_with_budget(P1, factor_time_budget_s)
            divs = _divisors_from_factorization(fac)

            last = S[-1]
            for dk in divs:
                if dk <= last:
                    continue
                if proper and dk >= P1:
                    continue
                if odd_only and (dk % 2 == 0):
                    continue
                if dk % M != R % M:
                    continue
                S_full = [*S, dk]
                if _znam_verify_full(S_full, proper=proper):
                    solutions.append(tuple(S_full))
                    if len(solutions) >= max_examples or timed_out():
                        return
            return  # done at this depth

        # Non-final level: extend S by one candidate
        last = S[-1]
        remaining = (k - 1) - depth
        children = next_d_candidates(last, P, S, remaining)
        for d in children:
            if timed_out() or len(solutions) >= max_examples:
                return
            Pnext = P * d
            if Pnext.bit_length() > max_product_bits:
                # larger d will only increase product → prune rest
                break
            S2 = [*S, d]
            # Pairwise-coprime maintained by gcd(P,d)==1; final Znám checks at close
            backtrack_prefix(S2)

    # Kick off with anchor
    backtrack_prefix([start])

    return solutions


def _small_factorization_with_budget(N: int, time_budget_s: float) -> dict[int, int]:
    """
    Very small, budgeted trial-division factorization.
    Returns a (possibly partial) factorization {prime: exp}. If time runs out,
    any remaining >1 is placed as a single prime-like factor (exp=1).
    This is fine: we may miss some divisors of (P+1), but search stays bounded.
    """
    t0 = time.perf_counter()
    fac: dict[int, int] = {}
    n = N

    def step():
        return (time.perf_counter() - t0) >= time_budget_s

    # Pull out factors of 2
    c = 0
    while (n & 1) == 0:
        n >>= 1
        c += 1
        if step():
            break
    if c:
        fac[2] = c
    if step():
        if n > 1:
            fac[n] = fac.get(n, 0) + 1
        return fac

    # Trial division by odd numbers (6k±1 wheel-lite)
    p = 3
    while p * p <= n:
        if step():
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
                if step():
                    break
            fac[p] = fac.get(p, 0) + e
            if step():
                break
        p += 2

    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    return fac


def _crt_list(a_mod_m: list[tuple[int, int]]) -> tuple[int, int] | None:
    a, m = a_mod_m[0]
    for ai, mi in a_mod_m[1:]:
        res = _crt_pair(a, m, ai, mi)
        if res is None:
            return None
        a, m = res
    return (a, m)


def _crt_pair(a1: int, m1: int, a2: int, m2: int) -> tuple[int, int] | None:
    """
    Solve x ≡ a1 (mod m1), x ≡ a2 (mod m2).
    Returns (a, m) with x ≡ a (mod m) or None if incompatible.
    For pairwise-coprime moduli (our case), this always succeeds.
    """
    g = _gcd(m1, m2)
    if (a2 - a1) % g != 0:
        return None
    # Reduce to coprime case
    m1p, m2p = m1 // g, m2 // g
    inv = _inv_mod(m1p % m2p, m2p)
    if inv is None:
        return None
    t = ((a2 - a1) // g) % m2p
    k = (t * inv) % m2p
    a = (a1 + k * m1) % (m1 * m2p)
    m = m1 * m2p
    return (a, m)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    # returns (g, x, y) with ax + by = g = gcd(a,b)
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _inv_mod(a: int, m: int) -> int | None:
    a %= m
    g, x, _ = _egcd(a, m)
    if g != 1:
        return None
    return x % m


def _znam_verify_full(S: list[int], proper: bool) -> bool:
    """
    Verify Znám condition for every t in S:
      t | (P/t) + 1  and (if proper)  t < (P/t) + 1
    """
    P = prod(S)
    for t in S:
        other = P // t
        val = other + 1
        if val % t != 0:
            return False
        if proper and not (t < val):
            return False
    return True


def _format_znam_result(S: tuple[int, ...]) -> str:
    P = prod(S)
    # Proper Znám sets satisfy sum(1/d) = 1 - 1/P
    # We present both the chain and the identity.
    chain = " + ".join(f"1/{d}" for d in S)
    return f"{chain} = 1 − 1/{P})"


@classifier(
    label="Znám chain",
    description="Finds a proper Znám set of length k (=input).",
    oeis="A075441",
    category=CATEGORY,
)
def is_znam_chain(n: int) -> tuple[bool, str | None]:
    # Interpret input n as desired length k
    try:
        k = int(n)
    except Exception:
        return False, None
    if k < 2 or k > 10:
        return False, "k must be greater in the range 2 - 10."

    start = CFG("CLASSIFIER.ZNAM.START", 2)
    require_proper = CFG("CLASSIFIER.ZNAM.REQUIRE_PROPER", True)
    odd_only = CFG("CLASSIFIER.ZNAM.ODD_ONLY", False)
    max_examples = CFG("CLASSIFIER.ZNAM.MAX_EXAMPLES", 1)

    time_budget_s = CFG("CLASSIFIER.ZNAM.TIME_BUDGET_S", 0.5)
    max_product_bits = CFG("CLASSIFIER.ZNAM.MAX_PRODUCT_BITS", 512)
    factor_time_budget_s = CFG("FACTORING.MAX_TIME_S", 0.2)

    if require_proper and k < 5:
        return False, "k must be in the range 5 - 10."

    sols = _znam_find_sets(
        start, k,
        odd_only=odd_only,
        proper=require_proper,
        time_budget_s=time_budget_s,
        factor_time_budget_s=factor_time_budget_s,
        max_product_bits=max_product_bits,
        max_examples=max_examples,
    )
    if not sols:
        note = " (odd-only solutions are unknown; may legitimately fail)" if odd_only else ""
        return False, f"No Znám chain found within caps{note}."

    # Print up to ZNAM_MAX_EXAMPLES witnesses
    shown = sols[:max_examples]
    if len(shown) == 1:
        return True, _format_znam_result(shown[0])

    lines = [_format_znam_result(S) for S in shown]
    # Join on newlines; Numclass will wrap details if needed.
    return True, ", ".join(lines)
