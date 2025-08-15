# -----------------------------------------------------------------------------
#  Arithmetic and Divisor-based test functions
# -----------------------------------------------------------------------------

import time

from data.sum_of_3_cubes_solutions import SUM_OF_THREE_CUBES_SOLUTIONS
from decorators import classifier, limited_to
from functools import reduce
from math import log, copysign, isqrt
from sympy import factorint, divisor_sigma, divisors, gcd
from typing import Tuple, Optional
from user import settings
from utility import load_oeis_bfile, analyze_divisors, get_ordinal_suffix

CATEGORY = "Arithmetic and Divisor-based"


@classifier(
    label="Abundant number",
    description="Proper divisors sum > the given number.",
    oeis="A005101",
    category=CATEGORY
)
def is_abundant_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is an abundant number.
    An abundant number is a number for which the sum of its proper divisors
    exceeds the number itself.
    """
    if n < 12:
        return False, None

    _, aliquot, _ = analyze_divisors(n)
    if aliquot > n:
        details = f"Sum of proper divisors: {aliquot} > {n}."
        return True, details
    return False, None


@classifier(
    label="Achilles number",
    description="A powerful number that is not a perfect power.",
    oeis="A052486",
    category=CATEGORY
)
def is_achilles_number(n: int) -> Tuple[bool, str]:
    """
    Returns True if n is an Achilles number, else False.
    An Achilles number is powerful but not a perfect power.
    """
    if n < 72:
        return False, None

    # Factorization: {prime: exponent}
    factors = factorint(n)
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
    label="Amicable number",
    description="Proper divisors sum to m, whose divisors sum to n.",
    oeis="A063990",
    category=CATEGORY
)
def is_amicable_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is an amicable number, else False.
    Amicable numbers are two distinct numbers so that each is the sum of the proper divisors of the other.
    """
    if n < 220:
        return False, None

    a = divisor_sigma(n) - n  # aliquot sum

    # Optimization: if parity doesn't match, can't be an amicable pair
    if n % 2 == 0 and a % 2 != 0:
        return False, None

    if a != n and (divisor_sigma(a) - a) == n:
        return True, f"{n} and {a} are an amicable pair: σ({n})−{n} = {a}, and σ({a})−{a} = {n}."
    return False, None


@classifier(
    label="Deficient number",
    description="Proper divisors sum < the given number.",
    oeis="A005100",
    category=CATEGORY
)
def is_deficient_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a deficient number.
    A deficient number is a number for which the sum of its proper divisors
    is less than the number itself.
    """
    if n < 1:
        return False, None
    _, aliquot, _ = analyze_divisors(n)
    if aliquot < n:
        details = f"Sum of proper divisors: {aliquot} < {n}."
        return True, details
    return False, None


@classifier(
    label="Highly abundant number",
    description="An integer whose sum of divisors is greater than that of any smaller positive integer.",
    oeis="A002093",
    category=CATEGORY
)
@limited_to(149918408641)
def is_highly_abundant_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a highly abundant number (OEIS A005857).
    Uses a preloaded OEIS b-file for performance.
    """
    if n < 1:
        return False, None

    HIGHLYABUNDANT_SET, HIGHLYABUNDANT_LIST = load_oeis_bfile("data/b002093.txt")

    if n in HIGHLYABUNDANT_SET:
        idx = HIGHLYABUNDANT_LIST.index(n)
        sigma_n = divisor_sigma(n, 1)  # Faster!

        if idx > 0:
            prev = HIGHLYABUNDANT_LIST[idx - 1]
            sigma_prev = divisor_sigma(prev, 1)
            details = (
                f"σ({n}) = {sigma_n}, previous record at σ({prev}) = {sigma_prev}"
            )
        else:
            details = f"{n} is the first highly abundant: σ({n}) = {sigma_n}"
        return True, details

    return False, None


@classifier(
    label="Highly composite number",
    description="More divisors than any smaller number.",
    oeis="A002182",
    category=CATEGORY
)
@limited_to(6385128751)
def is_highly_composite_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a highly composite number.
    A highly composite number has more divisors than any smaller positive
    integer.
    """
    # finite list to speed up calculation
    KNOWN = {
      1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260,
      1680, 2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720,
      45360, 50400, 55440, 83160, 110880, 166320, 221760, 277200,
      332640, 498960, 554400, 665280, 720720, 1081080, 1441440,
      2162160, 2882880, 3603600, 4324320, 6486480, 73513440,
      147026880, 294053760, 367567200, 698377680, 1396755360,
      2095133040, 4190266080, 6385128750
    }
    return n in KNOWN


@classifier(
    label="Perfect number",
    description="a positive integer equal to the sum of its proper divisors.",
    oeis="A000396",
    category=CATEGORY
)
def is_perfect_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a perfect number, Example: 28 = 1 + 2 + 4 + 7 + 14
    """
    if n < 1:
        return False, None
    _, aliquot, _ = analyze_divisors(n)
    if aliquot == n:
        details = f"Sum of proper divisors: {aliquot} = {n}."
        return True, details
    return False, None


@classifier(
    label="Perfect power",
    description="An integer of the form m^k, with m > 1, k > 1 (i.e., a square, cube, etc.)",
    oeis="A001597",
    category=CATEGORY
)
def is_perfect_power(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a perfect power (n = m^k, m > 1, k > 1), else False.
    """
    if n < 4:
        return False, None
    for k in range(2, int(log(n, 2)) + 2):  # check all k ≥ 2
        m = round(n ** (1/k))
        if m > 1 and m**k == n:
            return True, f"{n} = {m}^{k}"
    return False, None


@classifier(
    label="Powerful number",
    description="Every prime factor appears with exponent at least 2.",
    oeis="A001694",
    category=CATEGORY
)
def is_powerful_number(n) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a powerful number, else (False, None).
    A number is powerful if every prime factor appears with an exponent ≥ 2.
    """
    if n < 1:
        return False, None
    if n == 1:
        return True, "1 is powerful by definition."

    factors = factorint(n)
    if any(exp < 2 for exp in factors.values()):
        return False, None

    detail = "Prime factorization: " + " × ".join(f"{p}^{exp}" for p, exp in factors.items()) + "."
    return True, detail


@classifier(
    label="Practical number",
    description="All smaller positive integers can be written as sums of distinct divisors of the number.",
    oeis="A005153",
    category=CATEGORY
)
def is_practical_number(n) -> Tuple[bool, str]:
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

    prime_factors = factorint(n)
    # Defensive: Ensure all keys are int (not string)
    try:
        primes = sorted(int(p) for p in prime_factors.keys())
    except Exception:
        return False, None
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
        prev_sigma = sum(divisors(prev_prod))
        check_str = f"• {p} ≤ 1 + σ({prev_prod}) = {1 + prev_sigma}"
        if p > 1 + prev_sigma:
            return False, None
        check_str += " ✓"
        lines.append(check_str)

    lines.append(f"All conditions are satisfied, so {n} is practical.")
    details = "\n".join(lines)
    return True, details


@classifier(
    label="Semiperfect number",
    description="A positive integer that is equal to the sum of a subset of its proper divisors.",
    oeis="A005835",
    category=CATEGORY
)
def is_semiperfect_number(n: int, time_limit: float = 5.0) -> Tuple[[bool], [str]]:
    """
    Returns (True, details) if n is semiperfect, else (False, None), or (None, reason) if undetermined.
    Fast for most n < 10^7 unless highly composite.
    """
    start = time.time()
    D, proper_sum, is_prime = analyze_divisors(n)
    if n < 6:
        # Too small to be semiperfect
        return False, None
    if is_prime:
        # Prime numbers cannot be semiperfect
        return False, None
    if proper_sum < n:
        # Sum of proper divisors less than n
        return False, None
    if proper_sum == n:
        # Perfect number, not semiperfect
        return False, None
    # Proper divisors (sorted high to low for greedy, low to high for DP)
    P = [d for d in D if d < n][::-1]
    if not P:
        # No proper divisors
        return False, None
    # Quick greedy shortcut (good for most cases)
    total = 0
    used = []
    for d in sorted(P, reverse=True):
        if total + d <= n:
            total += d
            used.append(d)
        if total == n:
            subset_str = " + ".join(str(x) for x in sorted(used))
            return True, f"{n} = {subset_str}"
        if time.time() - start > time_limit:
            return None, "Timeout (greedy check)"
    # If greedy fails, try DP if reasonable
    if len(P) <= 150 and n <= 10_000_000:
        dp = [False] * (n + 1)
        dp[0] = True
        prev = [None] * (n + 1)
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
    # Too many divisors or n too large for safe DP
    return None, f"Too many divisors ({len(P)}) or n too large for fast check"


@classifier(
    label="Sociable number",
    description="Chains of aliquot sums cycle back to start.",
    oeis="A003416",
    category=CATEGORY
)
@limited_to(999999)
def is_sociable_number(n: int, max_cycle_len=28) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a sociable number of cycle length <= max_cycle_len.
    Returns False otherwise.
    """
    if n < 2:
        return False, None
    seq = [n]
    current = n
    for _ in range(max_cycle_len):
        # Aliquot sum: sum of proper divisors
        next_n = sum(d for d in divisors(current) if d < current)
        if next_n == 0:
            return False, None
        if next_n == n:
            # Sociable cycle found!
            cycle_len = len(seq)
            # For details, print the cycle
            cycle_str = " → ".join(str(x) for x in seq + [n])
            details = f"Sociable cycle of length {cycle_len}: {cycle_str}"
            return True, details
        if next_n in seq:
            # We hit a repeat, not sociable (maybe stuck at an earlier loop)
            return False, None
        seq.append(next_n)
        current = next_n
    # Did not close within max_cycle_len
    return False, None


@classifier(
    label="Sphenic number",
    description="A positive integer that is the product of exactly three distinct primes.",
    oeis="A007304",
    category=CATEGORY
)
def is_sphenic_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is sphenic (product of 3 distinct primes), else False.
    """
    if n < 30:  # smallest sphenic number is 2*3*5=30
        return False, None
    f = factorint(n)
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
def is_squarefree_number(n) -> Tuple[bool, str]:
    """
    Returns True if n is squarefree (no repeated prime factors).
    Details: lists the prime exponents.
    """
    if n < 1:
        return False, None
    factors = factorint(n)
    for p, exp in factors.items():
        if exp > 1:
            detail = f"{n} is divisible by {p}^2 = {p**2}"
            return (False, detail)
    detail = " × ".join([str(p) for p in sorted(factors)]) if factors else "1"
    return (True, f"{n} = {detail} is squarefree (no repeated prime factors).")


def format_sum_of_powers(n: int, *terms, power=3) -> str:
    """
    Returns a string like: 42 = (-5)^3 + 6^3 + 7^3
    Parentheses around negative numbers only.
    No redundant '+' on the first term.
    """
    def fmt(t):
        return f"({t})^{power}" if t < 0 else f"{t}^{power}"
    s = f"{n} = " + " + ".join(fmt(t) for t in terms)
    return s


@classifier(
    label="Sublime number",
    description="A number with a perfect number of divisors, and the sum of those divisors is also perfect.",
    oeis="A081734",
    category=CATEGORY
)
def is_sublime_number(n: int) -> Tuple[bool, str]:
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
    label="Sum of 2 cubes",
    description="Can be written as n = a³ + b³ for integers a, b.",
    oeis="A025475",
    category=CATEGORY
)
@limited_to(9999999999)
def is_sum_of_2_cubes(n: int, max_results: int = None) -> Tuple[bool, str]:
    """
    Returns (True, details) if n can be written as a³ + b³ for integers a, b.
    Shows up to max_results results (None = all), smallest |a|,|b| first, unordered a ≤ b.
    Honors settings.ALLOW_ZERO_IN_DECOMP: if False, excludes any result with a==0 or b==0.
    """
    allow_zero = getattr(settings, "ALLOW_ZERO_IN_DECOMP", False)
    cap = getattr(settings, "MAX_SOLUTIONS_SUM_OF", {}).get("2_CUBES", None)

    absn = abs(n)
    B = int(round(absn ** (1/3))) + 2  # simple bound; adjust if you have a better one

    # precompute a^3 map: value -> list of a (capture both signs)
    cubes = {}
    for a in range(-B, B + 1):
        cubes.setdefault(a*a*a, []).append(a)

    seen = set()   # canonical pairs (a≤b) over integers
    listed = []

    for x in range(-B, B + 1):
        need = n - x*x*x
        ys = cubes.get(need, [])
        for y in ys:
            a, b = (x, y) if x <= y else (y, x)
            if allow_zero or (a != 0 and b != 0):
                tup = (a, b)
                if tup not in seen:
                    seen.add(tup)
                    if cap is None or len(listed) < cap:
                        listed.append(tup)

    if not seen:
        return False, None

    header = f"found {len(seen)}"
    if cap is not None and len(listed) < len(seen):
        header += f" (showing first {len(listed)})"

    tail = "; ".join(f"{a}³+{b}³" for (a, b) in listed)
    details = f"{header}: {tail}" if listed else header
    return True, details


def _icbrt_exact(m: int) -> Optional[int]:
    """
    Exact integer cube-root test.
    Returns x if x^3 == m, else None. Works for negative m as well.
    """
    if m == 0:
        return 0
    neg = m < 0
    a = -m if neg else m
    # integer cube root via Newton's method (fast, no floats)
    x = int(round(a ** (1/3)))  # good initial guess
    # polish to exact (avoid float drift)
    # adjust up/down until x^3 crosses a
    while x > 0 and x**3 > a:
        x -= 1
    while (x + 1)**3 <= a:
        x += 1
    if x**3 != a:
        return None
    return -x if neg else x


@classifier(
    label="Sum of 3 cubes",
    description="Can be written as n = a³ + b³ + c³ for integers a, b, c.",
    oeis="A003072",
    category=CATEGORY,
)
def is_sum_of_3_cubes(n: int, max_results: int = None) -> Tuple[bool, str]:
    """
    Show up to max_results found solutions (canonical a<=b<=c), and ALWAYS show
    the known 'celebrity' solution if available, even when capped.
    """

    # trivial / modular obstruction
    if n == 0:
        return False, None
    if n % 9 in (4, 5):
        return False, None

    allow_zero = getattr(settings, "ALLOW_ZERO_IN_DECOMP", True)
    if max_results is None:
        max_results = getattr(settings, "MAX_SOLUTIONS_SUM_OF", {}).get("3_CUBES", 20)
    B = getattr(settings, "MAX_ABS_FOR_SUM_OF_3_CUBES", 100)

    seen = set()    # canonical triples (a<=b<=c)
    listed = []     # what we print (respect cap when adding *found* ones)

    # search with double loop + exact cube-root for c
    for a in range(-B, B + 1):
        a3 = a*a*a
        for b in range(a, B + 1):
            need = n - a3 - b*b*b
            c = _icbrt_exact(need)
            if c is None:
                continue
            if b <= c and (allow_zero or (a != 0 and b != 0 and c != 0)):
                tup = (a, b, c)
                if tup not in seen:
                    seen.add(tup)
                    if (max_results is None) or (len(listed) < max_results):
                        listed.append(tup)

    # inject celebrity (canonicalize and ensure it is visible)
    celeb = None
    if n in SUM_OF_THREE_CUBES_SOLUTIONS:
        raw = SUM_OF_THREE_CUBES_SOLUTIONS[n]
        celeb = tuple(sorted(raw))
        if (allow_zero or all(v != 0 for v in celeb)) and celeb not in seen:
            seen.add(celeb)
            if max_results is None:
                listed.append(celeb)
            else:
                if len(listed) < max_results:
                    listed.append(celeb)
                else:
                    # ensure celebrity appears even when cap reached
                    listed[-1] = celeb

    if not seen:
        return False, None

    def fmt(x: int) -> str:
        return f"({x})³" if x < 0 else f"{x}³"

    # mark celebrity in the text (optional but helpful)
    celeb_tag = (n in SUM_OF_THREE_CUBES_SOLUTIONS)
    parts = []
    for t in listed:
        s = f"{fmt(t[0])} + {fmt(t[1])} + {fmt(t[2])}"
        parts.append(s)

    details = f"{n} = " + "; ".join(parts) if parts else f"found {len(seen)}"
    return True, details


def r2_sum_of_two_squares(factors: dict[int, int]) -> int:
    prod = 1
    for p, e in factors.items():
        if p % 4 == 3 and (e % 2 == 1):
            return 0
        if p % 4 == 1:
            prod *= (e + 1)
    return 4 * prod


@classifier(
    label="Sum of 2 squares",
    description="Can be written as n = a² + b² for integers a, b.",
    oeis="A001481",
    category=CATEGORY
)
def is_sum_of_2_squares(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n can be written as a^2 + b^2 for integers a, b.
    Shows up to max_results decompositions with a >= 0, b >= 0 and a <= b.
    Uses Fermat's theorem for quick check.
    """
    if n < 0:
        return False, None

    allow_zero = getattr(settings, "ALLOW_ZERO_IN_DECOMP", False)
    cap = getattr(settings, "MAX_SOLUTIONS_SUM_OF", {}).get("2_SQUARES", None)

    N = n
    # exact total (ordered, with signs)
    total_ordered = r2_sum_of_two_squares(factorint(N))
    if total_ordered == 0:
        return False, None

    # enumerate canonical nonnegative pairs 0<=x<=y
    listed = []
    seen_count = 0  # number of canonical solutions found
    r = isqrt(N)
    for x in range(0, r + 1):
        y2 = N - x*x
        if y2 < 0:
            break
        y = isqrt(y2)
        if x*x + y*y == N and x <= y:
            if allow_zero or (x > 0 and y > 0):
                seen_count += 1
                if cap is None or len(listed) < cap:
                    listed.append((x, y))

    if seen_count == 0 and not allow_zero:
        # total_ordered>0 but all reps use a zero term and zeros are disallowed
        return False, None

    header = f"total {total_ordered} (ordered, with signs)"
    if cap is not None and len(listed) < seen_count:
        header += f" (showing first {len(listed)} canonical)"
    if listed:
        total = len(listed) * 8
        canonical_count = len(listed)
        tail = "; ".join(f"{x}²+{y}²" for x, y in listed)
        details = (f"total {total} (ordered, with signs), "
                   f"{canonical_count} canonical: {tail}")
    else:
        details = f"total 0 (ordered, with signs), 0 canonical"

    return True, details


def _legendre_three_square_possible(n: int) -> bool:
    """Legendre: n is sum of three squares iff n ≠ 4^a(8b+7)."""
    if n < 0:
        return False
    while n % 4 == 0:
        n //= 4
    return (n % 8) != 7


@classifier(
    label="Sum of 3 squares",
    description="Can be written as n = a² + b² + c² for integers a, b, c.",
    oeis="A000378",
    category=CATEGORY
)
@limited_to(9999999999)
def is_sum_of_3_squares(n: int) -> Tuple[bool, str]:
    """
    Fast check with Legendre's theorem; finds up to cap canonical triples (x≤y≤z), x,y,z≥0
    unless ALLOW_ZERO_IN_DECOMP is False (then x,y,z>0).
    """
    if n < 0:
        return False, None
    N = n

    # Legendre quick reject
    if not _legendre_three_square_possible(N):
        return False, None

    allow_zero = getattr(settings, "ALLOW_ZERO_IN_DECOMP", False)
    cap = getattr(settings, "MAX_SOLUTIONS_SUM_OF", {}).get("3_SQUARES", 20)

    r = isqrt(N)

    # Precompute squares once
    squares = [i * i for i in range(r + 1)]

    seen = set()   # canonical (x,y,z)
    listed = []

    # Iterate z from large to small helps find solutions quickly for many N
    z_start = r
    z_min = 0 if allow_zero else 1

    for z in range(z_start, z_min - 1, -1):
        z2 = squares[z]
        T = N - z2
        if T < 0:
            continue

        # Quick modular pruning for x^2 + y^2 = T
        # Squares mod 4 ∈ {0,1} ⇒ sums mod 4 ∈ {0,1,2}; skip T≡3 (mod 4).
        if (T & 3) == 3:
            continue
        # Squares mod 8 ∈ {0,1,4} ⇒ sums mod 8 ∈ {0,1,2,4,5}; prune others.
        mod8 = T & 7
        if mod8 not in (0, 1, 2, 4, 5):
            continue

        # Two-pointer over squares to solve x^2 + y^2 = T
        i = 0 if allow_zero else 1
        j = isqrt(T)
        if j > r:
            j = r  # clamp

        while i <= j:
            s = squares[i] + squares[j]
            if s == T:
                x, y = i, j
                if allow_zero or (x > 0 and y > 0 and z > 0):
                    # canonicalize x<=y<=z guaranteed by loop order and i<=j, but assert anyway
                    if x <= y <= z:
                        tup = (x, y, z)
                        if tup not in seen:
                            seen.add(tup)
                            if cap is None or len(listed) < cap:
                                listed.append(tup)
                            # Early exit if we reached the display cap
                            if cap is not None and len(listed) >= cap:
                                break
                # Move both pointers to find other pairs
                i += 1
                j -= 1
            elif s < T:
                i += 1
            else:
                j -= 1

        if cap is not None and len(listed) >= cap:
            break

    if not seen:
        return False, None

    header = f"found {len(seen)}"
    if cap is not None and len(listed) < len(seen):
        header += f" (showing first {len(listed)})"

    tail = "; ".join(f"{x}²+{y}²+{z}²" for (x, y, z) in listed)
    details = f"{header}: {tail}" if listed else header
    return True, details


@classifier(
    label="Superabundant number",
    description="The ratio of sum of divisors to the number is greater than for any smaller number.",
    oeis="A004394",
    category=CATEGORY
)
@limited_to(25484247877474623694559469201315033045359474150161923076850486576760360768000)
def is_superabundant_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a superabundant number.
    Uses a list loaded from OEIS for speed.
    """
    if n < 1:
        return False, None
    SUPERABUNDANT_SET, SUPERABUNDANT_LIST = load_oeis_bfile("data/b004394.txt")

    if n in SUPERABUNDANT_SET:
        idx = SUPERABUNDANT_LIST.index(n)
        sigma_n = sum(divisors(n))
        ratio_n = sigma_n / n
        if idx > 0:
            prev = SUPERABUNDANT_LIST[idx - 1]
            sigma_prev = sum(divisors(prev))
            ratio_prev = sigma_prev / prev
            details = (
                f"{n} is superabundant: σ({n})={sigma_n}, σ({n})/{n}={ratio_n:.5f}; "
                f"previous record at {prev}: σ({prev})/{prev}={ratio_prev:.5f}"
            )
        else:
            details = (
                f"{n} is the first superabundant: σ({n})={sigma_n}, σ({n})/{n}={ratio_n:.5f}"
            )
        return True, details
    return False, None

    # --- Original slow algorithm for reference ---
    # def sigma_ratio(n):
    #     return divisor_sigma(n) / n
    # ratios = [sigma_ratio(k) for k in range(1, n)]
    # this_ratio = sigma_ratio(n)
    # return all(this_ratio > r for r in ratios)


@classifier(
    label="Triperfect number",
    description="A number n whose sum of divisors sigma(n) equals 3n.",
    oeis="A007539",
    category=CATEGORY
)
def is_triperfect_number(n) -> Tuple[bool, str]:
    TRIPERFECT_NUMBERS = {
        120,
        672,
        523776,
        459818240,
        1476304896,
        51001180160,
        89068662763520,
        153722867280912930,
    }
    if n in TRIPERFECT_NUMBERS:
        return True, f"Sum of divisors (sigma) = {divisor_sigma(n)} = 3×{n}"
    return False, None


@classifier(
    label="Untouchable number",
    description="Cannot be expressed as the sum of the proper divisors of any number.",
    oeis="A005114",
    category=CATEGORY
)
@limited_to(100006)
def is_untouchable_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is untouchable, else (False, None).
    Uses precomputed OEIS b-file for n < 100000, else falls back to slow calculation if allowed.
    """
    if n < 2:
        return False, None

    # Load up to 99998 from b-file (precompute for speed)
    UNTOUCHABLE_SET, UNTOUCHABLE_LIST = load_oeis_bfile("data/b005114.txt")
    PRECOMP_LIMIT = max(UNTOUCHABLE_SET) + 1

    # Fast path for precomputed range
    if n < PRECOMP_LIMIT:
        if n in UNTOUCHABLE_SET:
            index = UNTOUCHABLE_LIST.index(n) + 1  # 1-based
            details = f"{n} is the {get_ordinal_suffix(index)} untouchable number."
            if n == 5:
                details += " The only known odd untouchable number."
            return True, details
        else:
            return False, None  # If not in b-file, definitely not untouchable (fast!)

    # For larger n: optionally allow slow calculation if user enables it
    allow_slow = getattr(settings, "ALLOW_SLOW_CALCULATIONS", False)
    if not allow_slow:
        return False, None

    # SLOW calculation fallback for n >= PRECOMP_LIMIT
    limit = n * 10
    for k in range(2, limit):
        if divisor_sigma(k, 1) - k == n:
            return False, None  # n is touchable
    details = f"No integer k < {limit} has aliquot sum = {n}. {n} is untouchable."
    return True, details


@classifier(
    label="Weird number",
    description="Abundant but not semiperfect.",
    oeis="A006037",
    category=CATEGORY
)
@limited_to(999999)
def is_weird_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a weird number.
    A weird number is abundant (the sum of its proper divisors is greater than
    itself), but not semiperfect (no subset of divisors sums to n).
    """
    if n < 70:
        return False, None
    # 1) Abundance test
    _, aliquot, _ = analyze_divisors(n)
    if aliquot <= n:
        return False, None

    # 2) Semiperfect test
    semip, _ = is_semiperfect_number(n)
    if semip:
        return False, None

    # 3) Weird: abundant + not semiperfect
    details = f"Sum of proper divisors: {aliquot} > {n}; no subset of divisors sums to {n}."
    return True, details
