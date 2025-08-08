# -----------------------------------------------------------------------------
#  Arithmetic and Divisor-based test functions
# -----------------------------------------------------------------------------

import time

from decorators import classifier, limited_to
from functools import reduce
from math import log
from sympy import factorint, divisor_sigma, divisors, gcd
from typing import Tuple
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
    description="Poper divisors sum to m, whose divisors sum to n.",
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
    if n == 0:
        return False, None
    if max_results is None:
        max_results = getattr(settings, "MAX_SOLUTIONS_SUM_OF", {}).get("2_CUBES", None)

    allow_zero = getattr(settings, "ALLOW_ZERO_IN_DECOMP", True)

    lim = int(abs(n) ** (1/3)) + 3  # search window

    results = []
    seen = set()
    for a in range(-lim, lim + 1):
        for b in range(a, lim + 1):  # a <= b avoids repeats
            if a**3 + b**3 == n:
                if not allow_zero and (a == 0 or b == 0):
                    continue
                pair = (a, b)
                if pair not in seen:
                    seen.add(pair)
                    results.append(pair)
            if max_results is not None and len(results) >= max_results:
                break
        if max_results is not None and len(results) >= max_results:
            break

    if results:
        prefix = f"{n} = "
        details = prefix + "; ".join(
            f"{a}³ + {b}³" for (a, b) in results
        )
        return True, details
    return False, None


SUM_OF_THREE_CUBES_SOLUTIONS = {  # known hard solutions < 1000
    3: (5699368212219623807203, -569936821113563493509, -472715493453327032),
    33: (8866128975287528, -8778405442862239, -2736111468807040),
    42: (80435758145817515, 80538738812075974, -12602123297335631),
    74: (-284650292555885, 66229832190556, 283450105697727),
    # 114: (unsolved (July 2025), search area ∣z∣ ≤ 10^16)
    165: (383344975542639445, -385495523231271884, 98422560467622814),
    # 390: (unsolved (July 2025), search area ∣z∣ ≤ 10^16)
    579: (143075750505019222645, -143070303858622169975, -6941531883806363291),
    # 627: (unsolved (July 2025), search area ∣z∣ ≤ 10^16)
    # 633: (unsolved (July 2025), search area ∣z∣ ≤ 10^16)
    # 732: (unsolved (July 2025), search area ∣z∣ ≤ 10^16)
    795: (-14219049725358227, 14197965759741571, 2337348783323923),
    906: (-74924259395619397, 72054089679353378, 35961979615356503),
    # 921: (unsolved (July 2025), search area ∣z∣ ≤ 10^16)
    # 975: (unsolved (July 2025), search area ∣z∣ ≤ 10^16)
}


@classifier(
    label="Sum of 3 cubes",
    description="Can be written as n = a³ + b³ + c³ for integers a, b, c.",
    oeis="A003072",
    category=CATEGORY
)
@limited_to(9999999999)
def is_sum_of_3_cubes(n: int, max_results: int = None) -> Tuple[bool, str]:
    """
    Returns (True, details) if n can be written as a³ + b³ + c³ for integers a, b, c.
    Brute-forces up to a limit; always adds hardcoded "celebrity" solution if available.
    Shows up to max_results results (None = all).
    """
    if n == 0:
        return False, None
    if n % 9 in (4, 5):
        return False, None

    allow_zero = getattr(settings, "ALLOW_ZERO_IN_DECOMP", True)

    if max_results is None:
        max_results = getattr(settings, "MAX_SOLUTIONS_SUM_OF", {}).get("3_CUBES", None)

    limit = getattr(settings, "MAX_ABS_FOR_SUM_OF_3_CUBES", 100)  # make limit configurable
    results = []
    seen = set()
    for a in range(-limit, limit + 1):
        for b in range(a, limit + 1):
            for c in range(b, limit + 1):
                if a**3 + b**3 + c**3 == n:
                    if not allow_zero and (a == 0 or b == 0 or c == 0):
                        continue
                    tup = (a, b, c)
                    if tup not in seen:
                        seen.add(tup)
                        results.append(tup)
                if max_results is not None and len(results) >= max_results:
                    break
            if max_results is not None and len(results) >= max_results:
                break
        if max_results is not None and len(results) >= max_results:
            break

    # Always include known hard solution if not found by brute-force
    celeb_tuple = None
    if n in SUM_OF_THREE_CUBES_SOLUTIONS:
        celeb = tuple(sorted(SUM_OF_THREE_CUBES_SOLUTIONS[n]))
        if celeb not in seen:
            celeb_tuple = celeb

    def cube_fmt(x):
        return f"({x})³" if x < 0 else f"{x}³"

    details_list = []
    if results:
        details_list.extend(
            f"{n} = {cube_fmt(a)} + {cube_fmt(b)} + {cube_fmt(c)}"
            for (a, b, c) in results
        )
    if celeb_tuple:
        details_list.append(
            f"{n} = {cube_fmt(celeb_tuple[0])} + {cube_fmt(celeb_tuple[1])} + {cube_fmt(celeb_tuple[2])}"
        )

    if details_list:
        # Compact style: prefix, then joined
        prefix_eq = f"{n} ="
        exprs = [expr.split("=", 1)[1].strip() for expr in details_list]
        details = f"{prefix_eq} " + "; ".join(exprs)
        return True, details
    return False, None


@classifier(
    label="Sum of 2 squares",
    description="Can be written as n = a² + b² for integers a, b.",
    oeis="A001481",
    category=CATEGORY
)
def is_sum_of_2_squares(n: int, max_results: int = None) -> Tuple[bool, str]:
    """
    Returns (True, details) if n can be written as a^2 + b^2 for integers a, b.
    Shows up to max_results decompositions with a >= 0, b >= 0 and a <= b.
    Uses Fermat's theorem for quick check.
    """
    if max_results is None:
        max_results = getattr(settings, "MAX_SOLUTIONS_SUM_OF", {}).get("2_SQUARES", None)
    if n < 1:
        return False, None

    for p, e in factorint(n).items():
        if p % 4 == 3 and e % 2 != 0:
            return False, None

    results = []
    seen = set()
    lim = int(n**0.5) + 1
    for a in range(0, lim):
        b2 = n - a*a
        if b2 < 0:
            break
        b = int(b2**0.5)
        if b < a:
            continue
        if b*b == b2:
            pair = (a, b)
            if pair not in seen:
                seen.add(pair)
                results.append(pair)
            if max_results is not None and len(results) >= max_results:
                break

    if results:
        details = f"{n} = " + "; ".join(
            f"{a}² + {b}²" for (a, b) in results
        )
        return True, details
    else:
        return False, None


@classifier(
    label="Sum of 3 squares",
    description="Can be written as n = a² + b² + c² for integers a, b, c.",
    oeis="A000378",
    category=CATEGORY
)
@limited_to(9999999999)
def is_sum_of_3_squares(n: int, max_decomps: int = None) -> Tuple[bool, str]:
    """
    Returns (True, details) if n can be written as a² + b² + c² for integers a, b, c.
    Honors settings.ALLOW_ZERO_IN_DECOMP. Shows up to max_decomps decompositions (unordered).
    """
    allow_zero = getattr(settings, "ALLOW_ZERO_IN_DECOMP", True)
    if max_decomps is None:
        max_decomps = getattr(settings, "MAX_SOLUTIONS_SUM_OF", {}).get("3_SQUARES", None)
    if n < 1:
        return False, None

    m = n
    while m % 4 == 0:
        m //= 4
    if m % 8 == 7:
        return False, None

    results = []
    seen = set()
    lim = int(n ** 0.5) + 1
    start = 0 if allow_zero else 1

    for a in range(start, lim):
        na = n - a*a
        if na < 0:
            break
        for b in range(a, lim):
            nb = na - b*b
            if nb < 0:
                break
            c2 = nb
            c = int(c2 ** 0.5)
            if c < b:
                continue
            if c*c == c2:
                if not allow_zero and (a == 0 or b == 0 or c == 0):
                    continue
                tup = (a, b, c)
                if tup not in seen:
                    seen.add(tup)
                    results.append(tup)
                if max_decomps is not None and len(results) >= max_decomps:
                    break
        if max_decomps is not None and len(results) >= max_decomps:
            break

    if results:
        details = f"{n} = " + "; ".join(
            f"{a}² + {b}² + {c}²" for (a, b, c) in results
        )
        return True, details
    else:
        return False, None


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
