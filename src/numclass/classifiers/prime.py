# -----------------------------------------------------------------------------
#  prime.py
#  Prime and Prime-related test functions
# -----------------------------------------------------------------------------

from __future__ import annotations

from itertools import permutations

from sympy import isprime, nextprime, prevprime, prime, primepi

from numclass.context import NumCtx
from numclass.fmt import abbr_int_fast, format_factorization
from numclass.registry import classifier
from numclass.runtime import CFG
from numclass.utility import build_ctx, check_oeis_bfile, ctx_isprime, dec_digits, get_ordinal_suffix

CATEGORY = "Primes and Prime-related Numbers"


"""
Atomic prime test definitions only, intersection classes are dynamically generated.
Intersection exceptions are: Motzkin prime, Catalan prime and Keith prime
which have a fast lookup list.
"""


@classifier(
    label="Absolute prime",
    description=" All digit permutations are prime.",
    oeis="A003459",
    category=CATEGORY,
    limit=10**8-1
)
def is_absolute_prime(n: int):
    """
    Check if n is an absolute (permutable) prime.
    Details show all unique permutations and whether they're prime.
    """

    if n < 2:
        return False, None

    if n in {2, 3, 5, 7}:
        return True, "single-digit absolute prime"

    s = str(n)

    # quick digit filter (kills most candidates fast)
    if any(d in s for d in "024568"):
        return False, "has 0/even/5 digit → some permutation is composite"

    # optional: also prune by last digit rule
    if s[-1] not in "1379":
        return False, "ends with non-prime-possible digit"

    # check all unique, same-length permutations
    seen = set()
    for p in set(permutations(s)):
        if p[0] == '0':
            # presence of 0 already rejected above, but keep this guard if you drop that early reject
            return False, "permutation with leading zero is invalid"
        m = int("".join(p))
        if m not in seen:
            seen.add(m)
            if not isprime(m):
                return False, f"permutation {m} is composite"

    return True, "all same-length permutations are prime"


@classifier(
    label="Balanced prime",
    description="equal to the average of the nearest primes: p=(p₁+p₂)/2, where p₁, p₂ are the adjacent primes.",
    oeis="A006562",
    category=CATEGORY,
    limit=10**8-1
)
def is_balanced_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check if n is a balanced prime.
    Details show previous and next prime and the average.
    """
    if n < 2:
        return False, None
    if not ctx_isprime(n, ctx):
        return False, None
    i = primepi(n)
    if i < 2:
        return False, None  # Need one before and one after
    prev_prime = prime(i - 1)
    next_prime = prime(i + 1)
    if n == (prev_prime + next_prime) // 2:
        details = f"{n} is the average of previous prime {prev_prime} and next prime {next_prime}."
        return True, details
    return False, None


@classifier(
    label="Both-truncatable prime",
    description="Both left- and right-truncatable.",
    oeis="A020994",
    category=CATEGORY,
    limit=10**8-1
)
def is_both_truncatable_prime(n: int) -> tuple[bool, str]:
    """
    Check if n is both left- and right-truncatable prime.
    These are primes that remain prime when digits are removed from
    either the left or the right, one at a time, show both paths.
    """
    if n < 10:
        return False, None
    left = is_left_truncatable_prime(n)
    right = is_right_truncatable_prime(n)
    if isinstance(left, tuple) and left[0] and isinstance(right, tuple) and right[0]:
        details = f"left: {left[1]}, right: {right[1]}"
        return True, details
    return False, None


@classifier(
    label="Catalan prime",
    description="A Catalan number that is prime.",
    oeis="A000108",
    category=CATEGORY
)
def is_catalan_prime(n: int) -> tuple[bool, str]:
    """
    Check if n is a Catalan prime.
    A Catalan prime is a Catalan number that is also prime.
    Only two such numbers are known: 2 and 5.
    """
    KNOWN = {2, 5}
    if n in KNOWN:
        return True, f"{n} is a Catalan number and also a prime."
    return False, None


@classifier(
    label="Chen prime",
    description="Prime n where n+2 is prime or semiprime.",
    oeis="A109611",
    category=CATEGORY,
    limit=10**20-1
)
def is_chen_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Chen prime: n is prime and n+2 is either prime or semiprime (two prime factors counted with multiplicity).
    """
    if n < 2 or not ctx_isprime(n, ctx):
        return False, None

    q = n + 2

    # Fast path: twin-prime case
    if isprime(q):
        return True, f"{n} is a Chen prime: p+2 = {q} is prime (twin prime)."

    # Otherwise, check if q is semiprime via its factorization context
    ctx_q = build_ctx(q)              # factor q once, reuse its fac
    fac_q = ctx_q.fac                 # dict {prime: exponent}
    if sum(fac_q.values()) == 2:      # exactly two prime factors with multiplicity
        return True, f"{n} is a Chen prime: p+2 = {q} is semiprime ({format_factorization(fac_q)})."

    return False, None


@classifier(
    label="Circular prime",
    description="All digit rotations are prime.",
    oeis="A068652",
    category=CATEGORY,
    limit=10**20-1
)
def is_circular_prime(n: int) -> tuple[bool, str]:
    """
    Check if n is a circular prime, else False.
    Details show all rotations.
    """
    if n < 2:
        return False, None
    s = str(n)
    rotations = [int(s[i:] + s[:i]) for i in range(len(s))]
    are_all_primes = all(isprime(r) for r in rotations)

    if are_all_primes:
        rot_str = ", ".join(str(r) for r in rotations)
        details = f"All rotations are prime: {rot_str}"
        return True, details
    return False, None


@classifier(
    label="Cousin prime",
    description="Prime with another prime 4 away.",
    oeis="A046132",  # smaller member, A046132 for larger mmber
    category=CATEGORY
)
def is_cousin_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check if n is a cousin prime, else False.
    Details show the cousin prime pair.
    """
    if n < 2:
        return False, None
    if not ctx_isprime(n, ctx):
        return False, None
    if isprime(n - 4):
        details = f"{n} and {n-4} are cousin primes (differ by 4)."
        return True, details
    if isprime(n + 4):
        details = f"{n} and {n+4} are cousin primes (differ by 4)."
        return True, details
    return False, None


@classifier(
    label="emirp",
    description="Prime that is a different prime when reversed.",
    oeis="A006567",
    category=CATEGORY
)
def is_emirp(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check if n is an emirp prime.
    Details show the reversal and primality of both n and its reverse.
    """
    if n < 13:
        return False, None
    r = int(str(n)[::-1])
    if r != n and ctx_isprime(n, ctx) and isprime(r):
        details = f"{n} is prime, reverse is {r}, which is also prime."
        return True, details
    return False, None


@classifier(
    label="Factorial prime",
    description="Prime of the form k! ± 1",
    oeis="A002981",  # for k! + 1, A002982 for k! - 1
    category=CATEGORY,
    limit=1_307_674_368_002
)
def is_factorial_prime(n: int) -> tuple[bool, str]:
    """
    Return (True, msg) if n is a factorial prime (m! ± 1 and prime).
    Message includes rank by size, decomposition n = m! ± 1, and subtype.
    """

    # Smallest known factorial primes (both +1 and −1 subtypes), sorted by value.
    FACTORIAL_PRIMES = sorted([
        2, 3, 5, 7, 23, 719, 5039, 39916801,
        479001599, 6227020801, 87178291199, 1307674368001,
    ])

    if n not in FACTORIAL_PRIMES:
        return False, None

    # Rank by increasing value (k-th smallest)
    idx = FACTORIAL_PRIMES.index(n) + 1

    # Recover m and the ±1 subtype
    # We iterate factorials until we pass n+1.
    m = 1
    fact = 1  # 1! = 1
    m_found = None
    sign = 0  # +1 or -1

    # Loop invariant: fact == m!
    while fact <= n + 1:
        if n == fact + 1:
            m_found, sign = m, +1
            break
        if n == fact - 1:
            m_found, sign = m, -1
            break
        m += 1
        fact *= m  # (m)! from (m-1)! * m

    # Compose detail string
    if m_found is not None:
        sign_char = "+" if sign > 0 else "−"  # U+2212 minus
        subtype = "+1 subtype" if sign > 0 else "−1 subtype"
        msg = (
            f"{get_ordinal_suffix(idx)} smallest factorial prime; "
            f"{n} = {m_found}! {sign_char} 1 ({subtype}; m={m_found})"
        )
    else:
        # Fallback (shouldn't happen for the list above)
        msg = f"{get_ordinal_suffix(idx)} smallest factorial prime."

    return True, msg


@classifier(
    label="Fermat prime",
    description="Prime of form 2^(2^k)+1.",
    oeis="A019434",
    category=CATEGORY
)
def is_fermat_prime(n: int):
    """
    Check if n is a Fermat prime (of form 2^(2^k) + 1).
    """

    FERMAT_PRIMES = sorted([
        3, 5, 17, 257, 65537  # Only 5 known, F5 through F33 are composite
    ])

    if n in FERMAT_PRIMES:
        idx = FERMAT_PRIMES.index(n) + 1
        return True, f"{n} is the {get_ordinal_suffix(idx)} known Fermat prime. Only 5 such primes are known."
    return False, None


@classifier(
    label="Gaussian prime",
    description="Prime in the ring of Gaussian integers.",
    oeis="A055025",
    category=CATEGORY
)
def is_gaussian_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check whether n is a Gaussian prime in Z[i] lying on the real axis.
    Details include the type congruent to 3 mod 4.
    """
    if n < 2:
        return False, None
    # Prime and 3 mod 4
    if ctx_isprime(n, ctx) and n % 4 == 3:
        abbr = abbr_int_fast(n)
        details = f"{abbr} is a (rational) prime and {abbr} ≡ 3 mod 4; thus, a Gaussian prime."
        return True, details

    return False, None


@classifier(
    label="Good prime",
    description=(
        "Good prime (global): prime p_n satisfying "
        "p_n² > p_{n-i} × p_{n+i} for all 1 ≤ i ≤ n-1."
    ),
    oeis="A028388",
    category=CATEGORY,
    limit=10**7 - 1,
)
def is_good_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Global good prime (OEIS A028388).
    """
    if n < 2 or not ctx_isprime(n, ctx):
        return False, None

    i = primepi(n)
    if i < 2:
        return False, None  # need neighbors

    n2 = n * n

    # Check symmetric neighbors p_{i-j}, p_{i+j}
    for j in range(1, i):
        if n2 <= prime(i - j) * prime(i + j):
            return False, None

    details = (
        f"{n} is p_{i}. Verified p_n² > p_{{n-i}}×p_{{n+i}} "
        f"for all 1 ≤ i ≤ {i-1}×(global good prime)."
    )
    return True, details


@classifier(
    label="Good prime (local)",
    description=(
        "Local good prime: prime p_n such that "
        "p_n² > p_{n-1}×p_{n+1}."
    ),
    oeis="A046869",
    category=CATEGORY,
    limit=10**7 - 1,
)
def is_good_prime_local(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Local good prime (OEIS A046869).
    """
    if n < 2 or not ctx_isprime(n, ctx):
        return False, None

    i = primepi(n)
    if i < 2:
        return False, None

    p_prev = prime(i - 1)
    p_next = prime(i + 1)

    if n * n > p_prev * p_next:
        details = (
            f"{n}² = {n*n} > {p_prev}·{p_next} "
            f"(local good prime condition)."
        )
        return True, details

    return False, None


@classifier(
    label="Isolated prime",
    description="Prime number that is not part of a twin prime pair (n-2 and n+2 are not prime).",
    oeis="A007510",
    category=CATEGORY
)
def is_isolated_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Returns (True, details) if n is an isolated prime: neither n-2 nor n+2 is prime.
    """
    if n < 2 or not ctx_isprime(n, ctx):
        return False, None
    prev_is_prime = isprime(n - 2)
    next_is_prime = isprime(n + 2)
    abbr = abbr_int_fast(n)
    if not prev_is_prime and not next_is_prime:
        details = f"{abbr} is prime, but {abbr}-2 and {abbr}+2 are not."
        return True, details
    return False, None


@classifier(
    label="Keith prime",
    description=("An integer > 9 that is both a Keith number and a prime number, "
                 "using the standard base-10 digit-recurrence rule."),
    oeis="A048970",
    category=CATEGORY,
    limit=74596893730428
)
def is_keith_prime(n: int) -> tuple[bool, str]:
    """
    Returns (True, details) if n is a Keith prime, else False.
    Keith primes are very rare: sequence based on the strictbase-10
    digit-reocurrence definition:
    A Keith prime is an integer > 9 that is both a Keith number and
    a prime number, using the standard base-10 digit-recurrence rule.
    This means that a Keith prime is not just an intersection between a
    Keith number and a prime.
    """

    KEITH_PRIMES = sorted([
        19, 47, 61, 197, 1084051, 74596893730427
    ])

    if n in KEITH_PRIMES:
        idx = KEITH_PRIMES.index(n) + 1
        return True, f"{n} is the {get_ordinal_suffix(idx)} known Keith prime."
    return False, None


@classifier(
    label="Left-truncatable prime",
    description="Remains prime when leading digits removed.",
    oeis="A024785",
    category=CATEGORY,
    limit=10**8-1
)
def is_left_truncatable_prime(n: int) -> tuple[bool, str]:
    """
    Check if n is a left-truncatable prime.
    A left-truncatable prime remains prime when digits are successively
    removed from the left. Example: 317 -> 17 -> 7; all are primes.
    """
    if n < 10:
        return False, None
    s = str(n)
    seq = []
    for i in range(len(s)):
        val = int(s[i:])
        if not isprime(val):
            return False, None
        seq.append(val)
    seq_str = " → ".join(str(x) for x in seq)
    return True, seq_str


@classifier(
    label="Motzkin prime",
    description="A Motzkin number that is also a prime. Motzkin numbers count certain lattice paths and combinatorial structures",
    oeis="A092832",
    category=CATEGORY
)
def is_motzkin_prime(n: int) -> tuple[bool, str]:
    """
    Check if n is a Motzkin prime.
    A Motzkin prime is a Motzkin number that is also a prime number.
    Only 4 Motzkin primes are known.
    """

    MOTZKIN_PRIMES = sorted([
        2, 127, 15511, 953467954114363
    ])

    if n in MOTZKIN_PRIMES:
        idx = MOTZKIN_PRIMES.index(n) + 1
        return True, f"{n} is a Motzkin number and also the {get_ordinal_suffix(idx)} known Motzkin prime."
    return False, None


def _is_2a_3b(x: int) -> tuple[bool, int, int]:
    """
    Return (True,a,b) if x = 2^a * 3^b for integers a,b >= 0; else (False,0,0).
    """
    if x <= 0:
        return (False, 0, 0)
    a = 0
    while (x & 1) == 0:  # divide out 2s
        x >>= 1
        a += 1
    b = 0
    while x % 3 == 0:    # divide out 3s
        x //= 3
        b += 1
    return (x == 1, a, b)


@classifier(
    label="Pierpont prime",
    description="Prime of form 2^a·3^b+1, also known as Class 1 primes.",
    oeis="A005109",
    category=CATEGORY,
    limit=10**8-1
)
def is_pierpont_prime(n: int, ctx: NumCtx | None = None):
    """
    Pierpont prime test (first or second kind):
      n is prime and either n-1 or n+1 equals 2^a * 3^b for some a,b >= 0.
    Returns (True, message) or (False, None).
    """
    if n < 2 or not ctx_isprime(n, ctx):
        return (False, None)

    ok1, a1, b1 = _is_2a_3b(n - 1)
    if ok1:
        return (True, f"{n} is a Pierpont prime: n - 1 = 2^{a1} · 3^{b1}")

    ok2, a2, b2 = _is_2a_3b(n + 1)
    if ok2:
        return (True, f"{n} is a Pierpont prime: n + 1 = 2^{a2} · 3^{b2}")

    return (False, None)


@classifier(
    label="Prime number",
    description="Is a prime.",
    oeis=None,
    category=CATEGORY
)
def is_prime_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Determines if n is prime. The is_prime_number function is only
    needed for intersection rules, it is filtered out as classifier.
    """
    if ctx_isprime(n, ctx):
        return True, None
    return False, None


@classifier(
    label="Primorial number",
    description="Product of all primes ≤ p for some prime p (p#).",
    oeis="A002110",
    category=CATEGORY,
)
def is_primorial_number(n: int) -> tuple[bool, str | None]:
    """
    Check if n is a primorial number p# = 2·3·5·...·p for some prime p.
    Does not use factorization; grows the primorial product until it
    either matches n or exceeds it.
    """

    n = int(n)
    if n < 2:
        return False, None

    # Safety cap for absurdly large inputs
    max_digits = CFG("CLASSIFIER.PRIMORIAL.MAX_DIGITS", 200)
    if isinstance(max_digits, int) and max_digits > 0 and dec_digits(n) > max_digits:
        raise TimeoutError("primorial check: |n| exceeds MAX_DIGITS")

    prod = 1
    p = 2
    primes: list[int] = []

    # Optional: guard against absurdly many primes (paranoia only)
    max_primes = CFG("CLASSIFIER.PRIMORIAL.MAX_PRIMES", 5000)

    while True:
        prod *= p
        primes.append(p)

        if prod == n:
            # p is the last prime we multiplied in
            # primes already holds all primes ≤ p in order
            factors_str = " × ".join(str(q) for q in primes)
            details = (
                f"{abbr_int_fast(n)} is a primorial number: "
                f"{p}# = {factors_str}."
            )
            return True, details

        if prod > n:
            # We have overtaken n; it cannot be a primorial
            return False, None

        if isinstance(max_primes, int) and max_primes > 0 and len(primes) >= max_primes:
            raise TimeoutError("primorial check: exceeded MAX_PRIMES")

        p = nextprime(p)


@classifier(
    label="Primorial prime",
    description="Prime of form p# ± 1, where p# is the product of the first n primes.",
    oeis="A018239",  # p#+1, A006794 (indices for p#-1)
    category=CATEGORY,
)
def is_primorial_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Check if n is a primorial prime, i.e. n = p# ± 1 for some prime p.
    Also indicate whether it's a minus-one or plus-one case and which p is used.
    """

    # Optional: avoid silly work on tiny or composite n
    if n < 2 or not ctx_isprime(n, ctx):
        return False, None

    def find_primorial_neighbor(m: int) -> int | None:
        """
        If m is exactly a primorial p#, return the last prime p.
        Otherwise return None.
        """
        if m < 1:
            return None

        prod = 1
        p = 2
        last_p = None

        # Iterate until prod >= m
        while prod < m:
            prod *= p
            last_p = p
            if prod == m:
                return last_p
            p = nextprime(p)
        return None

    # Try n = p# - 1  (so m = n + 1)
    p_minus = find_primorial_neighbor(n + 1)
    if p_minus is not None:
        # Special-case: 2 = 1# + 1 if you want to count it;
        # here it's covered as 2 = 1# + 1 via p = 2
        details = (
            f"{abbr_int_fast(n)} is a primorial prime of the form "
            f"{abbr_int_fast(p_minus)}# − 1."
        )
        return True, details

    # Try n = p# + 1  (so m = n - 1)
    p_plus = find_primorial_neighbor(n - 1)
    if p_plus is not None:
        details = (
            f"{abbr_int_fast(n)} is a primorial prime of the form "
            f"{abbr_int_fast(p_plus)}# + 1."
        )
        return True, details

    # Optional: treat n = 2 as the degenerate case 1# + 1
    if n == 2:
        return True, "2 is usually counted as a trivial primorial prime: 1# + 1."

    return False, None


@classifier(
    label="Proth prime",
    description="Prime of form k×2^m+1 with k odd and k<2^m.",
    oeis="A080076",
    category=CATEGORY,
    limit=10**8-1
)
def is_proth_prime(n: int):
    """
    Return (True, msg) if n is a Proth prime, else (False, None).

    A Proth number has the form n = k*2^m + 1 with k odd and 2^m > k.
    A Proth prime is a Proth number that is prime. By Proth's theorem,
    a Proth number n is prime iff there exists a such that a^((n-1)/2) ≡ -1 (mod n).
    """
    # Quick rejects
    if n < 3 or (n & 1) == 0:
        return False, None

    t = n - 1

    # Extract the largest power of two dividing (n-1): t = (2^m) * k, k odd
    pow2 = t & -t                 # value 2^m
    m = pow2.bit_length() - 1     # m = v2(t)
    k = t // pow2                 # odd part

    # Check Proth form condition: k odd (by construction) and 2^m > k
    if not (pow2 > k and (k & 1) == 1):
        return False, None

    # Proth's theorem: find a witness 'a' with a^((n-1)/2) ≡ -1 (mod n)
    e = t // 2
    for a in (3, 5, 7, 10, 11, 13, 17, 19, 23, 29):
        if a % n == 0:
            continue
        # If gcd(a, n) > 1 then n is composite; skip/early-return False.
        # (We can skip gcd to keep it cheap; pow will still work, but this is tidy.)
        # from math import gcd
        # if gcd(a, n) != 1: return False, None
        if pow(a, e, n) == n - 1:  # ≡ -1 mod n
            return True, f"{n} is a Proth prime: n = {k}×2^{m} + 1, witness a = {a}"

    # No witness found among small bases → n is composite (for Proth numbers)
    return False, None


@classifier(
    label="Ramanujan prime",
    description="Smallest R where π(x) − π(x/2) ≥ n for all x ≥ R.",
    oeis="A104272",
    category=CATEGORY,
    limit=242058
)
def is_ramanujan_prime(n: int) -> tuple[bool, str | None]:
    """
    Returns (True, details) if n is a Ramanujan prime (OEIS A104272).
    Uses the OEIS b-file b104272.txt.
    """
    if n < 2:
        return False, None

    found, idx_file, series, idx_set = check_oeis_bfile("b104272.txt", n)
    if not series:
        # Graceful fallback if file missing or unreadable
        return False, None

    if found:
        # Prefer OEIS index (first column), else use 0-based position + 1
        idx = idx_file if idx_file is not None else (idx_set + 1 if idx_set is not None else None)
        if idx is not None:
            return True, f"{n} is the {get_ordinal_suffix(idx)} Ramanujan prime."
        else:
            return True, "Listed in OEIS A104272."
    return False, None


@classifier(
    label="Right-truncatable prime",
    description="Remains prime when trailing digits removed.",
    oeis="A024770",
    category=CATEGORY,
    limit=10**8-1
)
def is_right_truncatable_prime(n: int) -> tuple[bool, str]:
    """
    Check if n is a right-truncatable prime.
    A right-truncatable prime remains prime when digits are successively
    removed from the right.
    Example: 739397 -> 73939 -> 7393 -> 739 -> 73 -> 7; all are primes.
    """
    if n < 10:
        return False, None
    s = str(n)
    seq = []
    for i in range(len(s), 0, -1):
        val = int(s[:i])
        if not isprime(val):
            return False, None
        seq.append(val)
    seq_str = " → ".join(str(x) for x in seq)
    return True, seq_str


@classifier(
    label="Safe prime",
    description="Prime where (n−1)/2 is also prime.",
    oeis="A005385",
    category=CATEGORY
)
def is_safe_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check if n is a safe prime.
    A safe prime is a prime number of the form 2k + 1, where k is also prime.
    """
    if n < 5:
        return False, None
    if not ctx_isprime(n, ctx):
        return False, None
    k = (n - 1) // 2
    if 2 * k + 1 != n:
        return False, None  # n must be odd
    if isprime(k):
        return True, f"{n} = 2×{k} + 1 with {k} also prime (safe prime condition satisfied)"
    return False, None


@classifier(
    label="Sexy prime",
    description="Prime with another prime 6 away.",
    oeis="A023201",
    category=CATEGORY
)
def is_sexy_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check if n is a sexy prime.
    Details show the sexy prime pair.
    """
    if n < 5:
        return False, None
    if not ctx_isprime(n, ctx):
        return False, None
    if isprime(n - 6):
        details = f"{n} and {n-6} are sexy primes (differ by 6)."
        return True, details
    if isprime(n + 6):
        details = f"{n} and {n+6} are sexy primes (differ by 6)."
        return True, details
    return False, None


@classifier(
    label="Semiprime",
    description="Product of exactly two (not necessarily distinct) primes.",
    oeis="A001358",
    category=CATEGORY
)
def is_semiprime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Semiprime ⇔ sum of prime-factor exponents = 2
      - covers p*q and p^2
      - excludes primes and smaller numbers
    """
    m = abs(n)
    if m < 4:
        return False, None  # smallest semiprime is 4 = 2×2

    ctx = ctx or build_ctx(m)
    fac = ctx.fac  # {prime: exponent}

    # Exclude primes quickly: one prime with exponent 1
    if len(fac) == 1 and next(iter(fac.values())) == 1:
        return False, None

    total_exponents = sum(fac.values())
    if total_exponents == 2:
        # Reconstruct the two primes with multiplicity for a clean detail
        primes = []
        for p, e in fac.items():
            primes.extend([p] * e)
        primes.sort()
        detail = f"{m} = {' × '.join(map(str, primes))}"
        return True, detail

    return False, None


@classifier(
    label="Sophie Germain prime",
    description="p where 2p+1 is also prime.",
    oeis="A005384",
    category=CATEGORY
)
def is_sophie_germain_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check if n is a Sophie Germain prime.
    Details show that both n and 2n+1 are prime.
    """
    if n < 2:
        return False, None
    if ctx_isprime(n, ctx) and isprime(2 * n + 1):
        details = f"{n} is prime and 2×{n} + 1 = {2 * n + 1} is also prime."
        return True, details
    return False, None


@classifier(
    label="Strong prime",
    description=(
        "Prime larger than the average of the previous and next primes: "
        "pₙ > (pₙ₋₁ + pₙ₊₁) / 2."
    ),
    oeis="A051634",
    category=CATEGORY,
    limit=10**20 - 1,
)
def is_strong_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    A strong prime (in number theory) satisfies
        p_n > (p_{n-1} + p_{n+1}) / 2
    where p_n is the n-th prime.
    """
    if n < 3 or not ctx_isprime(n, ctx):
        return False, None

    p_prev = prevprime(n)
    p_next = nextprime(n)
    lhs = p_prev + p_next
    rhs = 2 * n

    if lhs < rhs:
        # Equivalent to n > (p_prev + p_next)/2
        details = (
            f"{n} is strong: {p_prev} + {p_next} = {lhs} < 2×{n} = {rhs} "
            f"⇒ {n} > ({p_prev}+{p_next})/2."
        )
        return True, details

    return False, None


@classifier(
    label="Super prime",
    description="Prime whose index in primes is prime.",
    oeis="A006450",
    category=CATEGORY,
    limit=10**8-1
)
def is_super_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Check if n is a super-prime.
    Details show the prime index of n.
    """
    if n < 1:
        return False, None
    if not ctx_isprime(n, ctx):
        return False, None
    pos = primepi(n)  # Number of primes ≤ n
    if isprime(pos):
        details = f"{n} is the {pos}th prime, and {pos} is also prime."
        return True, details
    return False, None


@classifier(
    label="Prime triplet member",
    description="n ∈ {3,5,7}, the only prime triplet (p, p+2, p+4).",
    oeis=None,
    category=CATEGORY,
)
def is_prime_triplet_member(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    # Build ctx if you want consistent invocation; not used here
    ctx = ctx or build_ctx(abs(n))

    if n in (3, 5, 7):
        pos = {3: "first", 5: "second", 7: "third"}[n]
        return True, f"{n} is the {pos} element of the only prime triplet (3, 5, 7)"
    return False, None


@classifier(
    label="Thin prime",
    description=(
        "Odd prime p such that p+1 is either a power of 2 or a prime times a power of 2: "
        "p+1 = 2^k or p+1 = q×2^k with q prime."
    ),
    oeis="A192869",
    category=CATEGORY,
    limit=10**20 - 1,
)
def is_thin_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Thin primes (OEIS A192869, Broughan–Zhou):
    odd primes p for which p+1 = 2^k or p+1 = q×2^k with q prime.
    """
    # Must be an odd prime ≥ 3
    if n < 3 or n % 2 == 0 or not ctx_isprime(n, ctx):
        return False, None

    m = n + 1
    k = 0

    # Factor out powers of 2 from p+1
    while m % 2 == 0:
        m //= 2
        k += 1

    # Now n+1 = m * 2^k with m odd and k >= 1
    if m == 1:
        # pure power of 2
        details = (
            f"{n}+1 = 2^{k} "
            f"(thin prime: p+1 is a pure power of 2)."
        )
        return True, details

    if isprime(m):
        # prime times a power of 2
        details = (
            f"{n}+1 = {m}×2^{k} with {m} prime "
            f"(thin prime: p+1 is a prime times a power of 2)."
        )
        return True, details

    # Otherwise not thin
    return False, None


@classifier(
    label="Twin prime",
    description="Prime with another prime 2 away.",
    oeis="A001359",  # lessor of twin primes, A006512 greater of twin primes.
    category=CATEGORY,
    limit=10**20-1
)
def is_twin_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Returns (True, details) if n is a twin prime, else False.
    Details show the twin prime pair.
    """
    if n < 3:
        return False, None
    if not ctx_isprime(n, ctx):
        return False, None
    if isprime(n - 2):
        details = f"{n} and {n-2} are twin primes (differ by 2)."
        return True, details
    if isprime(n + 2):
        details = f"{n} and {n+2} are twin primes (differ by 2)."
        return True, details
    return False, None


@classifier(
    label="Weak prime",
    description=(
        "Prime smaller than the average of the previous and next primes: "
        "pₙ < (pₙ₋₁ + pₙ₊₁) / 2."
    ),
    oeis="A051635",
    category=CATEGORY,  # same primes category
    limit=10**20 - 1,
)
def is_weak_prime(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    A weak prime (early prime) satisfies
        p_n < (p_{n-1} + p_{n+1}) / 2.
    """
    if n < 3 or not ctx_isprime(n, ctx):
        return False, None

    p_prev = prevprime(n)
    p_next = nextprime(n)
    lhs = p_prev + p_next
    rhs = 2 * n

    if lhs > rhs:
        # Equivalent to n < (p_prev + p_next)/2
        details = (
            f"{n} is weak: {p_prev} + {p_next} = {lhs} > 2×{n} = {rhs} "
            f"⇒ {n} < ({p_prev}+{p_next})/2."
        )
        return True, details

    return False, None


@classifier(
    label="Wieferich prime",
    description="2^(p−1) ≡ 1 mod p^2",
    oeis="A001220",
    category=CATEGORY
)
def is_wieferich_prime(n: int) -> tuple[bool, str]:
    """
    Returns (True, details) if n is a Wieferich prime, else False.
    Only two are known as of 2025.
    Details explain the Wieferich congruence.
    """
    KNOWN = {1093, 3511}
    if n in KNOWN:
        details = f"{n} is a Wieferich prime: 2^({n}-1) ≡ 1 mod {n}², only 2 such primes are known."
        return True, details
    return False, None


@classifier(
    label="Wilson prime",
    description="(p−1)! ≡ −1 mod p^2",
    oeis="A007540",
    category=CATEGORY
)
def is_wilson_prime(n: int) -> tuple[bool, str]:
    """
    Check if n is a Wilson prime.
    A Wilson prime is a prime p such that (p−1)! ≡ −1 mod p².
    Only three are known: 5, 13, and 563.
    """

    WILSON_PRIMES = [5, 13, 563]

    if n in WILSON_PRIMES:
        idx = WILSON_PRIMES.index(n) + 1
        return True, f"{n} is the {get_ordinal_suffix(idx)} known Wilson prime. Only 3 such primes are known."
    return False, None
