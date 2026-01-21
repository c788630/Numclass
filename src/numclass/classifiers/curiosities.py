# -----------------------------------------------------------------------------
#  curiosities.py
#  Mathematical curiosities test functions
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sympy import prime, primerange

if TYPE_CHECKING:
    from numclass.context import NumCtx

from numclass.fmt import abbr_int_fast, format_factorization
from numclass.registry import classifier
from numclass.runtime import CFG
from numclass.utility import build_ctx, check_oeis_bfile, dec_digits, get_ordinal_suffix, int_to_words, zeckendorf_decomposition

CATEGORY = "Mathematical Curiosities"


@classifier(
    label="Additive sequence",
    description="Digits can be split into a1, a2, … with a_{k+1} = a_k + a_{k-1}.",
    oeis=None,
    category=CATEGORY,
    limit=10**20-1
)
def is_additive_sequence(n: int):
    """
    Returns (True, "n = a1, a2, ...") if the base-10 digits of |n|
    can be split into an additive sequence:
        each term after the first two equals the sum of the previous two.
    Rules:
      - At least three terms overall.
      - No leading zeros in any multi-digit term (but the digit '0' itself is allowed).
    """
    s = str(abs(n))
    L = len(s)
    if L < 3:
        return False, None

    # If the number starts with '0', the first term can only be '0'
    # (any longer first term would have a leading zero and is invalid).
    first_len_range = range(1, L - 1) if s[0] != "0" else range(1, 2)

    for i in first_len_range:  # length of first term
        a_str = s[:i]
        # For the second term, avoid leading zeros in multi-digit pieces
        for j in range(1, L - i):  # length of second term
            b_str = s[i:i + j]
            if (a_str[0] == "0" and len(a_str) > 1) or (b_str[0] == "0" and len(b_str) > 1):
                continue

            # Build the sequence greedily
            seq = [int(a_str), int(b_str)]
            k = i + j  # index into s where the next term should start

            while k < L:
                c = seq[-1] + seq[-2]
                c_str = str(c)
                if not s.startswith(c_str, k):
                    break
                seq.append(c)
                k += len(c_str)

            # Success if we consumed all digits and have >= 3 terms
            if k == L and len(seq) >= 3:
                # Build "a+b=c" steps from term3 onward
                steps = [f"{seq[t-2]}+{seq[t-1]}={seq[t]}" for t in range(2, len(seq))]
                # If you want to cap very long outputs, set a MAX_STEPS and slice here
                details = f"{int(s)}: " + ", ".join(steps)
                return True, details

    return False, None


@classifier(
    label="Binary-interpretable number",
    description="Number can be interpreted as a binary numeral.",
    category=CATEGORY
)
def is_binary_interpretable(n: int) -> tuple[bool, str]:
    """
    Returns (True, details) if the decimal representation of n consists only of 0s and 1s,
    and is at least 2 digits (to avoid matching trivial '0' and '1').
    """
    s = str(n)
    if set(s) <= {"0", "1"} and len(s) > 1:
        dec_value = int(s, 2)
        details = (
            f"As binary: {s}₂ = {dec_value}₁₀ = {format(dec_value, 'o')}₈ = {format(dec_value, 'X')}₁₆"
        )
        return True, details
    return False, None


@classifier(
    label="Cyclic permutation number",
    description="Multiples are cyclic rotations of its digits.",
    oeis="A180340",
    category=CATEGORY,
)
def is_cyclic_permutation_number(n: int) -> tuple[bool, str | None]:
    """
    Check if n is a cyclic permutation number (OEIS A180340).
    Uses OEIS b-file b180340.txt for fast membership.
    These numbers are the numerators of cyclic decimals: the first k multiples
    of n (where k = digit count) are cyclic rotations of its digits.
    """

    # Fast membership using OEIS b-file.
    found, idx_file, series, idx_set = check_oeis_bfile("b180340.txt", n)

    # Determine precomputed range:
    # series is the ordered list from the b-file.
    precomp_limit = (max(series) + 1) if series else 0

    # Inside OEIS precomputed range?
    if n < precomp_limit:
        if found:
            # Resolve ordinal index:
            ordinal = (
                idx_file if idx_file is not None
                else (idx_set + 1 if idx_set is not None
                      else series.index(n) + 1)
            )

            details = (
                f"{n} is the {get_ordinal_suffix(ordinal)} known cyclic "
                "permutation number. Its multiples are cyclic rotations "
                "of its digits."
            )
            return True, details

        # Not found in OEIS list
        return False, None

    # Outside the precomputed range — we do not compute them.
    return False, None


@classifier(
    label="Cyclops number",
    description="Odd number of digits and a single zero in the center digit.",
    oeis="A134808",
    category=CATEGORY,
)
def is_cyclops_number(n: int) -> tuple[bool, str]:
    """
    Returns (True, details) if n is a cyclops number: odd digits, one zero in center, rest nonzero.
    """
    if abs(n) < 10:
        return False, None

    s = str(abs(n))
    length = len(s)
    if length % 2 == 1:
        mid = length // 2
        # The center digit must be 0, all others nonzero
        if s[mid] == '0' and s.count('0') == 1 and all(ch != '0' for i, ch in enumerate(s) if i != mid):
            details = f"{n} has {length} digits with a single 0 in the center: {s}"
            return True, details
    return False, None


@classifier(
    label="Eban number",
    description="No letter 'e' in the English spelling.",
    oeis="A006933",
    category=CATEGORY,
    limit=66_000_000_000_000_000_000_000_000_000_000_066_066_000_000_066_066_066_066_066_066_066
)
def is_eban_number(n: int) -> tuple[bool, str | None]:

    words = int_to_words(n)

    if "e" in words.replace("-", "").replace(" ", ""):
        return False, words
    return True, f"{n}: {words} is an eban number (no 'e' in its English spelling)"


@classifier(
    label="Fibonacci-base palindrome",
    description="Zeckendorf Fibonacci-base bitstring (bits(Fk..F2)) is palindromic.",
    category=CATEGORY,
)
def is_fibonacci_base_palindrome(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n < 0:
        return False, None

    terms, idxs, bits = zeckendorf_decomposition(n)

    if bits == "0":
        return True, "bits=0 (palindrome)"

    if bits != bits[::-1]:
        return False, None

    k = max(idxs) if idxs else 0
    weight = len(terms)
    return True, f"bits(F{k}..F2)={bits} (palindrome), weight={weight}"


@dataclass(frozen=True)
class LcmPrefixCandidate:
    k: int
    lower: int
    upper: int


def _candidate_k_for_lcm_prefix_from_factorization(fac: dict[int, int]) -> LcmPrefixCandidate | None:
    """
    Given fac = {p: e} for n = ∏ p^e, compute the admissible interval for k such that
      floor(log_p(k)) == e   for every prime p dividing n.

    That condition is equivalent to:
      p^e <= k < p^(e+1)   for each p in fac

    Intersect these intervals across all primes to get [lower, upper].
    If non-empty, choosing k = lower is the easiest existence witness.
    """
    if not fac:
        return LcmPrefixCandidate(k=1, lower=1, upper=1)

    lower = 1
    upper = 10**30  # big sentinel; will shrink quickly

    for p, e in fac.items():
        if p < 2 or e < 1:
            return None
        lo = p ** e
        hi = (p ** (e + 1)) - 1
        lower = max(lower, lo)
        upper = min(upper, hi)
        if lower > upper:
            return None

    return LcmPrefixCandidate(k=lower, lower=lower, upper=upper)


@classifier(
    label="LCM-prefix number",
    description="Equal to lcm(1,2,…,k) for some k (the smallest number divisible by every integer 1..k).",
    oeis="A003418",
    category=CATEGORY,
)
def is_lcm_prefix_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    True iff n = lcm(1..k) for some integer k >= 1.

    Key facts:
      - If n = ∏ p^e then for a valid k we must have p^e <= k < p^(e+1) for each p|n.
        (Because lcm(1..k) includes p^floor(log_p k).)
      - Additionally, every prime q <= k must divide n (otherwise q would appear in lcm(1..k) but not in n).
    """
    if n <= 0:
        return False, None

    # Optional safety caps (LCM-prefix values grow *fast*, but factoring still may be expensive)
    MAX_DIGITS = int(CFG("CLASSIFIER.LCM_PREFIX.MAX_DIGITS", 2000))
    if dec_digits(n) > MAX_DIGITS:
        return False, f"n has more than {MAX_DIGITS} digits (classifier cap)"

    ctx = ctx or build_ctx(n)
    if ctx.incomplete:
        raise TimeoutError("incomplete factorization")

    if n == 1:
        return True, "k=1: lcm(1)=1"

    fac = dict(ctx.fac or {})
    if not fac:
        return False, None  # should not happen for n>1 if factorization is complete

    cand = _candidate_k_for_lcm_prefix_from_factorization(fac)
    if cand is None:
        return False, None

    # k-range might contain multiple k (e.g. 2520 is both lcm(1..9) and lcm(1..10)).
    # For existence, the easiest witness is k = lower.
    k = cand.k

    # Optional cap to avoid scanning primes up to a huge k (rare, but configurable).
    MAX_K = int(CFG("CLASSIFIER.LCM_PREFIX.MAX_K", 500_000))
    if k > MAX_K:
        return False, f"candidate k={k} exceeds MAX_K={MAX_K}"

    # Condition: every prime q <= k must divide n.
    # (Because q appears in lcm(1..k) with exponent >=1.)
    for q in primerange(2, k + 1):
        if q not in fac:
            return False, f"fails at k={k}: missing prime {q} (but {q} ≤ k, so it must divide lcm(1..k))"

    # If we get here, exponents are automatically consistent by construction of [lower, upper],
    # and we have all primes up to k present. Therefore n == lcm(1..k).
    # Provide a nice detail string and mention if there are multiple k values.
    lcm_fac: dict[int, int] = {}
    for p in primerange(2, k + 1):
        e = 1
        pe = p
        while pe * p <= k:
            pe *= p
            e += 1
        lcm_fac[p] = e
    powers = format_factorization(lcm_fac)

    if cand.upper != cand.lower:
        # Mention that n stays constant across a k-interval
        return True, (
            f"k={k}. Also valid for any k in [{cand.lower}..{cand.upper}] "
            f"(lcm(1..k) stays the same there). "
            f"Prime powers: {powers}"
        )

    return True, f"k={k}. Prime powers: {powers}"


@classifier(
    label="Lucky number of Euler",
    description="One of the nine Heegner numbers n for which e^{π√n} is remarkably close to an integer.",
    oeis="A003173",
    category=CATEGORY,
)
def is_lucky_number_of_euler(n) -> tuple[bool, str]:
    """
    Checks if n is a 'lucky number of Euler' (Heegner number).
    These are the unique positive integers n for which Q(√−n)
    has class number 1, giving rise to the near-integer phenomenon
    e^{π√n} ≈ integer.
    """

    EULER_LUCKY_NUMBERS = {
        1:      "e^{π√1} ≈ 23.140692632779..., rounded: 23",
        2:      "e^{π√2} ≈ 85.019695223650..., rounded: 85",
        3:      "e^{π√3} ≈ 230.835277861444..., rounded: 231",
        7:      "e^{π√7} ≈ 3355.912509175..., rounded: 3356",
        11:     "e^{π√11} ≈ 640320.000001..., rounded: 640320",
        19:     "e^{π√19} ≈ 885479.777680..., rounded: 885480",
        43:     "e^{π√43} ≈ 884736743.999777..., rounded: 884736744",
        67:     "e^{π√67} ≈ 147197952743.999998..., rounded: 147197952744",
        163:    "e^{π√163} ≈ 262537412640768743.99999999999925..., rounded: 262537412640768744",
    }

    value = EULER_LUCKY_NUMBERS.get(n)

    if value:
        details = (
            f"{n} is a 'lucky number of Euler' (Heegner number), {value}: a celebrated mathematical phenomenon!"
        )
        return True, details
    return False, None


@classifier(
    label="Munchausen number",
    description="Equals the sum of its digits each raised to the power of itself.",
    oeis="A046253",
    category=CATEGORY
)
def is_munchausen_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a Munchausen number (A069768).
    A Munchausen number is a number equal to the sum of its digits raised to the power of themselves.
    For these numbers, 0^0 is defined as 0 (see OEIS A069768).
    Could also be a list with just the only numbers 0, 1, 3435, but the algorithm us used
    for educational purposes.
    """
    if n < 1:
        return False, None
    digits = [int(d) for d in str(n)]
    total = sum((d ** d if d != 0 else 0) for d in digits)
    if total == n:
        detail = " + ".join([f"{d}^{d}" if d != 0 else "0^0" for d in digits])
        if n == 0:
            # Add note for n=0 only
            return True, (
                f"{n} = {detail} by the convention used for Munchausen numbers."
            )
        return True, f"{n} = {detail}"
    return False, None


@classifier(
    label="Octal-interpretable number",
    description="Number can be interpreted as an octal numeral.",
    category=CATEGORY
)
def is_octal_interpretable(n: int) -> tuple[bool, str]:
    """
    Returns (True, details) if the decimal representation of n consists only of digits 0–7,
    and is at least 2 digits (to avoid trivial single-digit octals).
    """
    s = str(n)
    if set(s) <= set("01234567") and len(s) > 1:
        dec_value = int(s, 8)
        details = (
            f"As octal: {s}₈ = {dec_value}₁₀ = {format(dec_value, 'b')}₂ = {format(dec_value, 'X')}₁₆"
        )
        return True, details
    return False, None


@classifier(
    label="Pandigital number (0-9)",
    description="Contains each digit 0–9 exactly once.",
    oeis="A050278",
    category=CATEGORY
)
def is_pandigital_number_0_9(n: int) -> tuple[bool, str]:
    """
    Check if n is a pandigital number (0-9) version 2.
    A 0–9 pandigital number contains each digit from 0 to 9 exactly once.
    """
    if n < 0:
        return False, None
    s = str(n)
    if len(s) == 10 and set(s) == set('0123456789'):
        details = f"{n} contains each digit 0–9 exactly once."
        return True, details
    return False, None


@classifier(
    label="Pandigital number (1-9)",
    description="Contains each digit 1–9 exactly once.",
    oeis="A050289",
    category=CATEGORY
)
def is_pandigital_number_1_9(n: int) -> tuple[bool, str]:
    """
    Check if n is a pandigital number (1-9).
    A 1–9 pandigital number contains each digit from 1 to 9 exactly once, and no others.
    """
    if n < 0:
        return False, None
    s = str(n)
    if len(s) == 9 and set(s) == set('123456789'):
        details = f"{n} contains each digit 1–9 exactly once."
        return True, details
    return False, None


def _prefixes_base10(n: int) -> list[int]:
    s = str(abs(n))
    # no leading zeros for normal integers; if you want to allow them, that's a different concept
    out: list[int] = []
    cur = 0
    for ch in s:
        cur = cur * 10 + (ord(ch) - 48)
        out.append(cur)
    return out


@classifier(
    label="Polydivisible number",
    description="In base 10: for every k = 1..d, the first k digits form a number divisible by k.",
    oeis=None,  # optional: add OEIS if you like
    category=CATEGORY,
)
def is_polydivisible_number(n: int, ctx=None) -> tuple[bool, str | None]:
    """
    Polydivisible (prefix-divisible) in base 10.
    Example: 3816547290 is polydivisible because its first k digits are divisible by k
    for all k = 1..10.
    """
    if n == 0:
        # 0 has prefixes 0,00,... if you allow leading zeros; NumClass inputs don't.
        return False, None

    s = str(abs(n))
    if not s:
        return False, None

    # Reject if the decimal representation has leading zeros (shouldn't happen for int input)
    if len(s) > 1 and s[0] == "0":
        return False, None

    prefixes = _prefixes_base10(n)

    bad_k: int | None = None
    bad_prefix: int | None = None
    for k, pk in enumerate(prefixes, start=1):
        if pk % k != 0:
            bad_k = k
            bad_prefix = pk
            break

    if bad_k is not None:
        # Optional small counterexample details for small-ish numbers
        if len(s) <= 18:
            return False, f"fails at k={bad_k}: prefix {bad_prefix} is not divisible by {bad_k}"
        return False, None

    # Success: build a compact, friendly Details line
    d = len(prefixes)

    # Include a few illustrative checks (first, a middle, and last), without spamming
    samples: list[str] = []
    if d >= 1:
        samples.append(f"P₁={prefixes[0]} divisible by 1")
    if d >= 4:
        samples.append(f"P₄={prefixes[3]} divisible by 4")
    if d >= 8:
        samples.append(f"P₈={prefixes[7]} divisible by 8")
    samples.append(f"P_{d}={abs(n)} divisible by {d}")

    detail = (
        f"Let Pₖ be the first k digits. For k=1..{d}, Pₖ ≡ 0 (mod k). "
        + " ; ".join(samples)
        + "."
    )

    # Special wink for the famous 10-digit example
    if abs(n) == 3816547290:
        detail += " This is the unique 10-digit polydivisible number in base 10."

    return True, detail


@classifier(
    label="Self-power number",
    description="Number of the form k^k (self power) for some integer k ≥ 1.",
    oeis="A000312",
    category=CATEGORY,
)
def is_self_power_number(n: int) -> tuple[bool, str | None]:
    """
    Return True iff n = k^k for some integer k ≥ 1.

    Method:
      - For n ≥ 2, solve k log k ≈ log n using a float-based approximation
        (with log n estimated from bit length to avoid converting n to float).
      - Then search a small integer window around the approximate k.
    """
    if n <= 0:
        return False, None
    if n == 1:
        # 1 = 1^1 is the natural convention
        return True, "1 = 1^1"

    # Rough size of n in bits
    bitlen = n.bit_length()

    # Optional safety guard for absurdly large inputs; adjust or remove as you like:
    # For bitlen > ~20000 (~6000 decimal digits) you may decide to skip.
    # if bitlen > 20000:
    #     return False, None

    # Estimate ln(n) from bit length: n < 2^bitlen, so ln(n) ≈ bitlen * ln 2
    ln_n = bitlen * math.log(2.0)
    if ln_n <= 0:
        return False, None

    # Initial guess for k from k log k ≈ ln n  ⇒  k ≈ ln n / ln ln n
    ln_ln_n = math.log(ln_n)
    k_est = ln_n / max(ln_ln_n, 1.0)
    k_est = max(k_est, 1.0)

    # A few Newton iterations on f(k) = k log k - ln n
    for _ in range(3):
        k = max(k_est, 1.0)
        f = k * math.log(k) - ln_n
        df = math.log(k) + 1.0
        k_est -= f / df

    k0 = max(1, round(k_est))

    # Check a small window around the estimated k
    # The log approximation is quite good; ±3 is plenty.
    start = max(1, k0 - 3)
    stop = k0 + 4

    for k in range(start, stop):
        val = pow(k, k)
        if val == n:
            return True, f"{n} = {k}^{k}"
        # If the values are growing past n and k is already above k0, we can stop early
        if val > n and k >= k0:
            break

    return False, None


@classifier(
    label="Smarandache Wellin number",
    description="Concatenation of the first k primes (e.g., 2, 23, 235, 2357, …).",
    oeis="A019518",
    category=CATEGORY,
)
def is_smarandache_wellin_number(n: int) -> tuple[bool, str | None]:
    """
    Returns (True, details) if n is a Smarandache–Wellin number, else False.
    Details: Shows which primes were concatenated.
    """
    if n < 2:
        return False, None

    n_str = str(n)

    # Quick necessary condition: all Smarandache–Wellin numbers start with 2
    if not n_str.startswith("2"):
        return False, None

    concat = ""
    primes_used: list[int] = []
    target_len = len(n_str)
    i = 1

    # Build concatenation of first k primes until length >= len(n)
    while len(concat) < target_len:
        p = prime(i)
        primes_used.append(p)
        concat += str(p)

        if concat == n_str:
            k = len(primes_used)
            details = (
                f"{abbr_int_fast(n)} is formed by concatenating the first {k} primes: "
                + ", ".join(str(p) for p in primes_used)
                + "."
            )
            return True, details

        i += 1

    return False, None


@classifier(
    label="Strobogrammatic number",
    description="Looks the same upside down.",
    oeis="A000787",
    category=CATEGORY
)
def is_strobogrammatic_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a strobogrammatic number.
    A strobogrammatic number remains the same when rotated 180 degrees.
    Only uses digits: 0, 1, 6, 8, 9.
    Examples: 69, 88, 818, 609, 101, 0, 1, 8.
    """
    if n < 0:
        return False, None
    strobo_map = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
    s = str(n)
    rotated = ''.join(strobo_map.get(ch, '?') for ch in reversed(s))
    if '?' in rotated:
        return False, None
    if s == rotated:
        # Build details showing rotation
        details = f"{n} reads the same when rotated 180°: {rotated}"
        return True, details
    return False, None


@classifier(
    label="Vampire number",
    description="Composite number factorable into two equal-length factors using original digits exactly once.",
    oeis="A014575",
    category=CATEGORY,
    limit=10**14-1
)
def is_vampire_number(n: int) -> tuple[bool, str | None]:
    m = abs(int(n))  # ignore sign for classification
    s = str(m)
    L = len(s)
    if L < 4 or (L % 2):
        return False, None

    half = L // 2
    # Optional quick reject (requires sympy):
    # if isprime(m):
    #     return False, None

    target = Counter(s)
    lo = 10**(half - 1)
    hi = 10**half - 1

    # iterate factor pairs with correct digit lengths
    for x in range(lo, hi + 1):
        if m % x != 0:
            continue
        y = m // x
        if y < lo or y > hi:
            continue
        # both fangs cannot end with zero
        if (x % 10 == 0) and (y % 10 == 0):
            continue
        # exact digit multiset match
        if Counter(f"{x}{y}") == target:
            sign = "-" if n < 0 else ""
            return True, f"{sign}{m} = {x} × {y} (vampire number)"
    return False, None
