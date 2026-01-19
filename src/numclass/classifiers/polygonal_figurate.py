# -----------------------------------------------------------------------------
#  polygonal_figurate.py
#  Polygonal and Figurate Numbers test functions
# -----------------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache
from math import isqrt, log2

from sympy import integer_nthroot

from numclass.context import NumCtx
from numclass.registry import classifier
from numclass.utility import base_name, get_ordinal_suffix

CATEGORY = "Polygonal and Figurate Numbers"


def _centered_k(n: int, s: int) -> int | None:
    """
    Return k>=1 such that n = 1 + (s/2)×k(k−1), else None.
    Works with big integers (uses exact sqrt via isqrt).
    """
    if n < 1 or s <= 0:
        return None
    # t = 2(n-1)/s must be integer
    num = 2 * (n - 1)
    if num % s != 0:
        return None
    t = num // s
    D = 1 + 4 * t
    r = isqrt(D)
    if r * r != D:
        return None
    # k = (1 + r) / 2 must be integer, k>=1
    if (1 + r) % 2 != 0:
        return None
    k = (1 + r) // 2
    return k if k >= 1 else None


def _is_polygonal(n: int, s: int) -> tuple[bool, int | None]:
    """
    Test if n is an s-gonal (polygonal) number.

    Uses the general formula:
        P_s(k) = ((s - 2)k² - (s - 4)k) / 2,  k ≥ 1.

    Returns:
        (True, k)  if such an integer k ≥ 1 exists with P_s(k) = n
        (False, None) otherwise.
    """
    if n < 1:
        return False, None

    # Only support the s-gonal types you actually use
    # (triangular=3, pentagonal=5, hexagonal=6, etc).
    # You can drop this check if you want it fully generic.
    if s < 3:
        return False, None

    # From n = ((s - 2)k² - (s - 4)k)/2
    # we get the quadratic:
    #   (s - 2)k² - (s - 4)k - 2n = 0
    a = s - 2
    b = 4 - s          # same as -(s - 4)
    c = -2 * n

    disc = b * b - 4 * a * c
    if disc < 0:
        return False, None

    root = isqrt(disc)
    if root * root != disc:
        return False, None

    # k = (-b + root) / (2a)  (we want the positive root)
    num = -b + root
    den = 2 * a

    if num <= 0 or num % den != 0:
        return False, None

    k = num // den
    if k <= 0:
        return False, None

    return True, k


@classifier(
    label="Centered hexagonal number",
    description="n = 1 + 3×k(k−1) for some k≥1.",
    oeis="A003215",
    category=CATEGORY,
)
def is_centered_hexagonal(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    k = _centered_k(n, 6)
    return (True, _detail(n, 6, k)) if k else (False, None)


@classifier(
    label="Centered square number",
    description="n = 1 + 2×k(k−1) for some k≥1 (equivalently, an odd square).",
    oeis="A001844",
    category=CATEGORY,
)
def is_centered_square(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    k = _centered_k(n, 4)
    return (True, _detail(n, 4, k)) if k else (False, None)


def _detail(n: int, s: int, k: int) -> str:
    return f"n = 1 + ({s}/2)×k(k−1) with k={k} ⇒ 2(n−1)/{s} = {2*(n-1)//s} = k(k−1)"


@classifier(
    label="Centered triangular number",
    description="n = 1 + (3/2)× k(k−1) for some k≥1.",
    oeis="A005448",
    category=CATEGORY,
)
def is_centered_triangular(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    k = _centered_k(n, 3)
    return (True, _detail(n, 3, k)) if k else (False, None)


@classifier(
    label="Hexagonal number",
    description="n = k(2k−1) for some k.",
    oeis="A000384",
    category=CATEGORY,
)
def is_hexagonal_number(n: int) -> tuple[bool, str | None]:
    """
    Check if n is a hexagonal number.
    Hexagonal numbers satisfy n = k(2k−1) for some integer k ≥ 1.
    """
    if n < 1:
        return False, None

    ok, k = _is_polygonal(n, 6)
    if not ok or k is None:
        return False, None

    details = (
        f"{n} is the {get_ordinal_suffix(k)} hexagonal number "
        f"(since {n} = {k}×(2×{k}−1))."
    )
    return True, details


def _lehmer_factors(n: int) -> list[tuple[int, int]]:
    """
    Find (a, k) pairs for which n = (a^k − 1)//(a − 1),
    for bases 2..10 and k>1.
    """
    if n < 3:
        return []

    results: list[tuple[int, int]] = []
    for a in range(2, 11):
        # Rearranged: a^k = n*(a-1) + 1
        target = n * (a - 1) + 1
        # k max from a^k <= target  ->  k <= log_a(target)
        # using log2: log_a(x) = log2(x)/log2(a)
        max_k = int(log2(target) / log2(a)) + 1

        p = a * a  # a^2
        for k in range(2, max_k + 1):
            if p == target:
                results.append((a, k))
            if p > target:
                break
            p *= a

    return results


@classifier(
    label="Lehmer number",
    description="Equals (a^k−1)/(a−1) for some integers 2≤a≤10, k>1; "
                "in base a, written as k consecutive 1's.",
    oeis=None,
    category=CATEGORY,
    limit=10**20-1
)
def is_lehmer_number(n: int) -> tuple[bool, str]:
    """
    Lehmer number generalizes repunits and Mersenne numbers.
    """
    if n < 1:
        return False, None
    facts = _lehmer_factors(n)
    if facts:
        parts = []
        for a, k in facts:
            base_str = base_name(a)
            parts.append(f"{base_str}, k={k}")
        detail_str = "; ".join(parts)
        return True, detail_str
    return False, None


@classifier(
    label="Pentagonal number",
    description="n = k(3k−1)/2 for some k.",
    oeis="A000326",
    category=CATEGORY
)
def is_pentagonal_number(n: int) -> tuple[bool, str | None]:
    """
    Check if n is a pentagonal number.
    Pentagonal numbers satisfy n = k(3k−1)/2 for some integer k ≥ 1.
    """
    if n < 1:
        return False, None

    ok, k = _is_polygonal(n, 5)
    if not ok or k is None:
        return False, None

    details = (
        f"{n} is the {get_ordinal_suffix(k)} pentagonal number "
        f"(since {n} = (3×{k}² − {k})/2)."
    )
    return True, details


@classifier(
    label="Pronic number",
    description="A number of the form n(n+1), the product of "
                "two consecutive integers.",
    oeis="A002378",
    category=CATEGORY
)
def is_pronic_number(n: int) -> tuple[bool, str]:
    """
    n is a pronic number (n = k*(k+1)).
    """
    if n < 0:
        return False, None
    # Solve k^2 + k - n = 0; k = (-1 + sqrt(1 + 4n)) / 2
    k = int((isqrt(1 + 4*n) - 1) // 2)
    if k * (k + 1) == n:
        details = f"{n} = {k} × {k+1}"
        return True, details
    else:
        return False, None


@classifier(
    label="Repunit",
    description="Consists only of digit 1.",
    oeis="A002275",
    category=CATEGORY
)
def is_repunit(n: int) -> tuple[bool, str]:
    """
    A repunit is a number consisting entirely of the digit 1 in base 10
    (e.g., 1, 11, 111).
    """
    if n < 1:
        return False, None

    if set(str(n)) == {'1'}:
        return True, f"{n} is a repunit composed only of digit '1'."
    return False, None


@classifier(
    label="Star number",
    description="n = 6×k(k−1) + 1 (centered hexagram; centered 12-gonal) for some k≥1.",
    oeis="A003154",
    category=CATEGORY,
)
def is_star_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    k = _centered_k(n, 12)
    return (True, _detail(n, 12, k)) if k else (False, None)


@classifier(
    label="Tetrahedral number",
    description="n = k(k+1)(k+2)/6 for some k (A000292).",
    oeis="A000292",
    category=CATEGORY,
)
def is_tetrahedral_number(n: int) -> tuple[bool, str | None]:
    """
    Returns (True, details) iff n is a tetrahedral number; details show k:
      n = k(k+1)(k+2)/6  with k ≥ 0.
    Bigint-safe: no floating point.
    """
    @lru_cache(maxsize=256)
    def tetrahedral_test(n: int) -> tuple[bool, str | None]:
        if n <= 0:
            return False, None

        m = 6 * n
        # integer cube root floor of 6n
        k, _ = integer_nthroot(m, 3)

        # tighten k to the correct side (guard against off-by-one)
        while (k + 1) ** 3 <= m:
            k += 1
        while k > 0 and k ** 3 > m:
            k -= 1

        # check a tiny neighborhood around k
        start = 0 if k < 3 else k - 3
        for i in range(start, k + 4):
            t = i * (i + 1) * (i + 2) // 6
            if t == n:
                return True, f"{n} = {i}×{i+1}×{i+2}/6 (k = {i})"
            if t > n and i > k:
                break

        return False, None

    return tetrahedral_test(n)


@classifier(
    label="Triangular number",
    description="n = k(k+1)/2 for some k.",
    oeis="A000217",
    category=CATEGORY,
)
def is_triangular_number(n: int) -> tuple[bool, str | None]:
    """
    Check if n is a triangular number.
    Triangular numbers satisfy n = k(k+1)/2 for some integer k ≥ 1.
    """
    if n < 1:
        return False, None

    ok, k = _is_polygonal(n, 3)
    if not ok or k is None:
        return False, None

    details = (
        f"{n} is the {get_ordinal_suffix(k)} triangular number "
        f"(since {n} = {k}×({k}+1)/2)."
    )
    return True, details
