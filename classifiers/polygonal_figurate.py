# -----------------------------------------------------------------------------
#  Polygonal and Figurate Numbers test functions
# -----------------------------------------------------------------------------

from decorators import classifier
from functools import lru_cache
from math import log2, isqrt
from sympy.ntheory.primetest import is_square
from utility import base_name, get_ordinal_suffix
from typing import List, Tuple


CATEGORY = "Polygonal and Figurate Numbers"


@classifier(
    label="Cyclic number",
    description="Multiples are cyclic rotations of digits.",
    oeis="A003285",
    category=CATEGORY
)
def is_cyclic_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a cyclic number.
    A cyclic number is an integer in which cyclic permutations of the digits
    are successive multiples of the number.
    For example, 142857 × 2 = 285714, a cyclic permutation of 142857.
    Very few cyclic numbers are known, using a list to save calculation time.
    """
    KNOWN = {142857, 588235294117647, 52631578947368421}
    if n in KNOWN:
        details = (
            f"{n} is a known cyclic number. "
            "Cyclic multiples are digit rotations of the number itself."
        )
        return True, details
    return False, None


def _is_polygonal(n: int, s: int) -> Tuple[bool, str]:
    """
    Helper function to test if n is a polygonal number with s sides.
    Recognized: triangular (3), pentagonal (5), hexagonal (6).
    """
    names = {3: "triangular", 5: "pentagonal", 6: "hexagonal"}
    if n < 1 or s not in names:
        return False, None
    a, b, c = s - 2, -(s - 4), -2 * n
    disc = b * b - 4 * a * c
    if disc < 0:
        return False, None
    root = isqrt(disc)
    if root * root != disc:
        return False, None
    k = (-(b) + root) / (2 * a)
    if k.is_integer() and k > 0:
        idx = int(k)
        kind = names[s]
        details = (
            f"{n} is the {get_ordinal_suffix(idx)} {kind} number "
            f"(formula: n = (({s}-2)×k² - ({s}-4)×k)/2, with k={idx})"
        )
        return True, details
    return False, None


@classifier(
    label="Hexagonal number",
    description="n = k(2k−1) for some k.",
    oeis="A000384",
    category=CATEGORY
)
def is_hexagonal_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a hexagonal number.
    A hexagonal number represents a pattern of dots that form a hexagon.
    The nth hexagonal number is n = k(2k−1) for some integer k ≥ 1.
    """
    if n < 1:
        return False, None
    return _is_polygonal(n, 6)


def _lehmer_factors(n: int) -> List[Tuple[int, int]]:
    """
    Helper function to find (a, k) pairs for which n = (a^k-1)//(a-1),
    for bases 2..10.
    """
    if n < 3:
        return []
    results = []
    max_k = int(log2(n + 1)) + 2
    for k in range(2, max_k):
        for a in range(2, 11):  # Only bases 2 to 10 inclusive
            if (a ** k - 1) == n * (a - 1):
                results.append((a, k))
    return results


@classifier(
    label="Lehmer number",
    description="Equals (a^k−1)/(a−1) for some integers 2≤a≤10, k>1; "
                "in base a, written as k consecutive 1's.",
    oeis=None,
    category=CATEGORY
)
def is_lehmer_number(n: int) -> Tuple[bool, str]:
    """
    Lehmer number generalizes repunits and Mersenne numbers.
    """
    if n < 1:
        return False, None
    facts = _lehmer_factors(n)
    if facts:
        # Build pretty details: e.g. "a=2, k=5 [binary (base 2): 11111]"
        def ones(k): return "1" * k
        parts = []
        for a, k in facts:
            base_str = base_name(a)
            num_str = ones(k)
            parts.append(f"{base_str}: {num_str}")
        detail_str = "; ".join(parts)
        return True, detail_str
    return False, None


@classifier(
    label="Pentagonal number",
    description="n = k(3k−1)/2 for some k.",
    oeis="A000326",
    category=CATEGORY
)
def is_pentagonal_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a pentagonal number.
    A pentagonal number counts dots that can form a pentagon.
    The nth pentagonal number is n = k(3k−1)/2 for some integer k ≥ 1.
    """
    if n < 1:
        return False, None
    return _is_polygonal(n, 5)


@classifier(
    label="Perfect cube",
    description="n = k^3 for some integer k.",
    oeis="A000578",
    category=CATEGORY
)
def is_perfect_cube(n: int) -> Tuple[bool, str]:
    """
    Check if n is a perfect cube.
    Returns (True, details) if so, else False.
    A perfect cube is an integer that can be written as n = k^3,
    for some integer k.
    """
    if n == 0:
        return True, "0 is 0³ (perfect cube)."
    k = round(abs(n) ** (1/3))
    if n < 0:
        k = -k
    if k ** 3 == n:
        return True, f"{n} = {k}³ (perfect cube)."
    return False, None


@classifier(
    label="Perfect square",
    description="n = k² for some integer k.",
    oeis="A000290",
    category=CATEGORY
)
def is_perfect_square(n: int) -> Tuple[bool, str]:
    """
    Check if n is a perfect square using SymPy.
    A perfect square is an integer that is the square
    of another integer; n = k².
    """
    if n < 1:
        return False, None
    if is_square(n):
        k = isqrt(n)
        return True, f"{n} = {k}² (perfect square)."
    return False, None


@classifier(
    label="Pronic number",
    description="A number of the form n(n+1), the product of "
                "two consecutive integers.",
    oeis="A002378",
    category=CATEGORY
)
def is_pronic_number(n: int) -> Tuple[bool, str]:
    """
    n is a pronic number (n = k*(k+1)).
    """
    if n < 0:
        return False, None
    # Solve k^2 + k - n = 0; k = (-1 + sqrt(1 + 4n)) / 2
    import math
    k = int((math.isqrt(1 + 4*n) - 1) // 2)
    if k * (k + 1) == n:
        details = f"{n} = {k} × {k+1}"
        return True, details
    else:
        return False, None


@classifier(
    label="Repunit",
    description="Consists only of digit 1.",
    oeis="A001318",
    category=CATEGORY
)
def is_repunit(n: int) -> Tuple[bool, str]:
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
    label="Tetrahedral number",
    description="n = k(k+1)(k+2)/6 for some k.",
    oeis="A000292",
    category=CATEGORY
)
def is_tetrahedral_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a tetrahedral number, else False.
    Details show the value of k: n = k(k+1)(k+2)/6
    """
    @lru_cache(maxsize=256)
    def tetrahedral_test(n):
        if n < 1:
            return False, None
        k = int((6 * n) ** (1 / 3))
        for i in range(max(1, k - 2), k + 3):
            if i * (i + 1) * (i + 2) // 6 == n:
                details = f"{n} = {i}×{i+1}×{i+2}/6 (k = {i})"
                return True, details
        return False, None
    return tetrahedral_test(n)


@classifier(
    label="Triangular number",
    description="n = k(k+1)/2 for some k.",
    oeis="A000217",
    category=CATEGORY
)
def is_triangular_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a triangular number.
    Triangular numbers are numbers that can form an equilateral triangle.
    The nth triangular number is the sum of the first n natural numbers:
    n = k(k+1)/2 for some k ≥ 1.
    """
    if n < 1:
        return False, None
    return _is_polygonal(n, 3)
