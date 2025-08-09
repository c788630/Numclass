# -----------------------------------------------------------------------------
#  Pseudoprimes and cryptographic number test functions
# -----------------------------------------------------------------------------

from decorators import classifier, limited_to
from math import gcd
from sympy import isprime, factorint, jacobi_symbol
from typing import Tuple

CATEGORY = "Pseudoprimes and Cryptographic Numbers"


@classifier(
    label="Carmichael number",
    description="Composite numbers passing Fermat primality tests for multiple bases.",
    oeis="A002997",
    category=CATEGORY
)
@limited_to(9999999)
def is_carmichael_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a Carmichael number, else (False, None).

    Korselt's criterion (1899):
      • n is composite
      • n is square-free
      • For every prime p | n, (p - 1) | (n - 1)
    """
    # Must be composite and ≥ 3; all Carmichael numbers are odd
    if n < 3 or isprime(n) or (n % 2 == 0):
        return False, None

    factors = factorint(n)  # {prime: exponent}

    # Square-free check
    if any(exp != 1 for exp in factors.values()):
        return False, None

    # (Optional speed tweak) Classic theorem: Carmichael numbers have ≥ 3 prime factors
    if len(factors) < 3:
        return False, None

    # Build details; use your bullet style and per-prime divisibility checks
    bullet = "•"
    pf_list = " * ".join(str(p) for p in sorted(factors))
    lines = [f"Prime factors: {pf_list}"]

    ok = True
    for p in sorted(factors):
        r = (n - 1) % (p - 1)
        cond = (r == 0)
        lines.append(f"{bullet} (n-1) % (p-1) = ({n}-1) % ({p}-1) = {r} {'✓' if cond else '✗'}")
        if not cond:
            ok = False
            break

    if not ok:
        return False, None

    lines.append("All conditions satisfied (composite, square-free, and (p−1)|(n−1) for all p).")
    return True, "\n".join(lines)


BASES = (2, 3, 5, 7, 11, 13)


@classifier(
    label="Fermat pseudoprime",
    description="Composite n such that a^(n-1) ≡ 1 mod n for at least one base in {2,3,5,7,11,13}.",
    oeis="A001567",  # base 2 only; multi-base sets are separate sequences
    category=CATEGORY
)
def _is_fermat_pseudoprime(n: int, bases=BASES):
    """
    Fermat pseudoprime test:
    Composite n such that a^(n-1) ≡ 1 mod n for at least one base in bases.
    """
    if n < 3 or isprime(n):
        return False, None

    passing = []
    for a in bases:
        if gcd(a, n) != 1:
            continue
        res = pow(a, n-1, n)
        if res == 1:
            passing.append(f"{a}^{n-1} ≡ 1 (mod {n})")

    if passing:
        return True, "; ".join(passing)
    return False, None


@classifier(
    label="Euler–Jacobi pseudoprime",
    description="Composite n where a^((n-1)//2) ≡ Jacobi(a,n) mod n for at least one base in {2,3,5,7,11,13}.",
    oeis="A047713",
    category=CATEGORY
)
def _is_euler_jacobi_pseudoprime(n: int, bases=BASES):
    """
    Euler–Jacobi pseudoprime test:
    Composite n where a^((n-1)//2) ≡ Jacobi(a, n) mod n for at least one base in bases.
    """
    if n < 3 or n % 2 == 0 or isprime(n):
        return False, None

    passing = []
    exp = (n - 1) // 2
    for a in bases:
        if gcd(a, n) != 1:
            continue
        jac = jacobi_symbol(a, n)
        jac_mod = jac % n
        if jac == -1:
            jac_mod = n - 1
        res = pow(a, exp, n)
        if res == jac_mod:
            passing.append(f"{a}^{exp} ≡ {res} (mod {n}), Jacobi({a},{n})={jac}")

    if passing:
        return True, "; ".join(passing)
    return False, None


@classifier(
    label="Strong pseudoprime",
    description="Composite n passing the Miller–Rabin strong probable prime test for at least one base in {2,3,5,7,11,13}.",
    oeis="A001262",
    category=CATEGORY
)
def _is_strong_pseudoprime(n: int, bases=BASES):
    """
    Strong (Miller–Rabin) pseudoprime test:
    Composite n passing the strong probable prime test for at least one base in bases.
    """
    if n < 3 or n % 2 == 0 or isprime(n):
        return False, None

    # Factor n-1 as d*2^s with d odd
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    passing_bases = []
    for a in bases:
        if gcd(a, n) != 1:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            passing_bases.append(a)
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                passing_bases.append(a)
                break

    if passing_bases:
        bases_str = ", ".join(map(str, passing_bases))
        return True, f"passes strong test bases {bases_str}"
    return False, None
