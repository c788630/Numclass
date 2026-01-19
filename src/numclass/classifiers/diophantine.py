# -----------------------------------------------------------------------------
#  diophantine.py
#  Diophantine representations test
# -----------------------------------------------------------------------------

from __future__ import annotations

from math import isqrt

from numclass.context import NumCtx
from numclass.registry import classifier
from numclass.runtime import CFG
from numclass.utility import build_ctx, load_sum_of_three_cubes_toml

CATEGORY = "Diophantine representations"


# --- helpers ---


def _legendre_three_square_possible(n: int) -> bool:
    """Legendre: n is sum of three squares iff n ≠ 4^a(8b+7)."""
    if n < 0:
        return False
    if n == 0:
        return True
    while n % 4 == 0:
        n //= 4
    return (n % 8) != 7


def r2_sum_of_two_squares(factors: dict[int, int]) -> int:
    """
    r₂(n): number of integer solutions (x, y) with signs and order
    to x² + y² = n. Formula:
      - If any prime p ≡ 3 (mod 4) divides n to an odd power → r₂(n) = 0.
      - Otherwise r₂(n) = 4 * ∏_{p≡1 (mod 4)} (e_p + 1),
        where e_p is the exponent of p in n.
    """
    prod = 1
    for p, e in factors.items():
        if p % 4 == 3 and (e % 2 == 1):
            return 0
        if p % 4 == 1:
            prod *= (e + 1)
    return 4 * prod


# --- classifiers ---


@classifier(
    label="Sum of 2 cubes",
    description="Can be written as n = a³ + b³ for integers a, b.",
    oeis="A003325",
    category=CATEGORY,
    limit=9999999999
)
def is_sum_of_2_cubes(n: int, max_results: int | None = None) -> tuple[bool, str]:
    """
    Returns (True, details) if n can be written as a³ + b³ for integers a, b.
    Shows up to max_results results (None = all),
    smallest |a|,|b| first, unordered a ≤ b.
    ALLOW_ZERO_IN_DECOMP: if False, excludes any result with a==0 or b==0.
    """
    allow_zero = CFG("DIOPHANTINE.ALLOW_ZERO_IN_DECOMP", False)
    cap = CFG("DIOPHANTINE.MAX_SOL_SUM_OF_2_CUBES", 20)

    absn = abs(n)
    B = round(absn ** (1/3)) + 2  # simple bound

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


# exact integer cube root: return int if k^3 == m else None
def _icbrt_exact(m: int) -> int | None:
    if m == 0:
        return 0
    sgn = -1 if m < 0 else 1
    a = abs(m)
    # initial guess, then correct
    k = round(a ** (1.0/3.0))
    # nudge until exact
    while k * k * k < a:
        k += 1
    while k * k * k > a:
        k -= 1
    return sgn * k if k * k * k == a else None


@classifier(
    label="Sum of 3 cubes",
    description="Can be written as n = a³ + b³ + c³ for integers a, b, c.",
    oeis="A060464",
    category=CATEGORY,
    limit=10**8-1,
)
def is_sum_of_3_cubes(n: int, max_results: int | None = None) -> tuple[bool, str | None]:
    """
    Show up to max_results solutions (canonical a<=b<=c) and ALWAYS include the
    known 'celebrity' solution (from TOML) if available, even when capped.
    """
    # quick obstructions
    if n == 0:
        return False, None
    if n % 9 in (4, 5):
        return False, None

    # settings
    allow_zero = bool(CFG("DIOPHANTINE.ALLOW_ZERO_IN_DECOMP", True))
    max_results = CFG("DIOPHANTINE.MAX_SOL_SUM_OF_3_CUBES", 20)
    B = CFG("DIOPHANTINE.MAX_ABS_FOR_SUM_OF_3_CUBES", 100)

    seen: set[tuple[int, int, int]] = set()
    listed: list[tuple[int, int, int]] = []

    # search a<=b and compute c by exact cube root
    for a in range(-B, B + 1):
        a3 = a * a * a
        for b in range(a, B + 1):  # enforce a <= b
            need = n - a3 - b * b * b
            c = _icbrt_exact(need)
            if c is None:
                continue
            if b <= c and (allow_zero or (a != 0 and b != 0 and c != 0)):
                triple = (a, b, c)
                if triple not in seen:
                    seen.add(triple)
                    if max_results is None or len(listed) < max_results:
                        listed.append(triple)

    # celebrity injection (new strict format: [solutions] N = [a,b,c])
    raw_map = load_sum_of_three_cubes_toml("sum_of_three_cubes.toml")
    raw = raw_map.get(n)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        ca, cb, cc = int(raw[0]), int(raw[1]), int(raw[2])
        celeb = tuple(sorted((ca, cb, cc)))
        if (allow_zero or all(v != 0 for v in celeb)) and celeb not in seen:
            if max_results is None or len(listed) < max_results:
                listed.append(celeb)
            else:
                listed[-1] = celeb
            seen.add(celeb)

    if not seen:
        return False, None

    def fmt(x: int) -> str:
        return f"({x})³" if x < 0 else f"{x}³"

    parts = [f"{fmt(a)} + {fmt(b)} + {fmt(c)}" for (a, b, c) in listed]
    details = f"{n} = " + "; ".join(parts) if parts else f"found {len(seen)}"
    return True, details


@classifier(
    label="Sum of 2 squares",
    description="Can be written as n = a² + b² for integers a, b.",
    oeis="A001481",
    category=CATEGORY,
    limit=99_000_000_000_000

)
def is_sum_of_2_squares(n: int,
                        ctx: NumCtx | None = None) -> tuple[bool, str]:
    """
    Returns (True, details) if n can be written as a^2 + b^2 for integers a, b.
    Shows up to max_results decompositions with a >= 0, b >= 0 and a <= b.
    Uses Fermat's theorem for quick check.
    """
    if n < 0:
        return False, None

    allow_zero = CFG("DIOPHANTINE.ALLOW_ZERO_IN_DECOMP", False)
    cap = CFG("DIOPHANTINE.MAX_SOL_SUM_OF_2_SQUARES", 20)

    N = n
    # exact total (ordered, with signs)
    ctx = ctx or build_ctx(n)
    total_ordered = r2_sum_of_two_squares(ctx.fac)
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
        if x * x + y * y == N and x <= y and (allow_zero or (x > 0 and y > 0)):
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
        details = "total 0 (ordered, with signs), 0 canonical"

    return True, details


@classifier(
    label="Sum of 3 squares",
    description="Can be written as n = a² + b² + c² for integers a, b, c.",
    oeis="A000378",
    category=CATEGORY,
    limit=21_000_000
)
def is_sum_of_3_squares(n: int) -> tuple[bool, str]:
    """
    Fast check with Legendre's theorem; finds up to cap canonical triples
    (x≤y≤z), x,y,z≥0 unless ALLOW_ZERO_IN_DECOMP is False (then x,y,z>0).
    """
    if n < 0:
        return False, None
    N = n

    # Legendre quick reject
    if not _legendre_three_square_possible(N):
        return False, None

    allow_zero = CFG("DIOPHANTINE.ALLOW_ZERO_IN_DECOMP", False)
    cap = CFG("DIOPHANTINE.MAX_SOL_SUM_OF_3_SQUARES", 20)

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
        j = min(j, r)  # clamp

        while i <= j:
            s = squares[i] + squares[j]
            if s == T:
                x, y = i, j
                if (s == T and (allow_zero or (x > 0 and y > 0 and z > 0)) and x <= y <= z):
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
