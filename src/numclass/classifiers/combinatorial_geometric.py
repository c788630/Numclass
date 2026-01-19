# -----------------------------------------------------------------------------
#  combinatorial_geometric.py
#  Counting structures, partitions, and geometric divisions.
# -----------------------------------------------------------------------------

from __future__ import annotations

import functools
import time
from math import factorial as fact

from sympy import binomial, integer_nthroot

from numclass.fmt import abbr_int_fast
from numclass.registry import classifier
from numclass.runtime import current as _rt_current
from numclass.utility import CFG, _fmt_pairs_ramsey, _fmt_pairs_vdw, check_oeis_bfile

try:
    from pysat.formula import CNF
    from pysat.solvers import Solver as _PySATNames  # for Solver.names()
    from pysat.solvers import Solver as _PySATSolver
    _HAS_PYSAT = True
except Exception:  # no pysat installed
    _HAS_PYSAT = False
    _PySATSolver = None
    _PySATNames = None
    CNF = None


CATEGORY = "Combinatorial and Geometric"


# Schur numbers — Version 1 (OEIS A030126)
# SAT-based exact checker: proves n = S′(k) by UNSAT([1..n], k) & SAT([1..n-1], k).
# Requires PySAT (python-sat). Falls back to a CSP backtracker if PySAT not available.

# --- Schur number helpers ---

def _schur_triples_upto(n: int):
    T = []
    for x in range(1, n+1):
        for y in range(x, n+1):   # x ≤ y
            z = x + y
            if z <= n:
                T.append((x, y, z))
    return T


def _pick_solver_name() -> str | None:
    if not _HAS_PYSAT:
        return None
    from pysat.solvers import Solver  # noqa: PLC0415  (optional dependency; lazy import)

    names_attr = getattr(Solver, "names", None)
    if callable(names_attr):
        try:
            names = names_attr()
        except Exception:
            names = []
    elif isinstance(names_attr, (list, tuple)):
        names = list(names_attr)
    else:
        names = []
    names = [str(n).lower() for n in names if n]

    for cand in ("cadical153", "minisat22", "glucose4", "glucose3", "maplechrono", "maplecm"):
        if cand in names:
            return cand
    return names[0] if names else None


def _var(i: int, c: int, k: int) -> int:
    # 1..n mapped to colors 0..k-1  →  positive int variable
    # v(i,c) == 1  ↔ "number i has color c"
    # simple row-major encoding: i*k + c + 1
    return i * k + c + 1


def _encode_schur_cnf(n: int, k: int, *, symmetry_basic: bool = True, symmetry_ladder: bool = False) -> CNF:
    cnf = CNF()
    # triples x+y=z (x ≤ y)
    triples = []
    for x in range(1, n+1):
        for y in range(x, n+1):
            z = x + y
            if z <= n:
                triples.append((x, y, z))

    def var(i: int, c: int) -> int:
        # Use (i-1)*k + (c+1) to keep variable ids compact and standard.
        return (i - 1) * k + (c + 1)

    # Exactly-one color per i
    for i in range(1, n+1):
        # at least one
        cnf.append([var(i, c) for c in range(k)])
        # at most one (pairwise)
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                cnf.append([-var(i, c1), -var(i, c2)])

    # Forbid monochromatic Schur triples
    for (x, y, z) in triples:
        for c in range(k):
            cnf.append([-var(x, c), -var(y, c), -var(z, c)])

    # Symmetry breaking
    if symmetry_basic:
        # Fix color(1) = 0
        cnf.append([var(1, 0)])
        for c in range(1, k):
            cnf.append([-var(1, c)])

    if symmetry_ladder and k >= 2:
        # Only enable if smoke tests pass; otherwise keep it off
        for c in range(1, k):
            prefix = []
            for i in range(1, n+1):
                prefix.append(var(i, c - 1))
                cnf.append([-var(i, c), *prefix])

    return cnf


def _sat_colorable(n: int, k: int, *, use_ladder: bool | None = None) -> tuple[bool | None, str | None]:
    """
    Returns (SAT?, backend_name)
      SAT?          → True / False / None
      backend_name  → 'Minisat22', 'Glucose3', 'Cadical153', or 'CSP-backtracker'
    """
    if not _HAS_PYSAT:
        return _csp_colorable(n, k), "CSP-backtracker"

    cnf = _encode_schur_cnf(n, k, symmetry_basic=True, symmetry_ladder=bool(use_ladder))
    picked = _pick_solver_name()

    if picked:
        with _PySATSolver(name=picked, bootstrap_with=cnf.clauses) as S:
            ok = S.solve()
            inner = getattr(S, "solver", None)  # the real backend object
            backend = type(inner).__name__ if inner is not None else picked
    else:
        with _PySATSolver(bootstrap_with=cnf.clauses) as S:
            ok = S.solve()
            inner = getattr(S, "solver", None)
            # if inner missing, fall back to something informative
            backend = type(inner).__name__ if inner is not None else "auto"

    return bool(ok), backend


def _csp_colorable(n: int, k: int, time_budget_s: float = 12.0) -> bool | None:
    """
    Returns True if [1..n] is k-colorable with no monochromatic Schur triple x+y=z,
            False if UNSAT, None on timeout.
    Symmetry: only fix color(1)=0.  (No sequential color-introduction.)
    """

    start = time.time()

    # Build triples x+y=z with x <= y and z <= n
    T: list[tuple[int, int, int]] = []
    for x in range(1, n + 1):
        for y in range(x, n + 1):
            z = x + y
            if z <= n:
                T.append((x, y, z))

    # Incidence: for each v in 1..n, list (triple-index, role: 0=x,1=y,2=z)
    inc: list[list[tuple[int, int]]] = [[] for _ in range(n + 1)]
    for ti, (x, y, z) in enumerate(T):
        inc[x].append((ti, 0))
        inc[y].append((ti, 1))
        inc[z].append((ti, 2))

    # Variable order: most constrained first (highest triple-degree)
    order = list(range(1, n + 1))
    order.sort(key=lambda v: (-len(inc[v]), v))

    UN = -1
    color = [UN] * (n + 1)
    # forb[c][v] = True forbids color c on variable v
    forb = [[False] * (n + 1) for _ in range(k)]

    # Symmetry break: fix 1 -> color 0
    color[1] = 0
    used_colors = 1  # colors 0..used_colors-1 are currently in use

    def timed_out() -> bool:
        return (time.time() - start) > time_budget_s

    def propagate(v: int, c: int, stack: list[tuple[int, int]]) -> bool:
        """
        After assigning color[v] = c, enforce 'no monochromatic x+y=z':
          - If both other members already have color c -> conflict.
          - If exactly one other member has color c -> forbid c on the remaining one.
        """
        for ti, role in inc[v]:
            x, y, z = T[ti]
            if role == 0:      # v == x
                o1, o2 = y, z
            elif role == 1:    # v == y
                o1, o2 = x, z
            else:              # v == z
                o1, o2 = x, y

            c1 = (o1 <= n and color[o1] == c)
            c2 = (o2 <= n and color[o2] == c)

            if c1 and c2:
                return False  # would create a monochromatic triple

            if c1 ^ c2:
                rem = o2 if c1 else o1
                if color[rem] == c:
                    return False  # defensive; should be caught above
                if not forb[c][rem]:
                    forb[c][rem] = True
                    stack.append((c, rem))
        return True

    def domain_empty(v: int, used: int) -> bool:
        """
        Empty domain test allowing either an already used color (0..used-1)
        or introducing ONE new color (index 'used') as long as used<k.
        """
        max_try = min(used + 1, k)  # include the "next new color" slot
        return all(forb[c][v] for c in range(max_try))

    def dfs(idx: int, used: int):
        if timed_out():
            return None
        if idx == len(order):
            return True

        v = order[idx]
        if color[v] != UN:
            return dfs(idx + 1, used)

        # Try any existing color 0..used-1, and optionally introduce 'used' if used<k
        max_c = min(used, k - 1)  # if used==k, no new colors remain
        for c in range(0, max_c + 1):
            if forb[c][v]:
                continue

            # assign
            color[v] = c
            stack: list[tuple[int, int]] = []
            ok = propagate(v, c, stack)

            if ok:
                # Fail-fast: unassigned variable with empty domain?
                doomed = False
                for u in range(1, n + 1):
                    if color[u] == UN and domain_empty(u, used if c < used else used + (1 if c == used and used < k else 0)):
                        doomed = True
                        break
                ok = not doomed

            if ok:
                # bump 'used' if we just introduced a new color (c == used)
                new_used = used + 1 if c == used and used < k else used
                r = dfs(idx + 1, new_used)
                if r is not False and r is not None:
                    return r

            # undo
            color[v] = UN
            while stack:
                cc, vv = stack.pop()
                forb[cc][vv] = False

        return False

    return dfs(0, used_colors)


def _is_Sprime_k_via_sat(n: int, k: int) -> tuple[str, str | None]:
    """
    Returns (verdict, solver_name)
       verdict = "yes" / "no" / "unknown"
    """
    sat_n, solver = _sat_colorable(n, k, use_ladder=False)
    if sat_n is True:
        return "no", solver
    sat_nm1, _ = _sat_colorable(n - 1, k, use_ladder=False)
    if sat_nm1 is True and sat_n is False:
        return "yes", solver
    if sat_n is False and sat_nm1 is False:
        return "no", solver
    return "unknown", solver


# --- Classifiers ---


@classifier(
    label="Bell number",
    description="Number of partitions of an n-element set (B(k)).",
    oeis="A000110",
    category=CATEGORY,
    # NOTE: no limit here — avoid pre-skip for huge values
)
def is_bell_number(n: int) -> tuple[bool, str | None]:
    if n < 1:
        return False, None
    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    budget_s = CFG("COMBINATORIAL.TIME_BUDGET_S", 0.5)
    max_k = CFG("CLASSIFIER.BELL.MAX_K", 2500)
    t0 = time.perf_counter()

    def ok() -> bool:
        return (not fast_mode) or (time.perf_counter() - t0) < budget_s

    @functools.cache
    def S2_row(k: int) -> tuple[int, ...]:
        if k == 0:
            return (1,)
        prev = S2_row(k - 1)
        cur = [0] * (k + 1)
        cur[k] = 1
        for j in range(1, k):
            cur[j] = j * prev[j] + prev[j - 1]
        return tuple(cur)

    # compute B(k) = sum_j {k \brace j} until we pass n or time out
    for k in range(0, max_k + 1):
        if not ok():
            break
        Bk = sum(S2_row(k))
        if Bk == n:
            return True, f"{abbr_int_fast(n)} is Bell number B({k}) (computed)."
        if Bk > n:
            break

    # fallback: OEIS b-file
    found, idx_file, series, idx_set = check_oeis_bfile("b000110.txt", n)
    if found:
        k = idx_file if idx_file is not None else (idx_set if idx_set is not None else "?")
        return True, f"{abbr_int_fast(n)} is Bell number B({k}) (from OEIS b-file)."
    return False, None


@classifier(
    label="Cake number",
    description="Max pieces from k planar cuts in 3D: C(k) = (k^3 + 5k + 6)//6.",
    oeis="A000125",
    category=CATEGORY,
)
def is_cake_number(n: int) -> tuple[bool, str | None]:
    """
    Return (True, details) iff n = C(k) for some integer k ≥ 0, where
    C(k) = (k^3 + 5k + 6)//6. Uses integer arithmetic only.
    """
    if n < 1:
        return False, None

    # k ≈ (6n)^(1/3) is an excellent integer starting point
    m = 6 * n
    k_est, exact = integer_nthroot(m, 3)  # floor(cuberoot(m)), exact flag unused

    # Check a tiny neighborhood around the estimate; C(k) is strictly increasing for k≥0
    for kk in (k_est - 2, k_est - 1, k_est, k_est + 1, k_est + 2):
        if kk >= 0:
            pieces = (kk*kk*kk + 5*kk + 6) // 6
            if pieces == n:
                return True, f"{n} is the cake number for k={kk} straight cuts."
            # Optional early exit if we passed it (monotone in kk)
            if pieces > n and kk > k_est + 2:
                break

    return False, None


@classifier(
    label="Catalan number",
    description="Cₙ = binomial(2n,n)/(n+1), counts Dyck paths.",
    oeis="A000108",
    category=CATEGORY,
)
def is_catalan_number(n: int) -> tuple[bool, str | None]:
    if n < 1:
        return False, None

    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    budget_s = CFG("COMBINATORIAL.TIME_BUDGET_S", 0.5)
    max_k = CFG("CLASSIFIER.CATALAN.MAX_K", 200000)  # index, not value
    t0 = time.perf_counter()

    def ok() -> bool:
        if not fast_mode:
            return True
        return (time.perf_counter() - t0) < budget_s

    # Heuristic inverse: C_k ~ 4^k / (k^{3/2} * sqrt(pi)) → start near this and adjust.
    # But we can also just grow until we exceed n, with a cap & budget.
    C = 1
    k = 0
    while k <= max_k and n >= C and ok():
        if n == C:
            return True, f"{abbr_int_fast(n)} is Catalan number C({k})."
        k += 1
        C = binomial(2*k, k) // (k + 1)
    # fallback
    found, idx_file, *_ = check_oeis_bfile("b000108.txt", n)
    if found:
        k = idx_file if idx_file is not None else "?"
        return True, f"{abbr_int_fast(n)} is Catalan number C({k})."
    return False, None


@classifier(
    label="Motzkin number",
    description="Counts certain lattice paths of length n.",
    oeis="A001006",
    category=CATEGORY,
)
def is_motzkin_number(n: int) -> tuple[bool, str | None]:
    if n < 1:
        return False, None

    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    budget_s = float(CFG("COMBINATORIAL.TIME_BUDGET_S", 0.5))
    max_k = int(CFG("CLASSIFIER.MOTZKIN.MAX_K", 200000))
    t0 = time.perf_counter()

    def ok() -> bool:
        if not fast_mode:
            return True
        return (time.perf_counter() - t0) < budget_s

    # M_0=1, M_1=1, M_{n+1} = ((2n+1)M_n + (3n-3)M_{n-1})/(n+2)
    M0, M1 = 1, 1
    if n == 1:
        return True, "1 is Motzkin number M(0) and M(1)."
    k = 1
    while k < max_k and n >= M1 and ok():
        if n == M1:
            return True, f"{abbr_int_fast(n)} is Motzkin number M({k-1})."
        k += 1
        M2 = ((2*k - 1) * M1 + (3*k - 6) * M0) // (k + 1)
        M0, M1 = M1, M2

    found, idx_file, *_ = check_oeis_bfile("b001006.txt", n)
    if found:
        k = idx_file if idx_file is not None else "?"
        return True, f"{abbr_int_fast(n)} is Motzkin number M({k})."
    return False, None


@classifier(
    label="Schur number",
    description=" (v1, exact) Certifies n = S′(k) by UNSAT([1..n],k) and SAT([1..n−1],k); SAT-based with CSP fallback.",
    oeis="A030126",
    category=CATEGORY,
    limit=45
)
def is_schur_number_v1(n: int, k_max: int = 5) -> tuple[bool, str | None]:
    if n <= 1:
        return False, "n must be > 1"
    if not _HAS_PYSAT and k_max > 4:
        k_max = 4  # CSP is practical for k≤4 only
    for k in range(1, k_max+1):
        verdict, solver = _is_Sprime_k_via_sat(n, k)
        if verdict == "yes":
            return True, f"{n} is Schur number (v1) {k}: S′({k})={n} (proved via SAT using {solver})."
    return False, None


def _find_stirling2_pairs(m: int, max_pairs: int = 6) -> list[tuple[int, int]]:
    if m < 0:
        return []
    # S rows as lists indexed by k=0..n
    S_prev = [1]  # n=0: [S(0,0)]
    hits: list[tuple[int, int]] = []
    if m == 1:
        # Many hits: S(n,1)=1 and S(n,n)=1 for all n>=1 (and S(0,0)=1).
        # We'll still generate a few concrete small pairs below.
        pass

    n = 0
    while True:
        # scan current row (n)
        for k, val in enumerate(S_prev):
            if val == m:
                hits.append((n, k))
                if len(hits) >= max_pairs:
                    return hits

        # stopping conditions before building next row:
        # If m != 1 and for some n>=2 we have S(n,2)=2^{n-1}-1 > m and S(n,n-1)=C(n,2) > m,
        # then for all larger n, every inner entry > m (edges are 1's).
        if (
            m != 1
            and n >= 2
            and (2**(n-1) - 1) > m
            and (n * (n - 1) // 2) > m
        ):
            return hits

        # build next row (n+1)
        n1 = n + 1
        S_next = [0] * (n1 + 1)
        S_next[0] = 0 if n1 > 0 else 1
        S_next[n1] = 1
        for k in range(1, n1):
            # k in 1..n
            S_next[k] = k * S_prev[k] + S_prev[k - 1]
        S_prev = S_next
        n = n1


@classifier(
    label="Stirling number (2nd kind)",
    description="Checks if n equals S(N,K): partitions of an N-set into K nonempty unlabeled blocks.",
    oeis="A008277",
    category=CATEGORY,
)
def is_stirling_second(m: int) -> tuple[bool, str | None]:
    if m < 0:
        return False, None

    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    budget_s = float(CFG("COMBINATORIAL.TIME_BUDGET_S", 0.5))
    t0 = time.perf_counter()

    def ok() -> bool:
        if not fast_mode:
            return True
        return (time.perf_counter() - t0) < budget_s

    pairs = []
    S_prev = [1]  # n=0
    n = 0
    while ok():
        for k, val in enumerate(S_prev):
            if val == m:
                pairs.append((n, k))
                if len(pairs) >= 6:
                    break
        if pairs and len(pairs) >= 6:
            break
        if m != 1 and n >= 2 and (2**(n-1) - 1) > m and (n*(n-1)//2) > m:
            break
        n1 = n + 1
        S_next = [0] * (n1 + 1)
        S_next[n1] = 1
        for k in range(1, n1):
            S_next[k] = k * S_prev[k] + S_prev[k - 1]
        S_prev = S_next
        n = n1

    if not pairs:
        return False, None
    shown = ", ".join([f"(N={n}, K={k})" for (n, k) in pairs[:5]])
    more = "" if len(pairs) <= 5 else f" (+{len(pairs)-5} more)"
    return True, f"{abbr_int_fast(m)} = S(N,K) for {shown}{more}."

# ---------- Unsigned Stirling numbers of the 1st kind ----------
# Recurrence: c(n,k) = c(n-1,k-1) + (n-1)*c(n-1,k)   with c(0,0)=1; c(n,0)=0 for n>0; c(n,n)=1
# (These count permutations of n elements with exactly k cycles.)


def _find_stirling1u_pairs(m: int, max_pairs: int = 6) -> list[tuple[int, int]]:
    if m < 0:
        return []
    C_prev = [1]  # n=0: [c(0,0)]
    hits: list[tuple[int, int]] = []

    n = 0
    while True:
        # scan current row (n)
        for k, val in enumerate(C_prev):
            if val == m:
                hits.append((n, k))
                if len(hits) >= max_pairs:
                    return hits

        # stop when inner extremes exceed m:
        # c(n,2) (grows super fast) and c(n,n-1)=C(n,2); once both > m (and m != 1 edge),
        # larger n won't produce new matches (except 1's on the diagonal).
        if m != 1 and n >= 2:
            cn_n1 = (n * (n - 1)) // 2  # c(n, n-1) = C(n,2)
            # We don't know c(n,2) in closed form cheaply, but we have it in the row:
            cn_2 = C_prev[2] if len(C_prev) > 2 else (1 if n == 2 else 0)
            if cn_2 > m and cn_n1 > m:
                return hits

        # build next row (n+1)
        n1 = n + 1
        C_next = [0] * (n1 + 1)
        C_next[0] = 0 if n1 > 0 else 1
        C_next[n1] = 1
        for k in range(1, n1):
            C_next[k] = C_prev[k - 1] + (n) * C_prev[k]  # (n-1) where n is previous size; here n == current n
        C_prev = C_next
        n = n1


@classifier(
    label="Stirling number (1st kind)",
    description="Checks if n equals c(N,K)=|s(N,K)|: permutations of N elements with exactly K cycles.",
    oeis="A132393",
    category=CATEGORY,
    limit=1_200_000
)
def is_stirling_first_unsigned(m: int) -> tuple[bool, str | None]:
    if m < 0:
        return False, None
    pairs = _find_stirling1u_pairs(m, max_pairs=6)
    if not pairs:
        return False, None

    shown = ", ".join([f"(N={n}, K={k})" for (n, k) in pairs[:5]])
    more = "" if len(pairs) <= 5 else f" (+{len(pairs)-5} more)"
    return True, f"{abbr_int_fast(m)} = c(N,K) (unsigned Stirling-1) for {shown}{more}."


# --------------------------------------------
# Ramsey number Exact and conjectured values
# --------------------------------------------

# Canonicalize pairs as (min(s,t), max(s,t))
_EXACT_R_ST: dict[tuple[int, int], int] = {
    (1, 1): 1,
    (2, 2): 2,
    (3, 3): 6,
    (3, 4): 9,
    (3, 5): 14,
    (3, 6): 18,
    (3, 7): 23,
    (3, 8): 28,
    (3, 9): 36,
    (4, 4): 18,
    (4, 5): 25,
    # Small reliable table; can extend when new values are proved.
}

# Build reverse map m -> [(s,t), ...] for quick classification
_R_VALUE_TO_ST: dict[int, list[tuple[int, int]]] = {}
for (s, t), val in _EXACT_R_ST.items():
    _R_VALUE_TO_ST.setdefault(val, []).append((s, t))


# ---- A120414 diagonal conjecture (R(k,k)) ---------------------------------
# a(0) = 0, a(1) = 1; a(n) = ceil((3/2)^(n-3) * n * (n-1)) for n >= 2.

_A120414_MAX_K = 40  # plenty for realistic inputs
_A120414_VALUE_TO_K: dict[int, int] = {}


def _init_a120414_cache() -> None:
    """Precompute a(n) for 0 <= n <= _A120414_MAX_K and build value -> k map."""
    for k in range(0, _A120414_MAX_K + 1):
        if k == 0:
            val = 0
        elif k == 1:
            val = 1
        elif k == 2:
            val = 2
        else:
            # a(k) = ceil((3/2)^(k-3) * k * (k-1))
            # Use exact integer arithmetic: (3/2)^(k-3) = 3^(k-3) / 2^(k-3)
            e = k - 3
            num = pow(3, e) * k * (k - 1)
            den = pow(2, e)
            val = (num + den - 1) // den  # ceil(num / den)
        # If the same value appears twice, keep the smallest k
        _A120414_VALUE_TO_K.setdefault(val, k)


_init_a120414_cache()


@classifier(
    label="Ramsey number (2-colour)",
    description=(
        "Checks if n equals a 2-colour graph Ramsey number R(s,t) or diagonal(k, k) (Conjectured values for k≥5)."
    ),
    oeis="A059442",  # A120414 for conjectured values using a(k) = ceil((3/2)**(k-3)*k*(k-1))
    category=CATEGORY,
)
def is_ramsey_number(m: int) -> tuple[bool, str | None]:
    """
    Returns (True, details) if m is:
      • an exact 2-colour Ramsey number R(s,t) in the built-in table (A059442), and/or
      • a diagonal value R(k,k) matching the A120414 formula.

    Details indicate whether the match is exact or conjectured.
    """
    if m <= 0:
        return False, None

    # Exact values from the small R(s,t) table
    pairs_exact = _R_VALUE_TO_ST.get(m, [])

    # Diagonal candidate from A120414 (R(k,k))
    k_diag = _A120414_VALUE_TO_K.get(m)

    # Nothing matched at all
    if not pairs_exact and k_diag is None:
        return False, None

    parts: list[str] = []

    if pairs_exact:
        parts.append(f"{_fmt_pairs_ramsey(pairs_exact)} (exact)")

    if k_diag is not None:
        # For k <= 4, R(k,k) is known exactly and included in the table above.
        # For k >= 5 the value is conjectured via A120414.
        if k_diag <= 4:
            # It *should* already appear in pairs_exact, but we word this nicely.
            parts.append(
                f"diagonal R({k_diag},{k_diag}) (exact)"
            )
        else:
            parts.append(
                f"conjectured diagonal R({k_diag},{k_diag}): "
                f"a({k_diag}) = ceil((3/2)**({k_diag}-3)*{k_diag}*({k_diag}-1)) = {m}"

            )

    detail = f"{m} = " + " and ".join(parts) + "."
    return True, detail


# --------------------------------------------
# Van der Waerden numbers (exact recognizer)
# --------------------------------------------
# W(r,k): smallest N such that every r-coloring of {1..N} has a k-term
# monochromatic arithmetic progression.

# Non-trivial exact values (r >= 2, k >= 3) — curated from current sources.
# (Trivial families W(1,k)=k and W(r,2)=r+1 are handled optionally below.)
_VDW_EXACT: dict[tuple[int, int], int] = {
    (2, 3): 9,
    (2, 4): 35,
    (2, 5): 178,
    (2, 6): 1132,
    (3, 3): 27,
    (3, 4): 293,
    (4, 3): 76,
}

# Reverse map: value -> list of (r,k) that achieve it (here, each value is unique)
_VDW_VALUE_TO_RK: dict[int, list[tuple[int, int]]] = {}
for rk, val in _VDW_EXACT.items():
    _VDW_VALUE_TO_RK.setdefault(val, []).append(rk)


# Optional: recognize trivial identities (disabled by default)
def _vdw_trivial_witnesses(n: int, include_trivial=False) -> list[tuple[int, int]]:
    """
    If include_trivial=True, return a small set of (r,k) witnesses for:
      - W(1,k) = k  (choose r=1)
      - W(r,2) = r+1
    We cap witnesses to keep output short.
    """
    if not include_trivial or n <= 0:
        return []
    out: list[tuple[int, int]] = []
    # W(1,k)=k
    out.extend([(1, n)])               # one witness
    # W(r,2)=r+1  -> r = n-1 (if n>=2)
    if n >= 2:
        out.extend([(n-1, 2)])
    return out


@classifier(
    label="Van der Waerden number (exact)",
    description="Checks if n equals a proved non-trivial Van der Waerden number W(r,k); trivial cases optional.",
    oeis=None,
    category=CATEGORY,
)
def is_vdw_number_exact(n: int, *, include_trivial: bool = False) -> tuple[bool, str | None]:
    if n <= 0:
        return False, None

    pairs = list(_VDW_VALUE_TO_RK.get(n, []))

    # Optionally add a tiny sample of trivial witnesses
    pairs.extend(_vdw_trivial_witnesses(n, include_trivial=include_trivial))

    if not pairs:
        return False, None

    # Format details
    nontriv = [p for p in pairs if p in _VDW_EXACT]
    triv = [p for p in pairs if p not in _VDW_EXACT]

    msg_parts = []
    if nontriv:
        msg_parts.append(_fmt_pairs_vdw(nontriv))
    if triv:
        msg_parts.append(_fmt_pairs_vdw(triv) + " (trivial)")

    return True, f"{n} = " + "; ".join(msg_parts) + "."


def _partition_numbers_up_to(limit_val: int, max_n: int = 5000) -> list[int]:
    """
    Generate p(0)..p(n) until p(n) exceeds limit_val or n reaches max_n.
    Uses Euler's pentagonal recurrence.
    """
    p = [1]  # p(0)=1
    n = 1
    while True:
        total = 0
        k = 1
        while True:
            pent1 = k * (3 * k - 1) // 2
            pent2 = k * (3 * k + 1) // 2
            if pent1 > n and pent2 > n:
                break
            sign = -1 if (k % 2 == 0) else 1
            if pent1 <= n:
                total += sign * p[n - pent1]
            if pent2 <= n:
                total += sign * p[n - pent2]
            k += 1
        p.append(total)
        if total > limit_val or n >= max_n:
            return p
        n += 1


@classifier(
    label="Partition number",
    description="Checks if n equals p(k): number of integer partitions of k.",
    oeis="A000041",
    category=CATEGORY,
)
def is_partition_number(m: int, max_n: int = 5000) -> tuple[bool, str | None]:
    """
    Returns (True, details) if m == p(k) for some k≥0.
    """
    if m < 1:
        return False, None

    p = [1]
    n = 1
    while True:
        # Euler recurrence
        total = 0
        k = 1
        while True:
            pent1 = k * (3 * k - 1) // 2
            pent2 = k * (3 * k + 1) // 2
            if pent1 > n and pent2 > n:
                break
            sign = -1 if (k % 2 == 0) else 1
            if pent1 <= n:
                total += sign * p[n - pent1]
            if pent2 <= n:
                total += sign * p[n - pent2]
            k += 1
        p.append(total)

        if total == m:
            return True, f"{m} = p({n})."
        if total > m or n >= max_n:
            return False, None
        n += 1


@classifier(
    label="Fubini number",
    description="Ordered Bell number: number of ordered set partitions of an n-element set (F(n)).",
    oeis="A000670",
    category=CATEGORY,
)
def is_fubini_number(v: int) -> tuple[bool, str | None]:
    """
    Checks whether v equals F(n) = sum_{k=0}^n k! * {n \brace k} for some n>=0.
    Strategy: build Stirling-2 rows with memoization; compute F(n) until F(n) >= v
              or we run out of the (small) time budget (lifted if FAST_MODE=False).
    """
    if v <= 0:
        return False, None

    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    budget_s = float(CFG("COMBINATORIAL.TIME_BUDGET_S", 0.5))
    max_n = int(CFG("CLASSIFIER.FUBINI.MAX_INDEX", 5000))  # index cap for compute loop

    t0 = time.perf_counter()

    def ok() -> bool:
        if not fast_mode:
            return True
        return (time.perf_counter() - t0) < budget_s

    @functools.cache
    def _fact(n: int) -> int:
        if n <= 1:
            return 1
        return n * _fact(n - 1)

    @functools.cache
    def _stirling2_row(n: int) -> tuple[int, ...]:
        # returns ({n brace 0}, {n brace 1}, ..., {n brace n})
        if n == 0:
            return (1,)
        prev = _stirling2_row(n - 1)
        cur = [0] * (n + 1)
        cur[n] = 1
        for k in range(1, n):
            cur[k] = k * prev[k] + prev[k - 1]
        return tuple(cur)

    # F(0) = 1 quick check
    if v == 1:
        return True, f"{abbr_int_fast(v)} is Fubini (ordered Bell) number F(0) (computed)."

    # Compute F(n) increasing in n until we pass v or hit guards
    for n in range(1, max_n + 1):
        if not ok():
            break
        row = _stirling2_row(n)
        Fn = 0
        # accumulate k! * S(n,k); small optimization: skip k=0 (term is 0 for n>0)
        for k in range(1, n + 1):
            Fn += _fact(k) * row[k]
        if Fn == v:
            return True, f"{abbr_int_fast(v)} is Fubini (ordered Bell) number F({n})."
        if Fn > v:
            # Strictly increasing for n>=1, so no need to continue
            break

    return False, None


@classifier(
    label="Lah number",
    description="Equals L(n,k) for some n≥1,k≥1: partitions of n elements into k nonempty ordered lists.",
    oeis="A105278",
    category=CATEGORY,
)
def is_lah_number(v: int) -> tuple[bool, str | None]:
    """
    Pure-compute: finds all (n,k) with L(n,k) = v.
    Fast row generation via L(n,k+1) = L(n,k) * (n-k) / (k*(k+1)).
    """
    if v <= 0:
        return False, None

    rt = _rt_current()
    fast_mode = bool(rt.fast_mode)
    budget_s = float(CFG("COMBINATORIAL.TIME_BUDGET_S", 0.5))
    max_n = int(CFG("CLASSIFIER.LAH.MAX_INDEX", 10000))

    t0 = time.perf_counter()

    def ok() -> bool:
        if not fast_mode:
            return True
        return (time.perf_counter() - t0) < budget_s

    hits: list[tuple[int, int]] = []

    # Scan n=1..max_n; row via fast recurrence starting from L(n,1)=n!
    # L(n,1) == n! grows very fast
    for n in range(1, max_n + 1):
        if not ok():
            break

        Ln1 = fact(n)            # L(n,1) = n!
        if Ln1 == v:
            hits.append((n, 1))
            # don't return; continue to find other k (e.g., 120 = L(5,1) and L(5,3))

        # Generate k=2..n using the multiplicative update:
        # L(n,k+1) = L(n,k) * (n-k) / (k*(k+1))
        Lnk = Ln1
        for k in range(1, n):
            # compute next
            num = n - k
            den = k * (k + 1)
            # exact integer step; do multiplication before division to stay integral
            Lnk = (Lnk * num) // den
            if Lnk == v:
                hits.append((n, k + 1))

        # No aggressive early-break here; another n might still yield v for a different k.

    if hits:
        # Prefer non-trivial k first (k>1), then sort by (n,k); show up to 5
        hits.sort(key=lambda nk: (1 if nk[1] > 1 else 2, nk[0], nk[1]))
        shown = ", ".join([f"L({n},{k})" for (n, k) in hits[:5]])
        more = "" if len(hits) <= 5 else f" (+{len(hits)-5} more)"
        return True, f"{abbr_int_fast(v)} = {shown}{more}."

    return False, None
