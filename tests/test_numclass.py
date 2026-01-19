# tests/test_numclass.py
"""
Tests for numclass (limit-aware; selective execution of needed classifiers only).

Run: pytest -v
"""

from __future__ import annotations

import inspect
import re

import pytest

from numclass.registry import discover
from numclass.utility import build_ctx
from numclass.workspace import ensure_workspace_seeded, workspace_dir

# ---------- helpers -----------------------------------------------------------


def _tokenize(lbl: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", lbl).upper().strip("_")


def _exceeds_limit(index, label: str, n: int) -> bool:
    lim = getattr(index, "limits", {}).get(label)
    return lim is not None and abs(int(n)) > int(lim)


def _build_needed_atomic_labels(index, targets: set[str]) -> set[str]:
    """
    Given human-readable target labels (which may include intersections),
    return the set of *atomic* labels that must be executed to potentially
    satisfy those targets.
    """
    funcs = index.funcs
    inters = getattr(index, "intersections", []) or []
    tokmap = getattr(index, "label_to_token", None) or {lbl: _tokenize(lbl) for lbl in funcs}
    rev = {}
    for lbl, tok in tokmap.items():
        rev.setdefault(tok, lbl)

    needed: set[str] = set()
    for t in targets:
        if t in funcs:
            needed.add(t)
            continue
        for rule in inters:
            if rule.label == t:
                for req_tok in rule.requires:
                    base_lbl = rev.get(req_tok)
                    if base_lbl and base_lbl in funcs:
                        needed.add(base_lbl)
                break

    if not needed:
        for t in targets:
            if t in funcs:
                needed.add(t)
    return needed


def _call_classifier(fn, n: int, ctx):
    """Call classifier with (n) or (n, ctx=...) depending on signature; return (ok, detail|None)."""
    sig = inspect.signature(fn)
    kwargs = {}
    if "ctx" in sig.parameters:
        kwargs["ctx"] = ctx
    if "allow_slow" in sig.parameters:
        kwargs["allow_slow"] = False
    elif "allow_slow_calculations" in sig.parameters:
        kwargs["allow_slow_calculations"] = False

    res = fn(n, **kwargs)
    if isinstance(res, tuple):
        return bool(res[0]), (res[1] if len(res) > 1 else None)
    return bool(res), None


def run_needed_only(n: int, index, *, targets: set[str]) -> set[str]:
    """
    Execute only the atomic classifiers needed for `targets`, enforcing per-label limits.
    Then synthesize intersections based on the resulting tokens.
    """
    funcs = index.funcs
    tokmap = getattr(index, "label_to_token", None) or {lbl: _tokenize(lbl) for lbl in funcs}

    needed = _build_needed_atomic_labels(index, targets)
    labels_ok: set[str] = set()

    # Lazily build ctx only if at least one needed classifier requests it
    ctx = None
    ctx_built = False

    def get_ctx():
        nonlocal ctx, ctx_built
        if not ctx_built:
            ctx = build_ctx(abs(int(n)))
            ctx_built = True
        return ctx

    for label in needed:
        fn = funcs.get(label)
        if not fn:
            continue
        # Enforce per-label limit: skip execution for out-of-range n
        if _exceeds_limit(index, label, n):
            continue
        try:
            # Only construct ctx if this classifier needs it
            sig = inspect.signature(fn)
            if "ctx" in sig.parameters:
                ok, _ = _call_classifier(fn, n, get_ctx())
            else:
                ok, _ = _call_classifier(fn, n, None)
        except Exception:
            continue

        if ok:
            labels_ok.add(label)

    # Intersections
    present_tokens = {tokmap.get(lbl, _tokenize(lbl)) for lbl in labels_ok}
    for rule in (getattr(index, "intersections", []) or []):
        if all(req in present_tokens for req in rule.requires):
            labels_ok.add(rule.label)

    return labels_ok


# ---------- session bootstrap -------------------------------------------------


@pytest.fixture(scope="session")
def index():
    """Seed workspace (if needed) and discover classifiers + intersections once."""
    ensure_workspace_seeded()
    return discover(workspace_dir())


# --- 234 tests for 198 atomic classifiers + 26 intersections ---

TEST_CASES = [
    # Arithmetic and Divisor-based (49)
    (945,   ["Abundant number"]),
    (72,    ["Achilles number"]),
    (16,    ["Almost perfect number"]),
    (220,   ["Amicable number"]),
    (119,   ["Aspiring number"]),
    (75,    ["Betrothed number"]),
    (120,   ["Colossally abundant number"]),
    (10,    ["Cube-free number"]),
    (216,   ["Cube-full number"]),
    (76,    ["Deficient number"]),
    (198585576189, ["Descartes number"]),
    (30,    ["Friendly number"]),
    (858,   ["Giuga number"]),
    (24,    ["Hemi perfect"]),
    (154345556085770649600, ["Hexaperfect number"]),
    (6720,  ["Highly abundant number"]),
    (36,    ["Highly composite number"]),
    (325,   ["k-Hyperperfect number"]),
    (81,    ["k-full number (k=4)"]),
    (2047,  ["Mersenne number"]),
    (24,    ["Near-perfect number"]),
    (496,   ["Ore (harmonic divisor) number"]),
    (14182439040, ["Pentaperfect number"]),
    (125,   ["Perfect cube"]),
    (8128,  ["Perfect number"]),
    (8,     ["Perfect power"]),
    (100,   ["Perfect square"]),
    (27,    ["Perfect totient number"]),
    (10,    ["Polite number"]),
    (108,   ["Powerful number"]),
    (18,    ["Practical number"]),
    (1806,  ["Primary pseudoperfect number"]),
    (4030,  ["Primitive weird number"]),
    (24,    ["q-aspiring number"]),
    (1215571544, ["q-sociable number"]),
    (92,    ["q-socially aspiring number"]),
    (30240, ["Quadriperfect number"]),
    (562,   ["Quasi-sociable number"]),
    (20,    ["Semiperfect number"]),
    (12496, ["Sociable number"]),
    (562,   ["Socially aspiring number"]),
    (7,     ["Solitary number"]),
    (30,    ["Sphenic number"]),
    (10,    ["Squarefree number"]),
    (12,    ["Sublime number"]),
    (60,    ["Superabundant number"]),
    (4096,  ["Superperfect number"]),
    (120,   ["Triperfect number"]),
    (87360, ["Unitary perfect number"]),
    (2,     ["Untouchable number"]),
    (836,   ["Weird number"]),
    (30,    ["Zumkeller number"]),

    # Combinatorial and Geometric (12)
    (4140,  ["Bell number"]),
    (26,    ["Cake number"]),
    (42,    ["Catalan number"]),
    (13,    ["Fubini number"]),
    (141120, ["Lah number"]),
    (323,   ["Motzkin number"]),
    (7,     ["Partition number"]),
    (6,     ["Ramsey number (2-colour)"]),
    (5,     ["Schur number"]),
    (11,    ["Stirling number (1st kind)"]),
    (7,     ["Stirling number (2nd kind)"]),
    (9,     ["Van der Waerden number (exact)"]),

    # Conjectures and Equation-based (8)
    (11,    ["Egyptian m/n (profile)"]),
    (31,    ["Erdős–Straus"]),
    (34,    ["Goldbach conjecture"]),
    (35,    ["Legendre prime interval"]),
    (37,    ["Lemoine's (Levy's)"]),
    (9,     ["Sierpiński"]),
    (91,    ["Weak Goldbach (ternary)"]),
    (6,     ["Znám chain"]),

    # Digit-Based (22)
    (376,   ["Automorphic number"]),
    (26,    ["Brazilian number"]),
    (135,   ["Disarium number"]),
    (512,   ["Dudeney number"]),
    (3,     ["Evil number"]),
    (40585, ["Factorion"]),
    (11235813,  ["Fibonacci-concatenation"]),
    (765,   ["Grafting number"]),
    (8281,  ["Happy number"]),
    (1012,  ["Harshad number"]),
    (58,    ["Hoax number"]),
    (703,   ["Kaprekar number"]),
    (196,   ["Lychrel number"]),
    (21,    ["Moran number"]),
    (92727, ["Narcissistic number"]),
    (7,     ["Odious number"]),
    (23432, ["Palindrome"]),
    (111,   ["Palindromic Harshad number"]),
    (7777,  ["Repdigit"]),
    (31,    ["Self number"]),
    (1210,  ["Self-descriptive number"]),
    (22,    ["Smith number"]),
    (2380,  ["Sum of 2 palindromes"]),
    (5276,  ["Sum of 3 palindromes"]),

    # Diophantine representations (4)
    (28,    ["Sum of 2 cubes"]),
    (25,    ["Sum of 2 squares"]),
    (3,     ["Sum of 3 cubes"]),
    (3,     ["Sum of 3 squares"]),

    # Dynamical Sequences (9)
    (35,    ["Collatz (3n+1) sequence"]),
    (10,    ["Generalized Collatz (5n+1)"]),
    (10,    ["Generalized Collatz (7n+1)"]),
    (10,    ["Generalized Collatz (an+b)"]),
    (36,    ["Ducci (4-digit) sequence"]),
    (37,    ["Fibonacci mod n: Pisano period"]),
    (19,    ["Happy number sequence"]),
    (234,   ["Kaprekar routine"]),
    (390,   ["Reverse-and-add sequence"]),

    # Fun numbers (1)
    (1337,  ["Fun number"]),

    # Mathematical Curiosities (20)
    (1235813, ["Additive sequence"]),
    (1000101, ["Binary-interpretable number"]),
    (1911,  ["Boring number"]),
    (142857, ["Cyclic permutation number"]),
    (12012, ["Cyclops number"]),
    (1089,  ["Digit-Reversal constant"]),
    (32,    ["Eban number"]),
    (1,     ["Harshad in all bases"]),
    (495,   ["Kaprekar Constant (3 digit)"]),
    (6174,  ["Kaprekar Constant (4 digit)"]),
    (2520,  ["LCM-prefix number"]),
    (163,   ["Lucky number of Euler"]),
    (3435,  ["Munchausen number"]),
    (76617, ["Octal-interpretable number"]),
    (1023456789, ["Pandigital number (0-9)"]),
    (123456789, ["Pandigital number (1-9)"]),
    (360,   ["Polydivisible number"]),
    (2357,  ["Smarandache Wellin number"]),
    (88,    ["Strobogrammatic number"]),
    (1260,  ["Vampire number"]),

    # Named Sequences (16)
    (107,   ["Busy Beaver number"]),
    (16127, ["Carol number"]),
    (385,   ["Cullen number"]),
    (366,   ["Erdős-Woods number"]),
    (720,   ["Factorial number"]),
    (144,   ["Fibonacci number"]),
    (4,     ["Hamming number"]),
    (197,   ["Keith number"]),
    (123,   ["Lucas number"]),
    (9,     ["Lucky number"]),
    (16,    ["Padovan number"]),
    (70,    ["Pell number"]),
    (1729,  ["Taxicab number"]),
    (24,    ["Tribonacci number"]),
    (106,   ["Ulam number"]),
    (159,   ["Woodall number"]),

    # Polygonal and Figurate Numbers (16)
    (91,    ["Centered hexagonal number"]),
    (13,    ["Centered square number"]),
    (85,    ["Centered triangular number"]),
    (45,    ["Harshad triangular number"]),
    (66,    ["Hexagonal number"]),
    (121,   ["Lehmer number"]),
    (171,   ["Palindromic triangular number"]),
    (70,    ["Pentagonal number"]),
    (90,    ["Pronic number"]),
    (1111,  ["Repunit"]),
    (37,    ["Star number"]),
    (1001452269, ["Tetrahedral number"]),
    (91,    ["Triangular number"]),

    # Prime & Prime-related Numbers (37)
    (113,   ["Absolute prime"]),
    (53,    ["Balanced prime"]),
    (877,   ["Bell prime"]),
    (313,   ["Both-truncatable prime"]),
    (8191,  ["Brazilian prime"]),
    (5,     ["Catalan prime"]),
    (89,    ["Chen prime"]),
    (197,   ["Circular prime"]),
    (967,   ["Cousin prime"]),
    (13,    ["emirp"]),
    (5039,  ["Factorial prime"]),
    (65537, ["Fermat prime"]),
    (19,    ["Gaussian prime"]),
    (967,   ["Good prime"]),
    (17,    ["Good prime (local)"]),
    (89,    ["Isolated prime"]),
    (197,   ["Keith prime"]),
    (13,    ["Left-truncatable prime"]),
    (97,    ["Pierpont prime"]),
    (5,     ["Prime triplet member"]),
    (30,    ["Primorial number"]),
    (30029, ["Primorial prime"]),
    (41,    ["Proth prime"]),
    (127,   ["Ramanujan prime"]),
    (59,    ["Right-truncatable prime"]),
    (47,    ["Safe prime"]),
    (33,    ["Semiprime"]),
    (461,   ["Sexy prime"]),
    (23,    ["Sophie Germain prime"]),
    (37,    ["Strong prime"]),
    (59,    ["Super prime"]),
    (67,    ["Thin prime"]),
    (17,    ["Twin prime"]),
    (1093,  ["Wieferich prime"]),
    (563,   ["Wilson prime"]),
    (13,    ["Weak prime"]),

    # Intersection primes (23)
    (5,     ["Automorphic prime"]),
    (877,   ["Bell prime"]),
    (393050634124102232869567034555427371542904833,   ["Cullen prime"]),
    (12049, ["Cyclops prime"]),
    (89,    ["Disarium prime"]),
    (2,     ["Factorion prime"]),
    (233,   ["Fibonacci prime"]),
    (383,   ["Happy palindromic prime"]),
    (79,    ["Happy prime"]),
    (2,     ["Harshad prime"]),
    (199,   ["Lucas prime"]),
    (223,   ["Lucky prime"]),
    (8191,  ["Mersenne prime"]),
    (15511, ["Motzkin prime"]),
    (7,     ["Narcissistic prime"]),
    (37,    ["Padovan prime"]),
    (787,   ["Palindromic prime"]),
    (29,    ["Pell prime"]),
    (11,    ["Repunit prime"]),
    (23,    ["Smarandache-Wellin prime"]),
    (3,     ["Triangular prime"]),
    (47,    ["Ulam prime"]),
    (23,    ["Woodall prime"]),

    # Pseudoprimes and Cryptographic Numbers (8) 2 classifiers are tested 6 times for each base.
    (21,    ["Blum integer"]),
    (561,   ["Carmichael number"]),
    (511,   ["Cunningham number"]),
    (561,   ["Euler-Jacobi pseudoprime"]),  # base 2
    (121,   ["Euler-Jacobi pseudoprime"]),  # base 3
    (781,   ["Euler-Jacobi pseudoprime"]),  # base 5
    (325,   ["Euler-Jacobi pseudoprime"]),  # base 7
    (133,   ["Euler-Jacobi pseudoprime"]),  # base 11
    (85,    ["Euler-Jacobi pseudoprime"]),  # base 13
    (341,   ["Fermat pseudoprime"]),        # base 2
    (121,   ["Fermat pseudoprime"]),        # base 3
    (781,   ["Fermat pseudoprime"]),        # base 5
    (10585, ["Fermat pseudoprime"]),        # base 7
    (2465,  ["Fermat pseudoprime"]),        # base 11
    (244,   ["Fermat pseudoprime"]),        # base 13
    (399,   ["Lucas-Carmichael number"]),
    (
        1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139,
        ["RSA challenge number"]
    ),
    (3215031751, ["Strong pseudoprime"]),   # in base 2, 3, 5, 7
]

TEST_IDS = [f"{','.join(labels)}_{n}" for n, labels in TEST_CASES]


@pytest.mark.parametrize("n,expected_labels", TEST_CASES, ids=TEST_IDS)
def test_classification_contains_expected_labels(index, n, expected_labels):
    """
    Run *only* the classifiers needed for the expected labels, enforcing per-label limits.
    If an expected label’s limit is below n, skip that particular expectation.
    """
    # Which expectations are in scope at this n?
    in_scope = [lbl for lbl in expected_labels if not _exceeds_limit(index, lbl, n)]
    if not in_scope:
        pytest.skip(f"All expected labels exceed limits for n={n}: {expected_labels}")

    got = run_needed_only(n, index, targets=set(in_scope))

    for label in in_scope:
        assert label in got, f"{n}: expected '{label}' in {sorted(got)}"


def test_discovery_counts_rough_sanity(index):
    """Guardrail to ensure discovery found a healthy number of classifiers."""
    atomic = len(index.funcs)
    inters = len(getattr(index, "intersections", []) or [])
    assert atomic >= 196, f"too few atomic classifiers discovered: {atomic}"
    assert inters >= 26, "too few intersections discovered — check intersections.toml and loader"
