# -----------------------------------------------------------------------------
#  digit_based.py
#  Digit based test functions
# -----------------------------------------------------------------------------

from __future__ import annotations

import textwrap
from decimal import Decimal, getcontext
from math import log2

from colorama import Fore, Style
from sympy import integer_nthroot, isprime

from numclass.context import NumCtx
from numclass.fmt import abbr_int_fast, format_factorization
from numclass.registry import classifier
from numclass.runtime import CFG
from numclass.utility import build_ctx, dec_digits, digit_sum, get_terminal_width

CATEGORY = "Digit-based"


# --- Helpers ---


def _base_witness(n: int, b: int, k: int, d: int) -> str:
    return (f"In base {b}, n is a repdigit of length {k} with digit {d}: "
            f"{d}×R_{k}({b}) where R_{k}({b})=( {b}^{k}−1 )/({b}−1)")


def _check_grafting_decimal_with_peek(
    n: int,
    order: int,
    side: str = "either",
    allow_straddle: bool = True,
    guard: int = 8,
) -> tuple[bool, str | None, str]:
    """
    Base-10 grafting check for a single order.
    Returns (ok, detail, frac_head3) where frac_head3 are the first 3 frac digits.
    """
    if n <= 0 or order < 2:
        return False, None, ""

    n_str = str(n)
    ndigs = len(n_str)

    # precision budget: generous, scales with ndigs
    prec = max(50, ndigs * 2 + 30)
    root = _nth_root_decimal(n, order, prec)
    i_part = int(root)
    frac = root - i_part

    i_str = str(i_part)
    f_str = _first_frac_digits_dec(frac, ndigs + guard)

    left_ok = i_str.endswith(n_str)
    right_ok = f_str.startswith(n_str)

    straddle_ok = False
    if allow_straddle and not (left_ok or right_ok):
        # split n_str into prefix (left of radix) and suffix (right)
        for k in range(1, ndigs):
            if i_str.endswith(n_str[:k]) and f_str.startswith(n_str[k:]):
                straddle_ok = True
                break

    match_ok = (
        (side == "left" and left_ok) or
        (side == "right" and right_ok) or
        (side == "either" and (left_ok or right_ok or straddle_ok))
    )

    if not match_ok:
        return False, None, f_str[:3]

    mode = "left" if left_ok else ("right" if right_ok else "straddle")
    preview = _colored_preview(i_str, f_str, n_str, mode, pad=3)
    detail = f"Order p={order}, match={mode}; n={n}, root≈{preview}"
    return True, detail, f_str[:3]


def _first_frac_digits_dec(frac: Decimal, k: int) -> str:
    """
    Return first k decimal digits of the fractional part (base 10),
    padding with zeros if the expansion terminates.
    """
    digs = []
    r = frac
    for _ in range(k):
        r *= 10
        d = int(r)            # floor
        digs.append(chr(ord('0') + d))
        r -= d
        if r == 0:
            if len(digs) < k:
                digs.extend('0' * (k - len(digs)))
            break
    return ''.join(digs)


def _colored_preview(i_str: str, f_str: str, n_str: str, mode: str, pad: int = 3) -> str:
    """
    Return a short, colorized preview like '…31{Y}622{R}.776…' with the digits of n_str
    highlighted in yellow at the radix point. `pad` controls context digits beside the match.
    """
    Y = Fore.YELLOW + Style.BRIGHT
    R = Style.RESET_ALL + Fore.GREEN + Style.BRIGHT
    nd = len(n_str)

    if mode == "left":
        # integer ends with n_str
        pre_int = i_str[:-nd]
        int_tail_col = f"{Y}{n_str}{R}"
        frac_head = f_str[: pad + nd]
        return f"{pre_int[-pad:]}{int_tail_col}.{frac_head}"

    if mode == "right":
        # fraction starts with n_str
        int_tail = i_str[-pad:]
        frac_col = f"{Y}{n_str}{R}" + f_str[nd: nd + pad]
        return f"{int_tail}.{frac_col}"

    # straddle: i_str endswith n_str[:k] and f_str startswith n_str[k:]
    for k in range(1, nd):
        left = n_str[:k]
        right = n_str[k:]
        if i_str.endswith(left) and f_str.startswith(right):
            pre_int = i_str[:-k]
            int_tail_col = f"{Y}{left}{R}"
            frac_col = f"{Y}{right}{R}" + f_str[len(right): len(right) + pad]
            return f"{pre_int[-pad:]}{int_tail_col}.{frac_col}"

    # Fallback (shouldn’t happen if mode was derived correctly)
    return f"{i_str[-pad:]}.{f_str[:pad]}"


def _int_nth_root(n: int, k: int) -> int:
    """Floor of n**(1/k) for integers (k>=1)."""
    if n <= 1:
        return n
    # float seed then correct
    x = int(n ** (1.0 / k))
    while pow(x + 1, k) <= n:
        x += 1
    while pow(x, k) > n:
        x -= 1
    return x


def _nth_root_decimal(n: int, p: int, prec: int) -> Decimal:
    """
    Compute n**(1/p) as a Decimal with 'prec' precision.
    Uses Decimal power for a seed, then refines by Newton to reduce bias.
    """
    ctx = getcontext()
    old_prec = ctx.prec
    try:
        ctx.prec = prec
        if n == 0:
            return Decimal(0)
        x = Decimal(n) ** (Decimal(1) / p)  # seed
        # Newton: x_{k+1} = ((p-1)*x + n / x^{p-1}) / p
        nD = Decimal(n)
        pD = Decimal(p)
        for _ in range(max(10, p + 5)):
            x_pow = x ** (p - 1)
            if x_pow == 0:
                break
            x = ((pD - 1) * x + nD / x_pow) / pD
        return +x  # quantize to current context
    finally:
        ctx.prec = old_prec


def _repunit_value(b: int, k: int) -> int:
    """R_k(b) = 1 + b + ... + b^{k-1} = (b^k - 1)//(b - 1)."""
    return (pow(b, k) - 1) // (b - 1)


def _try_k_eq_2_via_divisors(n: int, ctx: NumCtx, max_divisors: int = 5000) -> str | None:
    """
    For k=2, n = d*(b+1) with 1<=d<=b-1 and 2<=b<=n-2.
    Let r=b+1 divide n; then d=n/r and we require d <= r-2 and r<=n-1.
    We only attempt this if the divisor set is reasonably small.
    """
    # quick bound on tau = number of divisors
    if ctx.tau > max_divisors:
        return None
    # enumerate all divisors from the factorization
    divs = [1]
    for p, e in ctx.fac.items():
        new = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new.append(d * pe)
            pe *= p
        divs = new
        if len(divs) > max_divisors * 4:  # keep work bounded
            return None
    for r in sorted(divs):
        if r <= 2:
            continue
        if r >= n:        # would imply b = r-1 >= n-1 (disallowed)
            continue
        if n % r != 0:
            continue
        d = n // r
        b = r - 1
        if 1 <= d <= b - 1 and 2 <= b <= n - 2:
            return _base_witness(n, b, 2, d)
    return None


# --- Classifiers ---

@classifier(
    label="Automorphic number",
    description="a number whose square ends with the number itself (n >= 0)",
    oeis="A003226",
    category=CATEGORY
)
def is_automorphic_number(n: int) -> tuple[bool, str]:
    """
    Check if n is an automorphic number.
    """
    if n < 0:
        return False, None
    sq = n * n
    s, s_sq = str(n), str(sq)
    if s_sq.endswith(s):
        return True, f"{n}² = {sq} ends with {n}."
    return False, None


@classifier(
    label="Brazilian number",
    description="Representable as a repdigit in some base b with 2 ≤ b ≤ n−2.",
    oeis="A125134",
    category=CATEGORY,
    limit=6_000_000_000_000
)
def is_brazilian_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n <= 6:
        return False, None  # smallest Brazilian numbers start above this
    ctx = ctx or build_ctx(abs(n))
    N = ctx.n

    # --- fast search for k >= 3
    k_max = int(log2(N + 1))  # because R_k(2) = 2^k - 1 <= N
    for k in range(3, k_max + 1):
        b_max = min(N - 2, _int_nth_root(N, k - 1))  # r_k(b) >= b^{k-1} <= N
        if b_max < 2:
            continue
        b = 2
        while b <= b_max:
            r = _repunit_value(b, k)
            if r > N:
                break
            if N % r == 0:
                d = N // r
                if 1 <= d <= b - 1:
                    return True, _base_witness(N, b, k, d)
            b += 1

    # --- optional k=2 check (only when divisors are few)
    k2_detail = _try_k_eq_2_via_divisors(N, ctx, max_divisors=5000)
    if k2_detail:
        return True, k2_detail

    return False, None


@classifier(
    label="Disarium number",
    description="Sum of digits^position equals the given number.",
    oeis="A032799",
    category=CATEGORY
)
def is_disarium_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a Disarium number, return (True, details) if so, otherwise False.
    Formats the details for readability with wrapping and alignment.
    """
    KNOWN = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 89, 135, 175, 518, 598,
             1306, 1676, 2427, 2646798, 12157692622039623539}
    if n not in KNOWN:
        return False, None

    digits = list(map(int, str(abs(n))))
    powers = [f"{d}^{i+1}" for i, d in enumerate(digits)]
    values = [str(d**(i+1)) for i, d in enumerate(digits)]

    # Format: n = d1^1 + d2^2 ... = val1 + val2 ...
    number_part = f"{n} = "
    powers_part = " + ".join(powers)
    values_part = " + ".join(values)
    result_str = number_part + powers_part + " = " + values_part

    # Wrap with indent (subsequent_indent = 13 spaces)
    width = get_terminal_width() - 13
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent="",          # details already indented by print_results
        subsequent_indent=" " * 13
    )
    detail = wrapper.fill(result_str)
    return True, detail


@classifier(
    label="Dudeney number",
    description="Perfect cube whose digits sum to its cube root.",
    oeis="A061209",
    category=CATEGORY
)
def is_dudeney_number(n: int) -> tuple[bool, str | None]:
    """
    Returns (True, details) if n is a Dudeney number.
    A Dudeney number is a perfect cube such that the sum of its digits equals the cube root.
    Example: 512 → 5 + 1 + 2 = 8, and 8^3 = 512.
    """
    if n < 1:
        return False, None

    # exact integer cube-root check (no floats)
    r, exact = integer_nthroot(n, 3)
    if not exact:
        return False, None

    s_n = str(n)
    digit_sum = sum(int(c) for c in s_n)
    if digit_sum != r:
        return False, None

    # Formatting details (unchanged style)
    width = get_terminal_width()
    digits_expr = " + ".join(s_n)
    eqn = f"{n} = {r}^3, and {r} = {digits_expr}"
    wrapper = textwrap.TextWrapper(width=width, initial_indent="", subsequent_indent=" " * 13)
    details = wrapper.fill(eqn)
    return True, details


@classifier(
    label="Evil number",
    description="Has an even number of 1's in binary representation for |n|.",
    oeis="A001969",
    category=CATEGORY,
    limit=2**100000-1
)
def is_evil_number(n) -> tuple[bool, str]:
    """
    Returns (True/False, details) for whether n is an evil number.
    An evil number has an even number of 1's in its binary representation.
    Also reports the total number of bits.
    """
    if n < 0:
        return False, None

    bits = bin(n)[2:]  # binary string without the '0b' prefix
    num_ones = bits.count("1")
    total_bits = len(bits)
    is_evil = num_ones % 2 == 0
    line_length = get_terminal_width() - 22
    digits = line_length // 2
    details = (
        f"Binary: {abbr_int_fast(int(bits), digits, digits, line_length)}, number of 1's: {num_ones} (even), "
        f"total bits: {total_bits}"
    )
    if is_evil:
        return True, details
    return False, None


# Precompute 0!..9! once
_DIGIT_FACT = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]


@classifier(
    label="Factorion",
    description="Equals sum of factorials of digits.",
    oeis="A014080",
    category=CATEGORY,
)
def is_factorion(n: int) -> tuple[bool, str | None]:
    # Definition is for nonnegative integers
    if n < 0:
        return False, None

    digits = [int(ch) for ch in str(n)]
    s = sum(_DIGIT_FACT[d] for d in digits)

    if s == n:
        # Pretty detail: e.g. 40585 = 4!+0!+5!+8!+5! = 24+1+120+40320+120
        left = " + ".join(f"{d}!" for d in digits)
        right = " + ".join(str(_DIGIT_FACT[d]) for d in digits)
        detail = f"{n} = {left} = {right}"
        return True, detail

    return False, None


@classifier(
    label="Fibonacci-concatenation",
    description="Digits contain a concatenation of consecutive Fibonacci numbers (F0=0, F1=1).",
    category=CATEGORY,
    oeis="A019523",
    limit=10**40 - 1,
)
def is_fib_concat(n: int, mode: str = "prefix", min_terms: int = 5):
    if n < 0:
        return False, None

    s = str(n)
    L = len(s)

    # Build Fibonacci strings up to enough length
    fib: list[str] = ["0", "1"]  # F0, F1
    a, b = 0, 1

    # Grow Fibonacci terms until the largest term is comfortably longer than the input.
    while len(fib[-1]) <= L + 20:
        a, b = b, a + b
        fib.append(str(b))

    def try_prefix_from_k(k: int) -> list[str] | None:
        parts: list[str] = []
        cur = ""
        i = k
        while len(cur) < L and i < len(fib):
            term = fib[i]
            parts.append(term)
            cur += term
            i += 1
        return parts if cur == s else None

    def try_contains_from_pos(start: int, min_terms: int) -> tuple[int, int, int, int] | None:
        # For each k, try to grow a run that starts at s[start:]
        for k in range(0, len(fib) - 1):
            i = k
            idx = start
            terms = 0
            while i < len(fib) and idx < L:
                term = fib[i]
                if not s.startswith(term, idx):
                    break
                idx += len(term)
                i += 1
                terms += 1
            if terms >= min_terms:
                return (start, idx, terms, k)
        return None

    if mode == "prefix":
        for k in range(0, len(fib) - 1):
            parts = try_prefix_from_k(k)
            if parts:
                return True, f"digits = {'|'.join(parts)} (starts at F({k}))"
        return False, None

    if mode == "contains":
        for start in range(L):
            res = try_contains_from_pos(start, min_terms)
            if res:
                i0, i1, terms, k = res
                snippet = s[i0:i1]
                head, tail = snippet[:30], snippet[-30:]
                ell = "…" if len(snippet) > 60 else ""
                return True, f"contains run ({terms} terms, starts at F({k})) at [{i0}:{i1}]: {head}{ell}{tail}"
        return False, None

    return False, f"unknown mode '{mode}'"


@classifier(
    label="Grafting number",
    description="digits, represented in base 10, appear before or directly after the decimal point of its pth root.",
    oeis="A232087",
    category=CATEGORY,
)
def is_grafting_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Base-10 only. Iterates orders p starting at 2 with:
      - configurable max (GRAFTING.MAX_ORDER, default 6)
      - hard ceiling (GRAFTING.HARD_MAX_ORDER, default 12)
      - adaptive cap by number of digits
      - optional early stop when fractional part starts with many zeros
        (GRAFTING.EARLY_STOP_ZEROS, default 2; set 0 to disable)
      - side match control: GRAFTING.SIDE in {"left","right","either"} (default "either")
      - straddle toggle: GRAFTING.ALLOW_STRADDLE (default true)
    Returns on the first order that matches.
    """
    MAX_DIGITS = CFG("CLASSIFIER.GRAFTING.MAX_DIGITS", 200)
    if dec_digits(n) > MAX_DIGITS:
        return False, "Maximum number of digits exceeded."
    if n < 0:
        return False, None
    if n == 0:
        return True, "Trivial grafting: 0ⁿ = 0"
    if n == 1:
        return True, "Trivial grafting: 1ⁿ = 1"

    # ---- settings (with sensible defaults) ---------------------------------
    side = str(CFG("CLASSIFIER.GRAFTING.SIDE", "either")).lower()
    if side not in {"left", "right", "either"}:
        side = "either"
    allow_straddle = CFG("CLASSIFIER.GRAFTING.ALLOW_STRADDLE", True)

    default_max = int(CFG("CLASSIFIER.GRAFTING.MAX_ORDER", 6))
    hard_max = int(CFG("CLASSIFIER.GRAFTING.HARD_MAX_ORDER", 12))
    early_stop_zeros = int(CFG("CLASSIFIER.GRAFTING.EARLY_STOP_ZEROS", 2))
    guard = int(CFG("CLASSIFIER.GRAFTING.GUARD_EXTRA_DIGITS", 8))

    # ---- adaptive cap by digits --------------------------------------------
    d = len(str(n))
    if d <= 2:
        p_max_adapt = 10
    elif d == 3:
        p_max_adapt = 8
    else:
        p_max_adapt = default_max

    p_max = max(2, min(p_max_adapt, hard_max))

    # ---- iterate orders -----------------------------------------------------
    for p in range(2, p_max + 1):
        ok, detail, frac_head = _check_grafting_decimal_with_peek(
            n,
            order=p,
            side=side,
            allow_straddle=allow_straddle,
            guard=guard,
        )
        if ok:
            return True, detail

        # early stop: if n has ≥3 digits and fractional starts with enough zeros,
        # higher orders will push n^(1/p) even closer to 1, making a match implausible.
        if early_stop_zeros > 0 and d >= 3 and frac_head.startswith("0" * min(3, early_stop_zeros)):
            break

    return False, None


@classifier(
    label="Happy number",
    description="Summing squares of digits repeatedly reaches 1.",
    oeis="A007770",
    category=CATEGORY
)
def is_happy_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a happy number.
    A happy number is defined by the process: replace the number with the sum
    of the squares of its digits, and repeat. If this eventually reaches 1,
    the number is happy; otherwise, it is unhappy.
    Returns (True, details) if so, else False.
    """
    if n < 1:
        return False, None
    seen = set()
    seq = [n]
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
        seq.append(n)
    seq_str = " → ".join(abbr_int_fast(x) for x in seq)
    if n == 1:
        return True, f"{seq_str} (happy sequence reaches 1)"
    return False, f"{seq_str} (happy sequence cycles)"


@classifier(
    label="Harshad number",
    description="Base-10 Harshad (Niven) number: divisible by the sum of its decimal digits.",
    oeis="A005349",
    category=CATEGORY,
)
def is_harshad_number(n: int) -> tuple[bool, str | None]:
    """
    Harshad (Niven) number in base 10.
    A positive integer n is Harshad if n is divisible by the sum of its decimal digits.
    """
    if n < 1:
        return False, None

    digit_sum = sum(int(d) for d in str(n))  # n is already ≥ 1, so abs() not needed
    # digit_sum cannot be 0 here, but keep the guard for safety / future variants.
    if digit_sum == 0:
        return False, None

    if n % digit_sum == 0:
        q = n // digit_sum
        details = (f"{abbr_int_fast(n)} is divisible by the sum of its digits: {abbr_int_fast(n)} "
                   f"÷ {abbr_int_fast(digit_sum)} = {abbr_int_fast(q)}.")
        return True, details

    return False, None


@classifier(
    label="Hoax number",
    description=("Composite n with sum of decimal digits equal to the sum of "
                 "digits of its distinct prime factors (requires ≥2 distinct "
                 "primes)."),
    oeis="A019506",
    category=CATEGORY,
)
def is_hoax_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Hoax number: composite n with at least two distinct primes and
    sum of digits of n equals the sum of digits of its DISTINCT prime factors.
    Example: 22 -> S(22)=4, primes {2,11}, S(2)+S(11)=2+2=4.
    """
    if n == 0:
        return False, None
    m = abs(n)

    # build/reuse context (factorization etc.)
    ctx = ctx or build_ctx(m)
    fac = ctx.fac

    # Exclude primes and prime powers: must have ≥ 2 distinct primes
    if len(fac) < 2:
        return False, None

    s_n = digit_sum(m)
    s_primes = sum(digit_sum(p) for p in fac)

    if s_n == s_primes:
        fac_str = format_factorization(fac)  # e.g., "2 × 11"
        primes_breakdown = " + ".join(f"S({p})={digit_sum(p)}" for p in fac)
        details = f"S(n)={s_n}; {primes_breakdown}; n = {fac_str}"
        return True, details

    return False, None


@classifier(
    label="Kaprekar number",
    description="Square split sums back to the given number.",
    oeis="A006886",
    category=CATEGORY,
    limit=10**20-1
)
def is_kaprekar_number(n: int) -> tuple[bool, str]:
    """
    Returns (True/False, details) for whether n is a Kaprekar number.
    A Kaprekar number is a non-negative integer whose square can be split into
    two parts that sum to the original number.
    Example: 45^2 = 2025 → 20 + 25 = 45
    """
    if n < 1:
        return False, None

    if n == 1:
        return True, "1 is Kaprekar by definition."

    s = str(n * n)
    found = False
    steps = []

    for i in range(1, len(s)):
        left_str, right_str = s[:i], s[i:]
        left = int(left_str or "0")
        right = int(right_str or "0")
        check = f"{left_str or '0'} + {right_str or '0'} = {left} + {right} = {left + right}"
        if right > 0 and left + right == n:
            found = True
            steps.append(f"• {n}² = {s} → {left_str or '0'} | {right_str} → {check} ✓")

    if found:
        # Formatting
        details = "\n".join([(line) for line in steps])
        return True, details
    return False, None


@classifier(
    label="Lychrel number",
    description="Does not form a palindrome by reverse-and-add.",
    oeis="A023108",
    category=CATEGORY,
    limit=10**20-1
)
def is_lychrel_number(n: int, max_iter=1000) -> tuple[bool, str]:
    """
    Check if n is (likely) a Lychrel number.
    limited to 1000 iterations for speed,  The largest number of steps ever
    observed to reach a palindrome (for non-Lychrel candidates) is in the
    dozens or, at most, a few hundred.
    """
    if n < 10:
        return False, None
    t = n
    for _ in range(max_iter):
        t += int(str(t)[::-1])
        if str(t) == str(t)[::-1]:
            return False, None
    details = (f"Is a Lychrel number candidate: no palindrome found in {max_iter} iterations.")
    return True, details


@classifier(
    label="Moran number",
    description="Harshad (base 10) with n/s = prime, where s is the sum of decimal digits of n.",
    oeis="A001101",
    category=CATEGORY,
)
def is_moran_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Moran number: n is Harshad (base 10) and n / sum_digits(n) is prime.
    Examples: 21 (s=3, 21/3=7), 132 (s=6, 132/6=22 not prime) → not Moran.
    """
    m = abs(n)
    if m == 0:
        return False, None

    # build/reuse ctx (kept for consistency with invoke() that injects ctx)
    ctx = ctx or build_ctx(m)

    s = digit_sum(m)
    if s == 0 or m % s != 0:
        return False, None

    q = m // s  # must be prime for Moran
    if isprime(q):
        return True, f"Harshad with s(n)={s}; n/s = {q} is prime."
    return False, None


@classifier(
    label="Narcissistic number",
    description="Sum of digits^n where n is digit count (also called an Armstrong number).",
    oeis="A005188",
    category=CATEGORY,
    limit=10**39-1
)
def is_narcissistic_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a narcissistic number.
    A narcissistic (Armstrong) number is a number that is the sum of its own
    digits each raised to the power of the number of digits.
    Returns (True, details) if so, else False.
    """
    if n < 0:
        return False, None
    s = str(n)
    power = len(s)
    sum_of_powers = sum(int(d) ** power for d in s)
    if n == sum_of_powers:
        digits = " + ".join(f"{d}^{power}" for d in s)
        details = f"{n} = {digits} = {sum_of_powers}"
        return True, details
    return False, None


@classifier(
    label="Odious number",
    description="Has an odd number of 1's in binary representation for |n|.",
    oeis="A000069",
    category=CATEGORY,
    limit=2**100000-1
)
def is_odious_number(n: int) -> tuple[bool, str]:
    """
    Returns (True/False, details) for whether n is an odious number.
    An odious number has an odd number of 1's in its binary representation.
    Also reports the total number of bits.
    """
    if n < 0:
        return False, None
    bits = bin(n)[2:]
    num_ones = bits.count("1")
    total_bits = len(bits)
    is_odious = num_ones % 2 == 1
    line_length = get_terminal_width() - 22
    digits = line_length // 2
    details = (
        f"Binary: {abbr_int_fast(int(bits), digits, digits, line_length)}, number of 1's: {num_ones} (odd), "
        f"total bits: {total_bits}"
    )
    if is_odious:
        return True, details
    return False, None


@classifier(
    label="Palindrome",
    description="Reads the same forwards and backwards.",
    oeis="A002113",
    category=CATEGORY
)
def is_palindrome(n: int) -> tuple[bool, str]:
    """
    Check if n is a palindrome, and return details if so.
    A palindrome is a number that reads the same forwards and backwards.
    """
    if n < 10:
        return False, None
    s = str(abs(n))
    if s == s[::-1]:
        details = f"{n} reads the same forwards and backwards: {s}."
        return True, details
    return False, None


@classifier(
    label="Repdigit",
    description="A number consisting of repeated instances of the same digit.",
    oeis="A010785",
    category=CATEGORY
)
def is_repdigit(n: int) -> tuple[bool, str]:
    """
    Returns (True, details) if all digits are the same. One digit numbers are included by definition.
    """
    s = str(abs(n))
    if len(s) >= 1 and len(set(s)) == 1:
        return True, f"All digits are {s[0]}: {n} is a repdigit."
    return False, None


@classifier(
    label="Self number",
    description="Cannot be written as m + sum of digits of m for any m.",
    oeis="A003052",
    category=CATEGORY,
    limit=10**20-1
)
def is_self_number(n: int) -> tuple[bool, str]:
    """
    Returns (True, details) if n is a self number (Devlali, Colombian number), else False.
    A self number cannot be written as m + sum(digits of m) for any m < n.
    """
    if n < 10:
        return False, None
    for m in range(max(0, n - 9 * len(str(n))), n):
        if m + sum(map(int, str(m))) == n:
            return False, None
    return True, f"{n} cannot be written as m + sum(digits of m) for any m < {n}."


@classifier(
    label="Self-descriptive number",
    description="In base 10: digit i equals the number of occurrences of digit i in the whole number.",
    oeis="A108551",
    category=CATEGORY,
)
def is_self_descriptive_number(n: int) -> tuple[bool, str | None]:
    if n < 0:
        return False, None

    s = str(n)
    m = len(s)

    # Digits must be in 0..m-1 (otherwise they can’t describe counts for this length).
    digits = [ord(ch) - 48 for ch in s]
    if any(d < 0 or d >= m for d in digits):
        return False, None

    # Count occurrences of each digit 0..m-1 in one pass.
    counts = [0] * m
    for d in digits:
        counts[d] += 1

    # Condition: for each i, digit at position i equals count of digit i.
    # (Equivalently: digits[i] == counts[i] for all i.)
    for i in range(m):
        if digits[i] != counts[i]:
            return False, None

    # Details: show compact count vector + a short explanation.
    # Example for 1210: counts=[1,2,1,0]
    vec = ", ".join(f"{i}:{counts[i]}" for i in range(m))
    return True, f"{n} is self-descriptive (length {m}): counts [{vec}]"


@classifier(
    label="Smith number",
    description="Composite n whose digit sum equals the sum of digits of its prime factors (with multiplicity).",
    oeis="A006753",
    category=CATEGORY,
)
def is_smith_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Smith numbers are composite (primes excluded) and satisfy:
      sum_of_digits(n) == sum_of_digits(p1) + ... + sum_of_digits(pk)
    where p1..pk are the prime factors of n **with multiplicity**.
    """
    m = abs(n)
    if m <= 1:
        return False, None

    ctx = ctx or build_ctx(m)
    fac = ctx.fac  # dict {prime: exponent}

    if not fac:
        return False, None  # handles m in {0,1} defensively

    # Exclude primes; include prime powers (e.g., 4, 121 are valid Smith numbers)
    if len(fac) == 1 and next(iter(fac.values())) == 1:
        return False, None

    s_n = digit_sum(m)
    # sum digits of primes with multiplicity
    pf_digits_sum = sum(digit_sum(p) * e for p, e in fac.items())

    if s_n != pf_digits_sum:
        return False, None

    # Details: show S(n), the multiplicity summary, and factorization nicely
    fac_str = format_factorization(fac)           # like "3^3 × 5 × 7"
    terms = [f"S({p})={digit_sum(p)}×{e}" if e > 1 else f"S({p})={digit_sum(p)}"
             for p, e in fac.items()]
    details = f"S(n)={s_n}; " \
              f"{' + '.join(terms)} = {pf_digits_sum}; " \
              f"n = {fac_str}"
    return True, details


def _is_pal(x: int) -> bool:
    s = str(x)
    return s == s[::-1]


def _palindromes_up_to(n: int, *, min_digits: int) -> list[int]:
    """All base-10 palindromes in [0, n] with at least min_digits and not ending in 0 (unless the number is 0)."""
    out: list[int] = []
    # single digit
    for d in range(min(10, n + 1)):
        if d == 0:
            if min_digits <= 1:
                out.append(0)
        elif min_digits <= 1:
            out.append(d)
    if n < 10:
        return out

    max_len = len(str(n))
    for L in range(2, max_len + 1):
        half = (L + 1) // 2
        start = 10 ** (half - 1)
        end = 10 ** half
        for h in range(start, end):
            s = str(h)
            pal = int(s + (s[::-1] if L % 2 == 0 else s[-2::-1]))
            if pal > n:
                break
            # filter by digits and trailing-zero rule
            if len(str(pal)) >= min_digits and (pal % 10 != 0):
                out.append(pal)
    return out


@classifier(
    label="Sum of 2 palindromes",
    description="Can be written as the sum of two palindromic numbers in base 10.",
    oeis="A035137",
    category=CATEGORY,
    limit=220_000_000_000,
)
def is_sum_of_2_palindromes(n: int, max_solutions=None, min_palindrome_digits=1):
    if n < 0:
        return False, None

    # settings
    cfg_max = CFG("DIOPHANTINE.MAX_SOL_SUM_OF_2_PALINDROMES", None)
    max_solutions = cfg_max if cfg_max is not None else max_solutions

    cfg_min = CFG("DIOPHANTINE.MIN_PALINDROME_DIGITS", None)
    try:
        min_digits = int(cfg_min) if cfg_min is not None else int(min_palindrome_digits)
    except Exception:
        min_digits = 1
    min_digits = max(min_digits, 1)

    allow_zero = bool(CFG("DIOPHANTINE.ALLOW_ZERO_IN_DECOMP", True))

    # -----------------------------------------------------------------------
    #  FAST PATH: n = palindrome, so n = n + 0 ONLY IF allowed
    # -----------------------------------------------------------------------
    # Conditions for allowing the shortcut:
    #   - zero must be allowed
    #   - zero must have enough digits  → only when min_digits == 1
    #   - n itself must satisfy the minimum digit constraint
    #
    # This enforces:
    #   If MIN_PALINDROME_DIGITS > 1 → NEVER use n = n + 0
    # -----------------------------------------------------------------------

    if (
        allow_zero
        and min_digits == 1                      # zero allowed only for min_digits = 1
        and _is_pal(n)
        and (len(str(n)) >= min_digits)
        and (n % 10 != 0 or n == 0)
    ):
        return True, f"{n} = {n} + 0"

    # -----------------------------------------------------------------------
    #  SEARCH PATH: generate palindromes respecting min_digits and zero rules
    # -----------------------------------------------------------------------

    pals = _palindromes_up_to(n, min_digits=min_digits)
    pal_set = set(pals)

    # If zero is not allowed, remove it explicitly
    if not allow_zero and 0 in pal_set:
        pal_set.remove(0)
        pals = [p for p in pals if p != 0]

    results: list[tuple[int, int]] = []
    truncated = False

    for a in pals:
        b = n - a
        if b < a:
            break

        # if zero is banned, exclude b=0 as well
        if not allow_zero and b == 0:
            continue

        if b in pal_set:
            results.append((a, b))
            if max_solutions is not None and len(results) >= max_solutions:
                truncated = True
                break

    # -----------------------------------------------------------------------
    #  RESULTS
    # -----------------------------------------------------------------------

    if results:
        shown = results[:max_solutions] if max_solutions else results
        details = f"{n} = " + "; ".join(f"{a} + {b}" for a, b in shown)
        if truncated:
            details += f" (Limited to {max_solutions} solutions as specified in settings.)"
        return True, details

    if min_digits > 1:
        return False, (
            f"Minimum palindrome digit length is set to {min_digits} in settings. "
            "No decomposition found with palindromes of at least that many digits."
        )

    return False, None


@classifier(
    label="Sum of 3 palindromes",
    description="Can be written as the sum of three palindromic numbers in base 10.",
    oeis="A261132",
    category=CATEGORY,
    limit=7_000_000,
)
def is_sum_of_3_palindromes(n: int, max_solutions=None, min_palindrome_digits=1):
    if n < 0:
        return False, None

    # normalize settings
    cfg_max = CFG("DIOPHANTINE.MAX_SOL_SUM_OF_3_PALINDROMES", None)
    max_solutions = cfg_max if cfg_max is not None else max_solutions

    cfg_min = CFG("DIOPHANTINE.MIN_PALINDROME_DIGITS", None)
    try:
        min_digits = int(cfg_min) if cfg_min is not None else int(min_palindrome_digits)
    except Exception:
        min_digits = 1
    min_digits = max(min_digits, 1)

    allow_zero = bool(CFG("DIOPHANTINE.ALLOW_ZERO_IN_DECOMP", True))

    # FAST PATH: n palindrome ⇒ n = n + 0 + 0 (only if zero + min_digits == 1)
    if (
        allow_zero
        and min_digits == 1
        and _is_pal(n)
        and (len(str(n)) >= min_digits)
        and (n % 10 != 0 or n == 0)
    ):
        return True, f"{n} = {n} + 0 + 0"

    # SEARCH PATH: palindromes respecting min_digits and zero rules
    pals = list(_palindromes_up_to(n, min_digits=min_digits))  # <-- important change
    pal_set = set(pals)

    if not allow_zero and 0 in pal_set:
        pal_set.remove(0)
        pals = [p for p in pals if p != 0]

    results: list[tuple[int, int, int]] = []
    truncated = False

    # iterate non-decreasing a ≤ b ≤ c
    for ai, a in enumerate(pals):
        m = n - a
        if m < 0:
            break

        for bi in range(ai, len(pals)):
            b = pals[bi]
            if b > m:
                break
            c = m - b
            if c < b:
                break

            if not allow_zero and c == 0:
                continue

            if c in pal_set:
                results.append((a, b, c))
                if max_solutions is not None and len(results) >= max_solutions:
                    truncated = True
                    break
        if truncated:
            break

    if results:
        shown = results[:max_solutions] if max_solutions else results
        details = f"{n} = " + "; ".join(f"{a} + {b} + {c}" for a, b, c in shown)
        if truncated:
            details += f" (Limited to {max_solutions} solutions as specified in settings.)"
        return True, details

    if min_digits > 1:
        return False, (
            f"Minimum palindrome digit length is set to {min_digits} in settings. "
            "No decomposition found with palindromes of at least that many digits."
        )

    return False, None
