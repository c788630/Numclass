# -----------------------------------------------------------------------------
# Mathematical curiosities test functions
# -----------------------------------------------------------------------------

from decorators import classifier, limited_to
from itertools import permutations
from sympy import prime, isprime, prevprime, nextprime
from typing import Tuple

CATEGORY = "Mathematical Curiosities"


@classifier(
    label="Binary-interpretable number",
    description="Number can be interpreted as a binary numeral.",
    category=CATEGORY
)
def is_binary_interpretable(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if the decimal representation of n consists only of 0s and 1s,
    and is at least 2 digits (to avoid matching trivial '0' and '1').
    """
    s = str(n)
    if set(s) <= {"0", "1"} and len(s) > 1:
        dec_value = int(s, 2)
        details = (
            f"As binary: {s}₂ = {dec_value}₁₀ = {oct(dec_value)} (octal) = {hex(dec_value)} (hex)"
        )
        return True, details
    return False, None


@classifier(
    label="Boring number",
    description='The first "boring number", a positive number with the least number of classifications (3).',
    oeis=None
)
def is_boring_number(n: int) -> Tuple[bool, str]:
    return n == 796, "Ah, well... make that 4 now, since it’s also classified as a boring number!"


@classifier(
    label="Cake number",
    description="A number of regions formed by n straight cuts in a cake/circle.",
    oeis="A000125",
    category=CATEGORY
)
def is_cake_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Cake number (maximum pieces from straight cuts).
    Details: the cake number for k straight cuts.
    """
    if n < 1:
        return False, None
    # Estimate k using inverse of formula
    approx_k = int(round((6 * n) ** (1/3)))
    # Check a small range around approx_k (e.g. ±5)
    for k in range(max(0, approx_k - 5), approx_k + 6):
        pieces = (k**3 + 5*k + 6) // 6
        if pieces == n:
            return True, f"{n} is the cake number for {k} straight cuts"
    return False, None


@classifier(
    label="Cyclops number",
    description="Odd number of digits and a single zero in the center digit.",
    oeis="A134808",
    category=CATEGORY,
)
def is_cyclops_number(n: int) -> Tuple[bool, str]:
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
    label="Digit-Reversal constant",
    description=("Appears as the result of the classic three-digit reversal"
                 "procedure: reverse, subtract, reverse, add."),
    oeis="A037154",
    category=CATEGORY
)
def is_digit_reversal_constant(n: int) -> Tuple[bool, str]:
    if n == 1089:
        return True, "Start with 532 as example. Reverse: 235, Subtract: 532-235=297, Reverse: 792, Add: 297+792=1089."
    return False, None


@classifier(
    label="Eban number",
    description="No letter 'e' in the English spelling",
    oeis="A006933",
    category=CATEGORY
)
def is_eban_number(n: int) -> Tuple[bool, str]:
    """
    Checks if n is an eban number.
    An eban number has no letter 'e' in its English spelling.
    Only certain even numbers (with allowed digits and positions) qualify.
    All eban numbers >= 2 are even, and their decimal digits are 0, 2, 4, or 6,
    with allowed thousands/millions/billions groupings (which contain no 'e').
    """
    if n < 2 or n % 2 != 0:
        return False, None
    # Only digits 2, 4, 6, 0 allowed in each group of three digits (thousands, millions, etc.)
    # No 'e' in "two", "four", "six", "thousand", "million", "billion", "trillion"
    # So, we check for each group of three digits from the right
    allowed_digits = {'0', '2', '4', '6'}
    s = str(n)[::-1]  # Reverse for easier grouping
    for i, group_start in enumerate(range(0, len(s), 3)):
        group = s[group_start:group_start+3][::-1]
        if group.strip('0') == "":
            continue  # Skip empty groups
        if set(group) - allowed_digits:
            return False, None
    return True, f"{n} is an eban number (no 'e' in its English spelling; only 2,4,6,0 digits in each group)"


@classifier(
    label="Harshad in all bases",
    description="A number that is Harshad in every base.",
    oeis=None,
    category=CATEGORY
)
def is_harshad_in_all_bases(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a Harshad number in all bases 2 through n-1.
    Only n=1, 2, 4, 6 satisfy this property.
    """
    if n in {1, 2, 4, 6}:
        return True, f"{n} is a Harshad number in all bases 2..{n-1} (universal Harshad number)."
    return False, None


@classifier(
    label="Kaprekar Constant (3 digit)",
    description="The Kaprekar constant for 3-digit numbers.",
    oeis=None,
    category=CATEGORY
)
def is_kaprekar_constant_3_digit(n: int) -> Tuple[bool, str]:
    """
    a 3-digit number with at least two different digits, repeatedly:
    - Arrange the digits in descending and ascending order to form two numbers.
    - Subtract the smaller from the larger and repeat with the result.
    - This process converges to 495, the 3-digit Kaprekar constant.
    """
    if n == 495:
        return True, ("Repeatedly subtract any number formed by its digits in "
                      "ascending order from that in descending order. This "
                      "process converges to the Kaprekar constant 495.")
    return False, None


@classifier(
    label="Kaprekar Constant (4 digit)",
    description="The Kaprekar constant for 4-digit numbers.",
    oeis=None,
    category=CATEGORY
)
def is_kaprekar_constant_4_digit(n: int) -> Tuple[bool, str]:
    """
    For 4-digit numbers with at least two different digits, repeatedly:
    - Arrange the digits in descending and ascending order to form two numbers.
    - Subtract the smaller from the larger and repeat with the result.
    - This process converges to 6174, the 4-digit Kaprekar constant.
    """
    if n == 6174:
        return True, ("Repeatedly subtract any number formed by its digits in "
                      "ascending order from that in descending order. This "
                      "process converges to the Kaprekar constant 6175.")
    return False, None


@classifier(
    label="Lucky number of Euler",
    description="Numbers of the form n² + n + 41 which are prime for n = 0..39.",
    oeis="A003173",
    category=CATEGORY
)
def is_lucky_number_of_euler(n) -> Tuple[bool, str]:
    """
    Checks if n is a 'lucky number of Euler' (Heegner number).
    Details: include the near-integer value for e^{π√n}.
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
            f"{n} is a 'lucky number of Euler' (Heegner number).\n"
            f"    {value}\n"
            "    This property is a celebrated mathematical phenomenon!"
        )
        return True, details
    return False, None


@classifier(
    label="Munchausen number",
    description="Equals the sum of its digits each raised to the power of itself.",
    oeis="A046253",
    category=CATEGORY
)
def is_munchausen_number(n: int) -> Tuple[bool, str]:
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
def is_octal_interpretable(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if the decimal representation of n consists only of 0-7,
    and is at least 2 digits (to avoid matching trivial '0'-'7').
    """
    if n < 0:
        return False, None
    s = str(n)
    if set(s) <= set("01234567") and len(s) > 1:
        dec_value = int(s, 8)
        details = (
            f"As octal: {s}₈ = {dec_value}₁₀ = {bin(dec_value)} (binary) = {hex(dec_value)} (hex)"
        )
        return True, details
    return False, None


@classifier(
    label="Pandigital number (0-9)",
    description="Contains each digit 0–9 exactly once.",
    oeis="A171102",
    category=CATEGORY
)
def is_pandigital_number_0_9(n: int) -> Tuple[bool, str]:
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
    oeis="A050278",
    category=CATEGORY
)
def is_pandigital_number_1_9(n: int) -> Tuple[bool, str]:
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


@classifier(
    label="Smarandache Wellin number",
    description="Concatenation of the first k primes (e.g., 2, 23, 235, 2357, …).",
    oeis="A019518",
    category=CATEGORY
)
@limited_to(23571113171923293137414348)
def is_smarandache_wellin_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a Smarandache–Wellin number, else False.
    Details: Shows which primes were concatenated.
    """

    SMARANDACHE_WELLIN = sorted([
        2,
        23,
        235,
        2357,
        235711,
        23571113,
        2357111317,
        235711131719,
        23571113171923,
        2357111317192329,
        235711131719232931,
        23571113171923293137,
        2357111317192329313741,
        235711131719232931374143,
        23571113171923293137414347,
    ])
    if n in SMARANDACHE_WELLIN:
        idx = SMARANDACHE_WELLIN.index(n)
        primes_used = [prime(i + 1) for i in range(idx + 1)]
        details = f"{n} is formed by concatenating the first {idx+1} primes: {', '.join(str(p) for p in primes_used)}."
        return True, details
    return False, None


@classifier(
    label="Strobogrammatic number",
    description="Looks the same upside down.",
    oeis="A007376",
    category=CATEGORY
)
def is_strobogrammatic_number(n: int) -> Tuple[bool, str]:
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
    label="Thick prime",
    description="A prime where previous + next prime is more than twice itself, part of the prime gap class.",
    oeis="A007511",
    category=CATEGORY
)
def is_thick_prime(n) -> Tuple[bool, str]:
    if n < 3 or not isprime(n):
        return False, None
    p_prev = prevprime(n)
    p_next = nextprime(n)
    lhs = p_next + p_prev
    rhs = 2 * n
    if lhs > rhs:
        details = f"{p_prev} + {p_next} = {lhs} > 2×{n} = {rhs} (thick prime condition satisfied)"
        return True, details
    return False, None


@classifier(
    label="Thin prime",
    description="A prime where previous + next prime is less than twice itself, part of the prime gap class.",
    oeis="A192869",
    category=CATEGORY
)
def is_thin_prime(n) -> Tuple[bool, str]:
    if n < 3 or not isprime(n):
        return False, None
    p_prev = prevprime(n)
    p_next = nextprime(n)
    lhs = p_next + p_prev
    rhs = 2 * n
    if lhs < rhs:
        details = f"{p_prev}+{p_next} = {lhs} < 2×{n} = {rhs} (thin prime condition satisfied)"
        return True, details
    return False, None


@classifier(
    label="Vampire number",
    description=" Composite number factorable into two equal-length factors using original digits exactly once.",
    oeis="A014575",
    category=CATEGORY
)
@limited_to(999999)
def is_vampire_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a vampire number.
    A vampire number has an even number of digits and can be written as the
    product of two numbers (fangs), each with half as many digits as n,
    using all the digits of n exactly once.
    Returns (True, details) if so, else False.
    """
    if n < 1000:
        return False, None
    s = str(n)
    L = len(s)
    if L % 2 != 0 or L < 4:
        return False, None

    half = L // 2
    seen = set()
    for perm in set(permutations(s)):
        x = int(''.join(perm[:half]))
        y = int(''.join(perm[half:]))
        # Both fangs can't end in zero
        if x % 10 == 0 and y % 10 == 0:
            continue
        # Avoid duplicate pairs (e.g. (x, y) vs (y, x))
        fang_pair = tuple(sorted((x, y)))
        if fang_pair in seen:
            continue
        seen.add(fang_pair)
        if x * y == n:
            return True, f"{n} = {x} × {y} (vampire number)"
    return False, None
