# -----------------------------------------------------------------------------
#  Digit based test functions
# -----------------------------------------------------------------------------

import math
import textwrap
from decorators import classifier, limited_to
from utility import get_terminal_width
from sympy import isprime, factorint
from user.settings import MAX_SOLUTIONS_SUM_OF_PALINDROMES, MIN_PALINDROME_DIGITS
from typing import Tuple


CATEGORY = "Digit-based"


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
    label="Disarium number",
    description="Sum of digits^position equals the given number.",
    oeis="A032799",
    category=CATEGORY
)
def is_disarium_number(n: int) -> Tuple[bool, str]:
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
def is_dudeney_number(n: int) -> (bool, str):
    """
    Returns (True, details) if n is a Dudeney number.
    A Dudeney number is a perfect cube such that the sum of its digits equals the cube root.
    Example: 512 → 5 + 1 + 2 = 8, and 8^3 = 512.
    """
    if n < 1:
        return False, None
    cube_root = round(n ** (1 / 3))
    if cube_root ** 3 != n:
        return False, None
    s = sum(int(d) for d in str(n))
    if s != cube_root:
        return False, None

    # Formatting details
    width = get_terminal_width()
    digits = " + ".join(str(d) for d in str(n))
    eqn = f"{n} = {cube_root}^3, and {cube_root} = {digits}"
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent="",
        subsequent_indent=" " * 13
    )
    details = wrapper.fill(eqn)
    return True, details


@classifier(
    label="Evil number",
    description="Has an even number of 1's in binary representation for |n|.",
    oeis="A001969",
    category=CATEGORY
)
def is_evil_number(n) -> Tuple[bool, str]:
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
    details = (
        f"Binary: {bits}, number of 1's: {num_ones} (even), "
        f"total bits: {total_bits}"
    )
    if is_evil:
        return True, details
    return False, None


@classifier(
    label="Factorion",
    description="Equals sum of factorials of digits.",
    oeis="A014080",
    category=CATEGORY
)
def is_factorion(n: int) -> Tuple[bool, str]:
    """
    Check if n is a factorion.
    A factorion is a number equal to the sum of the factorials of its digits.
    Returns (True, details) or False.
    """
    KNOWN = {1, 2, 145, 40585}
    if n not in KNOWN:
        return False, None

    digits = list(map(int, str(abs(n))))
    factorials = [f"{d}!={math.factorial(d)}" for d in digits]
    detail = f"{n} = " + " + ".join(factorials)

    # Use your text wrapper for alignment (if desired)
    width = get_terminal_width() - 13
    import textwrap
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent="",
        subsequent_indent=" " * 13
    )
    details = wrapper.fill(detail)
    return True, details


@classifier(
    label="Happy number",
    description="Summing squares of digits repeatedly reaches 1.",
    oeis="A007770",
    category=CATEGORY
)
def is_happy_number(n: int) -> Tuple[bool, str]:
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
    orig = n
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
        seq.append(n)
    if n == 1:
        seq_str = " → ".join(str(x) for x in seq)
        details = f"{orig} → {seq_str} (happy sequence reaches 1)"
        return True, details
    return False, None


@classifier(
    label="Harshad number",
    description="A number divisible by the sum of its digits (also called a Niven number).",
    oeis="A005349",
    category=CATEGORY
)
def is_harshad_number(n: int) -> Tuple[bool, str]:
    if n < 1:
        return False
    digit_sum = sum(int(d) for d in str(abs(n)))
    if digit_sum == 0:
        return False
    if n % digit_sum == 0:
        details = f"{n} is divisible by the sum of its digits: {n} ÷ {digit_sum} = {n // digit_sum}"
        return True, details
    return False, None


@classifier(
    label="Kaprekar number",
    description="Square split sums back to the given number.",
    oeis="A006886",
    category=CATEGORY
)
def is_kaprekar_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True/False, details) for whether n is a Kaprekar number.
    A Kaprekar number is a non-negative integer whose square can be split into
    two parts that sum to the original number.
    Example: 45^2 = 2025 → 20 + 25 = 45
    """
    if n < 1:
        return False, None

    s = str(n * n)
    found = False
    steps = []
    for i in range(1, len(s)):
        l_str, r_str = s[:i], s[i:]
        l, r = int(l_str or "0"), int(r_str or "0")
        check = f"{l_str or '0'} + {r_str} = {l} + {r} = {l + r}"
        if r > 0 and l + r == n:
            found = True
            steps.append(f"• {n}² = {s} → {l_str or '0'} | {r_str} → {check} ✓")
    if n == 1:
        return True, "1 is Kaprekar by definition."

    if found:
        # Formatting
        details = "\n".join([(line) for line in steps])
        return True, details
    return False, None


@classifier(
    label="Lychrel number",
    description="Does not form a palindrome by reverse-and-add.",
    oeis="A023108",
    category=CATEGORY
)
def is_lychrel_number(n: int, max_iter=1000) -> Tuple[bool, str]:
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
    details = (f"{n} is (likely) a Lychrel number: no palindrome found in {max_iter} iterations.")
    return True, details


@classifier(
    label="Narcissistic number",
    description="Sum of digits^n where n is digit count (also called an Armstrong number.",
    oeis="A005188",
    category=CATEGORY
)
def is_narcissistic_number(n: int) -> Tuple[bool, str]:
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
    category=CATEGORY
)
def is_odious_number(n: int) -> Tuple[bool, str]:
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
    details = (
        f"Binary: {bits}, number of 1's: {num_ones} (odd), "
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
def is_palindrome(n: int) -> Tuple[bool, str]:
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
    description="A number consisting of repeated instances of the same digit",
    oeis="A010785",
    category=CATEGORY
)
def is_repdigit(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if all digits are the same.
    """
    if n < 10:
        return False, None
    s = str(abs(n))
    if len(s) > 1 and len(set(s)) == 1:
        return True, f"All digits are {s[0]}: {n} is a repdigit."
    return False, None


@classifier(
    label="Self number",
    description="Cannot be written as m + sum of digits of m for any m (also called Devlali number).",
    oeis="A003052",
    category=CATEGORY
)
def is_self_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a self number (Devlali number), else False.
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
    description="Each digit describes the count of that digit's position in the number.",
    oeis="A046043",
    category=CATEGORY
)
def is_self_descriptive_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a self-descriptive number.
    A self-descriptive number is an integer in which each digit describes
    how many times each digit (0, 1, ..., length-1) appears in the number.
    Returns (True, details) if so, else False.
    """
    if n < 0:
        return False, None
    s = str(n)
    length = len(s)
    # All digits must be in 0..length-1
    if any(int(ch) > length for ch in s):
        return False, None
    matches = [
        f"{s.count(str(i))} occurrence(s) of digit {i} (should match digit in position {i}: {s[i]})"
        for i in range(length)
    ]
    if all(s.count(str(i)) == int(s[i]) for i in range(length)):
        details = (
            f"{n} is self-descriptive: "
            + "; ".join(matches)
        )
        return True, details
    return False, None


@classifier(
    label="Smith number",
    description="A composite number whose digit sum equals the sum of its prime factors' digits.",
    oeis="A006753",
    category=CATEGORY
)
def is_smith_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Smith number.
    A Smith number is a composite number for which the sum of its digits equals
    the sum of the digits in its prime factorization.
    Returns (True, details) if so, else False.
    """
    if n < 4 or isprime(n):
        return False, None
    digits_sum = sum(int(d) for d in str(abs(n)))
    factors = factorint(n)
    pf_digits_sum = sum(sum(int(d) for d in str(p)) * exp for p, exp in factors.items())
    if digits_sum == pf_digits_sum:
        factors_str = ' * '.join([f"{p}^{exp}" if exp > 1 else f"{p}" for p, exp in factors.items()])
        details = (f"Sum of digits: {digits_sum}; "
                   f"Prime factorization: {factors_str} "
                   f"(digits sum to {pf_digits_sum})")
        return True, details
    return False, None


@classifier(
    label="Sum of 2 palindromes",
    description="Can be written as the sum of two palindromic numbers in base 10.",
    oeis="A088601",
    category=CATEGORY
)
@limited_to(99999)
def is_sum_of_2_palindromes(n: int, max_solutions=None, min_palindrome_digits=1) -> Tuple[bool, str]:
    """
    Returns (True, details) if n can be written as the sum of two palindromes (base 10).
    Shows up to max_solutions decompositions. If None, finds all solutions.
    """
    if n < 0:
        return False, None
    try:
        if max_solutions is None:
            max_solutions = MAX_SOLUTIONS_SUM_OF_PALINDROMES.get("2", None)
        min_digits = MIN_PALINDROME_DIGITS
    except ImportError:
        min_digits = min_palindrome_digits

    def is_palindrome(x):
        s = str(x)
        return s == s[::-1] and len(s) >= min_digits and not s.endswith('0')

    result_list = []
    truncated = False
    for a in range(1, n // 2 + 1):
        if not is_palindrome(a):
            continue
        b = n - a
        if b < a:
            break
        if is_palindrome(b):
            result_list.append((a, b))
            if max_solutions is not None and len(result_list) >= max_solutions:
                truncated = True
                break

    if result_list:
        shown = result_list[:max_solutions] if max_solutions else result_list
        details = f"{n} = " + "; ".join(f"{a} + {b}" for (a, b) in shown)
        if truncated:
            details += f" (Limited to {max_solutions} solutions as specified in settings.)"
        return True, details
    else:
        if min_digits > 1:
            details = (
                f"Minimum palindrome digit length is set to {min_digits} in settings. "
                "No decomposition found with palindromes of at least that many digits."
            )
            return False, details
        else:
            return False, None


@classifier(
    label="Sum of 3 palindromes",
    description="Can be written as the sum of three palindromic numbers in base 10.",
    oeis="A088602",
    category=CATEGORY
)
@limited_to(99999)
def is_sum_of_3_palindromes(n: int, max_solutions=None, min_palindrome_digits=1) -> Tuple[bool, str]:
    """
    Returns (True, details) if n can be written as the sum of three palindromes (in base 10).
    Shows up to max_solutions decompositions, and indicates if limited by settings.
    If None, finds all solutions.
    """
    if n < 0:
        return False, None
    try:
        if max_solutions is None:
            max_solutions = MAX_SOLUTIONS_SUM_OF_PALINDROMES.get("3", None)
        min_digits = MIN_PALINDROME_DIGITS
    except ImportError:
        min_digits = min_palindrome_digits

    def is_palindrome(x):
        s = str(x)
        return s == s[::-1] and len(s) >= min_digits and not s.endswith('0')

    result_list = []
    checked = set()
    truncated = False
    for a in range(1, n + 1):
        if not is_palindrome(a):
            continue
        for b in range(a, n - a + 1):
            if not is_palindrome(b):
                continue
            c = n - a - b
            if c < b:
                break
            if is_palindrome(c):
                triplet = (a, b, c)
                if triplet in checked:
                    continue
                checked.add(triplet)
                result_list.append(triplet)
                if max_solutions and len(result_list) >= max_solutions:
                    truncated = True
                    break
        if truncated:
            break

    if result_list:
        shown = result_list[:max_solutions] if max_solutions else result_list
        details = f"{n} = " + "; ".join(f"{a} + {b} + {c}" for (a, b, c) in shown)
        if truncated:
            details += f" (Limited to {max_solutions} solutions as specified in settings.)"
        return True, details
    else:
        if min_digits > 1:
            details = (
                f"Minimum palindrome digit length is set to {min_digits} in settings. "
                "No decomposition found with palindromes of at least that many digits."
            )
            return False, details
        else:
            return False, None
