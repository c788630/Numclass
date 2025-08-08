# -----------------------------------------------------------------------------
#  Prime and Prime-related test functions
# -----------------------------------------------------------------------------

from decorators import classifier, limited_to
from math import log, log2
from sympy import isprime, factorint, prime, primepi, primerange, prevprime, nextprime
from itertools import permutations
from typing import Tuple
from utility import get_ordinal_suffix


CATEGORY = "Primes & Prime-related Numbers"


"""
Atomic prime test definitions only, intersection classes are dynamically generated.
Intersection exceptions are: Motzkin prime, Catalan prime and Keith prime
which have a lookup list.
"""


@classifier(
    label="Absolute prime",
    description=" All digit permutations are prime.",
    oeis="A003459",
    category=CATEGORY
)
@limited_to(99999999)
def is_absolute_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is an absolute (permutable) prime.
    Details show all unique permutations and whether they're prime.
    """
    if n < 2:
        return False, None
    s = str(n)
    perms = sorted({int("".join(p)) for p in set(permutations(s))})
    prime_perms = [p for p in perms if isprime(p)]
    all_are_prime = len(prime_perms) == len(perms) and isprime(n)

    if all_are_prime and len(perms) > 1:
        details = "All digit permutations are prime: " + ", ".join(str(x) for x in perms)
        return True, details
    return False, None


@classifier(
    label="Balanced prime",
    description="equal to the average of the nearest primes: p=(p₁+p₂)/2, where p₁, p₂ are the adjacent primes.",
    oeis="A006562",
    category=CATEGORY
)
@limited_to(99999999)
def is_balanced_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a balanced prime.
    Details show previous and next prime and the average.
    """
    if n < 2:
        return False, None
    if not isprime(n):
        return False, None
    i = primepi(n)
    if i < 2:
        return False, None  # Need one before and one after
    prev_prime = prime(i - 1)
    next_prime = prime(i + 1)
    if n == (prev_prime + next_prime) // 2:
        details = f"{n} is the average of previous prime {prev_prime} and next prime {next_prime}."
        return True, details
    return False, None


@classifier(
    label="Both-truncatable prime",
    description="Both left- and right-truncatable.",
    oeis="A020994",
    category=CATEGORY
)
def is_both_truncatable_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is both left- and right-truncatable prime.
    These are primes that remain prime when digits are removed from
    either the left or the right, one at a time, show both paths.
    """
    if n < 10:
        return False, None
    left = is_left_truncatable_prime(n)
    right = is_right_truncatable_prime(n)
    if isinstance(left, tuple) and left[0] and isinstance(right, tuple) and right[0]:
        details = f"left: {left[1]}\nright: {right[1]}"
        return True, details
    return False, None


@classifier(
    label="Catalan prime",
    description="A Catalan number that is prime.",
    oeis=None,
    category=CATEGORY
)
def is_catalan_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Catalan prime.
    A Catalan prime is a Catalan number that is also prime.
    Only two such numbers are known: 2 and 5.
    """
    KNOWN = {2, 5}
    if n in KNOWN:
        return True, f"{n} is a Catalan number and also a prime."
    return False, None


@classifier(
    label="Chen prime",
    description="Prime n where n+2 is prime or semiprime.",
    oeis="A109611",
    category=CATEGORY
)
def is_chen_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Chen prime.
    Details explain whether n+2 is prime or semiprime.
    """
    if n < 2:
        return False, None
    if not isprime(n):
        return False, None

    q = n + 2
    if isprime(q):
        details = f"{n} is prime and {q} is also prime (Chen pair)."
        return True, details

    # Check if q is semiprime (exactly two prime factors, counted with multiplicity)
    factors = []
    temp = q
    for prime_factors in primerange(2, int(q**0.5) + 1):
        if temp % prime_factors == 0:
            factors.append(prime_factors)
            temp //= prime_factors
            if temp % prime_factors == 0:
                factors.append(prime_factors)
                temp //= prime_factors
                break
            if isprime(temp):
                factors.append(temp)
                temp = 1
                break
    if temp > 1 and temp != q:
        # temp is a factor greater than sqrt(q)
        factors.append(temp)
    if len(factors) == 2 and factors[0] != 1 and factors[1] != 1:
        details = f"{n} is prime and {q} = {factors[0]} × {factors[1]} is semiprime (Chen pair)."
        return True, details
    return False, None


@classifier(
    label="Circular prime",
    description="All digit rotations are prime.",
    oeis="A068652",
    category=CATEGORY
)
def is_circular_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a circular prime, else False.
    Details show all rotations.
    """
    if n < 2:
        return False, None
    s = str(n)
    rotations = [int(s[i:] + s[:i]) for i in range(len(s))]
    are_all_primes = all(isprime(r) for r in rotations)

    if are_all_primes:
        rot_str = ", ".join(str(r) for r in rotations)
        details = f"All rotations are prime: {rot_str}"
        return True, details
    return False, None


@classifier(
    label="Cousin prime",
    description="Prime with another prime 4 away.",
    oeis="A046132",
    category=CATEGORY
)
def is_cousin_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a cousin prime, else False.
    Details show the cousin prime pair.
    """
    if n < 2:
        return False, None
    if not isprime(n):
        return False, None
    if isprime(n - 4):
        details = f"{n} and {n-4} are cousin primes (differ by 4)."
        return True, details
    if isprime(n + 4):
        details = f"{n} and {n+4} are cousin primes (differ by 4)."
        return True, details
    return False, None


@classifier(
    label="emirp",
    description="Prime that is a different prime when reversed.",
    oeis="A006567",
    category=CATEGORY
)
def is_emirp(n: int) -> Tuple[bool, str]:
    """
    Check if n is an emirp prime.
    Details show the reversal and primality of both n and its reverse.
    """
    if n < 13:
        return False, None
    r = int(str(n)[::-1])
    if r != n and isprime(n) and isprime(r):
        details = f"{n} is prime, reverse is {r}, which is also prime."
        return True, details
    return False, None


@classifier(
    label="Factorial prime",
    description="Prime of form n!±1.",
    oeis="A007489",
    category=CATEGORY
)
def is_factorial_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a factorial prime (±1 from a factorial).
    """

    FACTORIAL_PRIMES = sorted([
        2, 3, 5, 7, 23, 719, 5039, 39916801,
        479001599, 87178291199, 6227020801, 1307674368001
    ])

    if n in FACTORIAL_PRIMES:
        idx = FACTORIAL_PRIMES.index(n) + 1
        return True, f"{n} is the {idx}{get_ordinal_suffix(idx)} known factorial prime."
    return False, None


@classifier(
    label="Fermat prime",
    description="Prime of form 2^(2^k)+1.",
    oeis="A019434",
    category=CATEGORY
)
def is_fermat_prime(n: int):
    """
    Check if n is a Fermat prime (of form 2^(2^k) + 1).
    """

    FERMAT_PRIMES = sorted([
        3, 5, 17, 257, 65537  # Only 5 known
    ])

    if n in FERMAT_PRIMES:
        idx = FERMAT_PRIMES.index(n) + 1
        return True, f"{n} is the {idx}{get_ordinal_suffix(idx)} known Fermat prime. Only 5 such primes are known."
    return False, None


@classifier(
    label="Gaussian prime",
    description="Prime in the ring of Gaussian integers.",
    oeis="A055025",
    category=CATEGORY
)
def is_gaussian_prime(n: int) -> Tuple[bool, str]:
    """
    Check whether n is a Gaussian prime in Z[i].
    Details include the type (congruent to 3 mod 4, or sum of squares, etc.)
    """
    if n < 2:
        return False, None
    # Case 1: Prime and 3 mod 4
    if isprime(n) and n % 4 == 3:
        details = f"{n} is a (rational) prime and {n} ≡ 3 mod 4; thus, a Gaussian prime."
        return True, details

    # Case 2: n = a^2 + b^2, n prime, both a and b nonzero
    if isprime(n) and n % 4 == 1:
        # Find a representation as a^2 + b^2
        for a in range(1, int(n ** 0.5) + 1):
            b2 = n - a*a
            b = int(b2 ** 0.5)
            if b > 0 and b * b == b2:
                details = (f"{n} is a (rational) prime ≡ 1 mod 4 and can be written as "
                           f"{a}^2 + {b}^2; thus, {a} + {b}i is a Gaussian prime.")
                return True, details

    # Case 3: n = p^2, where p is a rational prime ≡ 3 mod 4
    if n > 1:
        sqrt_n = int(n ** 0.5)
        if sqrt_n * sqrt_n == n and isprime(sqrt_n) and sqrt_n % 4 == 3:
            details = (f"{n} = {sqrt_n}^2, and {sqrt_n} is a prime ≡ 3 mod 4; "
                       "this is the norm of a Gaussian prime on the axes.")
            return True, details

    return False, None


@classifier(
    label="Good prime",
    description="Prime greater than average of its neighbors.",
    oeis="A028388",
    category=CATEGORY
)
@limited_to(9999999)
def is_good_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a good prime.
    A good prime is a prime number that is greater than the
    arithmetic mean of the nearest primes above and below it.
    """
    if n < 2:
        return False, None
    if not isprime(n):
        return False, None
    i = primepi(n)
    if i < 2:
        return False, None  # need at least one before and after
    prev_p = prime(i - 1)
    next_p = prime(i + 1)
    mean = (prev_p + next_p) // 2
    if n > mean:
        details = f"{n} > ({prev_p} + {next_p}) / 2 = {mean} (good prime condition satisfied)."
        return True, details
    return False, None


@classifier(
    label="Isolated prime",
    description="Prime number that is not part of a twin prime pair (n-2 and n+2 are not prime).",
    oeis="A007510",
    category=CATEGORY
)
def is_isolated_prime(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is an isolated prime: neither n-2 nor n+2 is prime.
    """
    if n < 2 or not isprime(n):
        return False, None
    prev_is_prime = isprime(n - 2)
    next_is_prime = isprime(n + 2)
    if not prev_is_prime and not next_is_prime:
        details = f"{n} is prime, but {n-2} and {n+2} are not."
        return True, details
    return False, None


@classifier(
    label="Keith prime",
    description=("An integer > 9 that is both a Keith number and a prime number, "
                 "using the standard base-10 digit-recurrence rule."),
    oeis="A048970",
    category=CATEGORY
)
@limited_to(74596893730428)
def is_keith_prime(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a Keith prime, else False.
    Keith primes are very rare: sequence based on the strictbase-10
    digit-reocurrence definition:
    A Keith prime is an integer > 9 that is both a Keith number and
    a prime number, using the standard base-10 digit-recurrence rule.
    This means that a Keith prime is not just an intersection between a
    Keith number and a prime.
    """

    KEITH_PRIMES = sorted([
        19, 47, 61, 197, 1084051, 74596893730427
    ])

    if n in KEITH_PRIMES:
        idx = KEITH_PRIMES.index(n) + 1
        return True, f"{n} is the {idx}{get_ordinal_suffix(idx)} known Keith prime."
    return False, None


@classifier(
    label="Left-truncatable prime",
    description="Remains prime when leading digits removed.",
    oeis="A024770",
    category=CATEGORY
)
def is_left_truncatable_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a left-truncatable prime.
    A left-truncatable prime remains prime when digits are successively
    removed from the left. Example: 317 -> 17 -> 7; all are primes.
    """
    if n < 10:
        return False, None
    s = str(n)
    seq = []
    for i in range(len(s)):
        val = int(s[i:])
        if not isprime(val):
            return False, None
        seq.append(val)
    seq_str = " → ".join(str(x) for x in seq)
    return True, seq_str


@classifier(
    label="Mersenne prime",
    description="Prime of form 2^p−1 where p is prime.",
    oeis="A000668",
    category=CATEGORY
)
def is_mersenne_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a known Mersenne prime (2^p - 1).
    """

    MERSENNE_PRIMES = sorted([
        3, 7, 31, 127, 8191, 131071, 524287, 2147483647,
        2305843009213693951, 618970019642690137449562111,
    ])

    if n in MERSENNE_PRIMES:
        idx = MERSENNE_PRIMES.index(n) + 1
        return True, f"{n} is the {idx}{get_ordinal_suffix(idx)} known Mersenne prime."
    return False, None


@classifier(
    label="Motzkin prime",
    description="A Motzkin number that is also a prime. \nMotzkin numbers count certain lattice paths and combinatorial structures",
    oeis="A092832",
    category=CATEGORY
)
def is_motzkin_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Motzkin prime.
    A Motzkin prime is a Motzkin number that is also a prime number.
    Only 4 Motzkin primes are known.
    """

    MOTZKIN_PRIMES = sorted([
        2, 127, 15511, 953467954114363
    ])

    if n in MOTZKIN_PRIMES:
        idx = MOTZKIN_PRIMES.index(n) + 1
        return True, f"{n} is a Motzkin number and also the {idx}{get_ordinal_suffix(idx)} known Motzkin prime."
    return False, None


@classifier(
    label="Pierpont prime",
    description="Prime of form 2^a·3^b+1, also known as Class 1 primes.",
    oeis="A005109",
    category=CATEGORY
)
@limited_to(99999999)
def is_pierpont_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Pierpont prime (of the first or second kind).
    Details show exponents: n = 2^a·3^b ± 1
    """
    if n < 2 or not isprime(n):
        return False, None

    for sign, opstr in [(-1, "- 1"), (1, "+ 1")]:
        candidate = n - sign
        max_a = int(log(candidate, 2)) + 1 if candidate > 0 else 0
        for a in range(max_a):
            pow2 = 2 ** a
            if candidate % pow2 != 0:
                continue
            quotient = candidate // pow2
            if quotient < 1:
                continue
            # Now check if quotient is a power of 3
            b = log(quotient, 3)
            if b.is_integer():
                details = f"{n} = 2^{a}·3^{int(b)} {opstr}"
                return True, details
    return False, None


@classifier(
    label="Prime number",
    description="Is a prime.",
    oeis=None,
    category=CATEGORY
)
def is_prime_number(n: int) -> Tuple[bool, str]:
    """
    Determines if n is prime. The is_prime_number function is only
    needed for intersection rules, it is filtered out as classifier.
    """
    if isprime(n):
        return True, None
    return False, None


@classifier(
    label="Primorial prime",
    description="Prime of form p#±1, p#=product of first n primes.",
    oeis="A005234",
    category=CATEGORY
)
def is_primorial_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a known primorial prime (±1 from product of first k primes).
    """

    PRIMORIAL_PRIMES = sorted([
        2, 3, 5, 29, 2309, 30029, 200560490131,
        304250263527209, 13082761331670030
    ])

    if n in PRIMORIAL_PRIMES:
        idx = PRIMORIAL_PRIMES.index(n) + 1
        return True, f"{n} is the {idx}{get_ordinal_suffix(idx)} known primorial prime."
    return False, None


@classifier(
    label="Proth prime",
    description="Prime of form k·2^m+1 with k odd and k<2^m.",
    oeis="A080076",
    category=CATEGORY
)
@limited_to(99999999)
def is_proth_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Proth prime.
    Details show n = k·2^n + 1 with odd k < 2^n.
    """
    if n < 3 or not isprime(n):
        return False, None
    for m in range(1, int(log2(n))):
        k = (n - 1) / (2 ** m)
        if k.is_integer() and int(k) % 2 == 1 and k < 2 ** n:
            details = f"{n} = {int(k)}·2^{m} + 1 (k = {int(k)}, m = {m})"
            return True, details
    return False, None


def load_ramanujan_primes(filename="data/b104272.txt"):
    """
    Classification of Ramanujan primes depends on file 'b104272.txt',
    present in the data subfolder.
    Download or update from: https://oeis.org/A104272/b104272.txt
    """
    value_to_index = {}
    try:
        with open(filename) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        idx, val = int(parts[0]), int(parts[1])
                        value_to_index[val] = idx
    except FileNotFoundError:
        print(f"Warning: '{filename}' file not found. Ramanujan primes test will be disabled.")
    except Exception as e:
        print(f"Warning: Error loading '{filename}': {e}")
    return value_to_index


@classifier(
    label="Ramanujan prime",
    description="Smallest R where π(x)-π(x/2) ≥ n for x ≥ R.",
    oeis="A104272",
    category=CATEGORY
)
@limited_to(242058)
def is_ramanujan_prime(n: int) -> Tuple[bool, str]:
    ramanujan_prime_lookup = load_ramanujan_primes()
    idx = ramanujan_prime_lookup.get(n)
    if idx:
        suffix = get_ordinal_suffix(idx)
        return True, f"{n} is the {suffix} Ramanujan prime."
    return False, None


@classifier(
    label="Right-truncatable prime",
    description="Remains prime when trailing digits removed.",
    oeis="A024785",
    category=CATEGORY
)
def is_right_truncatable_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a right-truncatable prime.
    A right-truncatable prime remains prime when digits are successively
    removed from the right.
    Example: 739397 -> 73939 -> 7393 -> 739 -> 73 -> 7; all are primes.
    """
    if n < 10:
        return False, None
    s = str(n)
    seq = []
    for i in range(len(s), 0, -1):
        val = int(s[:i])
        if not isprime(val):
            return False, None
        seq.append(val)
    seq_str = " → ".join(str(x) for x in seq)
    return True, seq_str


@classifier(
    label="Safe prime",
    description="Prime where (n−1)/2 is also prime.",
    oeis="A005385",
    category=CATEGORY
)
def is_safe_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a safe prime.
    A safe prime is a prime number of the form 2k + 1, where k is also prime.
    """
    if n < 5:
        return False, None
    if not isprime(n):
        return False, None
    k = (n - 1) // 2
    if 2 * k + 1 != n:
        return False, None  # n must be odd
    if isprime(k):
        return True, f"{n} = 2×{k} + 1 with {k} also prime (safe prime condition satisfied)"
    return False, None


@classifier(
    label="Sexy prime",
    description="Prime with another prime 6 away.",
    oeis="A023201",
    category=CATEGORY
)
def is_sexy_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a sexy prime.
    Details show the sexy prime pair.
    """
    if n < 5:
        return False, None
    if not isprime(n):
        return False, None
    if isprime(n - 6):
        details = f"{n} and {n-6} are sexy primes (differ by 6)."
        return True, details
    if isprime(n + 6):
        details = f"{n} and {n+6} are sexy primes (differ by 6)."
        return True, details
    return False, None


@classifier(
    label="Semiprime",
    description="Product of exactly two (not necessarily distinct) primes.",
    oeis="A001358",
    category=CATEGORY
)
def is_semiprime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a semiprime (the product of two primes, not necessarily distinct).
    Returns (True, details) or False.
    """
    if n < 4:
        return False, None  # Smallest semiprime is 4 = 2*2
    factors = factorint(n)
    total_factors = sum(factors.values())
    if total_factors == 2:
        primes = []
        for p, exp in factors.items():
            primes.extend([p] * exp)
        detail = f"{n} = {' × '.join(str(p) for p in primes)}"
        return True, detail
    return False, None


@classifier(
    label="Sophie Germain prime",
    description="p where 2p+1 is also prime.",
    oeis="A005384",
    category=CATEGORY
)
def is_sophie_germain_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Sophie Germain prime.
    Details show that both n and 2n+1 are prime.
    """
    if n < 2:
        return False, None
    if isprime(n) and isprime(2 * n + 1):
        details = f"{n} is prime and 2×{n} + 1 = {2 * n + 1} is also prime."
        return True, details
    return False, None


@classifier(
    label="Strong prime",
    description="A prime that is greater than the arithmetic mean of the nearest primes above and below it.",
    oeis="A051634",
    category=CATEGORY
)
def is_strong_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a strong prime.
    A strong prime is greater than the arithmetic mean of the nearest primes
    above and below it.
    """
    if n < 11 or not isprime(n):
        return False, None
    p_prev = prevprime(n)
    p_next = nextprime(n)
    mean = (p_prev + p_next) / 2
    if n > mean:
        details = (f"{n} > (prev: {p_prev} + next: {p_next}) / 2 = {mean:.1f}; "
                   "so it is a strong prime.")
        return True, details
    return False, None


@classifier(
    label="Super prime",
    description="Prime whose index in primes is prime.",
    oeis="A006005",
    category=CATEGORY
)
@limited_to(99999999)
def is_super_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a super-prime.
    Details show the prime index of n.
    """
    if n < 1:
        return False, None
    if not isprime(n):
        return False, None
    pos = primepi(n)  # Number of primes ≤ n
    if isprime(pos):
        details = f"{n} is the {pos}th prime, and {pos} is also prime."
        return True, details
    return False, None


@classifier(
    label="Twin prime",
    description="Prime with another prime 2 away.",
    oeis="A001097",
    category=CATEGORY
)
def is_twin_prime(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a twin prime, else False.
    Details show the twin prime pair.
    """
    if n < 3:
        return False, None
    if not isprime(n):
        return False, None
    if isprime(n - 2):
        details = f"{n} and {n-2} are twin primes (differ by 2)."
        return True, details
    if isprime(n + 2):
        details = f"{n} and {n+2} are twin primes (differ by 2)."
        return True, details
    return False, None


@classifier(
    label="Wieferich prime",
    description="2^(p−1) ≡ 1 mod p^2",
    oeis="A001220",
    category=CATEGORY
)
def is_wieferich_prime(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a Wieferich prime, else False.
    Only two are known as of 2025.
    Details explain the Wieferich congruence.
    """
    KNOWN = {1093, 3511}
    if n in KNOWN:
        details = f"{n} is a Wieferich prime: 2^({n}-1) ≡ 1 mod {n}², only 2 such primes are known."
        return True, details
    return False, None


@classifier(
    label="Wilson prime",
    description="(p−1)! ≡ −1 mod p^2",
    oeis="A007540",
    category=CATEGORY
)
def is_wilson_prime(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Wilson prime.
    A Wilson prime is a prime p such that (p−1)! ≡ −1 mod p².
    Only three are known: 5, 13, and 563.
    """

    WILSON_PRIMES = [5, 13, 563]

    if n in WILSON_PRIMES:
        idx = WILSON_PRIMES.index(n) + 1
        return True, f"{n} is the {get_ordinal_suffix(idx)} known Wilson prime. Only 3 such primes are known."
    return False, None
