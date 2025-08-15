# -----------------------------------------------------------------------------
#  Utility functions
# -----------------------------------------------------------------------------

import importlib
import inspect
import os
import pkgutil
import re
import shutil
import sys
import textwrap
import time
import traceback

from colorama import Fore, Style
from functools import lru_cache
from math import gcd
from sympy import divisors, factorint, divisor_sigma, isprime, lcm
from user import settings


def _divisors_from_factorization(f: dict[int, int]) -> list[int]:
    """Return all positive divisors from a prime-power factorization dict."""
    ds = [1]
    for p, e in f.items():
        cur = []
        pe = 1
        for _ in range(e + 1):
            for d in ds:
                cur.append(d * pe)
            pe *= p
        ds = cur
    return sorted(set(ds))


@lru_cache(maxsize=None)
def analyze_divisors(n: int) -> tuple[list[int], int, bool]:
    """
    Return divisor info for n.
    Args:
        n (int): The integer to analyze.
    Returns:
        tuple: (list of divisors, sum of proper divisors, is_prime flag)
    """
    if n < 1:
        return [], 0, False
    divs = divisors(n)  # Sympy returns a sorted list of all divisors
    proper_sum = sum(divs) - n
    prime_flag = isprime(n)
    return divs, proper_sum, prime_flag


def base_name(a: int) -> str:
    """Return string like 'binary (base 2)' for a given base integer."""
    NAMES = {
        2: "binary",
        3: "ternary",
        4: "quaternary",
        5: "quinary",
        6: "senary",
        7: "septenary",
        8: "octal",
        9: "nonary",
        10: "decimal",
    }
    return f"{NAMES.get(a, f'base {a}')}" + f" (base {a})"


def carmichael_with_details(n: int, factors: dict[int, int]):
    """
    Return Carmichael's λ(n) and a compact explanation string.

    Example: λ(2^2)=2, λ(3)=2, λ(5)=4 => lcm(2,2,4)=4
    """
    parts = []
    lam = 1
    for p, a in sorted(factors.items()):
        if p == 2:
            if a == 1:
                lam_pa = 1
            elif a == 2:
                lam_pa = 2
            else:
                lam_pa = 1 << (a - 2)
        else:
            lam_pa = (p - 1) * (p ** (a - 1))
        parts.append(f"λ({p}^{a})={lam_pa}" if a > 1 else f"λ({p})={lam_pa}")
        lam = lcm(lam, lam_pa)

    return lam, f"{', '.join(parts)} ⇒ lcm({', '.join(str(p.split('=')[1]) for p in parts)})={lam}"


def clear_screen():
    """
    Clear the terminal screen on Windows and Unix-based systems.
    """
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # macOS/Linux/Unix
        os.system("clear")


def compute_aliquot_sequence(n, max_steps=200, step_timeout=0.3, om=None):
    """
    Compute the aliquot sequence starting from n, with per-step timeouts and abort support.
    Uses sympy.divisor_sigma for speed. Only outputs per-step progress, not per divisor.
    """

    def clear_line():
        width = get_terminal_width()
        om.write_screen("\r" + " " * width + "\r", end="", flush=True)

    seq = [n]
    seen = set([n])
    highlight_idx = None
    skipped = False
    aborted = False

    allow_slow_calculations = getattr(settings, "ALLOW_SLOW_CALCULATIONS", False)
    if allow_slow_calculations:
        step_timeout = None

    try:
        for step in range(max_steps):
            t0 = time.perf_counter()
            # Compute aliquot sum quickly
            s = divisor_sigma(n, 1) - n
            t1 = time.perf_counter()

            # Progress indicator per step (not per divisor!)
            if om:
                om.write_screen(
                    f"\r  Aliquot sequence:     {Fore.CYAN}"
                    f"Step {step+1}/{max_steps}, aliquot sum {s}          "
                    f"{Style.RESET_ALL}",
                    end="", flush=True
                )

            if step_timeout is not None and (t1 - t0) > step_timeout:
                skipped = True
                clear_line()
                break

            seq.append(s)
            if s == 0 or s in seen:
                if s in seen:
                    highlight_idx = (seq.index(s), len(seq) - 1)
                break
            seen.add(s)
            n = s
        else:
            clear_line()
            aborted = True

    except KeyboardInterrupt:
        clear_line()
        if om:
            om.write("  Aliquot Sequence:     Aborted by user (Ctrl-C)")
        return [], None, False, False

    clear_line()
    return seq, highlight_idx, aborted, skipped


def digit_sum(n: int) -> int:
    """
    Calculate the sum of digits of n.
    Args: n (int): The number.

    Returns: int: The sum of the absolute value digits.
    """
    return sum(int(d) for d in str(abs(n)))


def digit_product(n: int) -> int:
    """
    Return the product of the digits of n (absolute value).
    """
    digits = [int(d) for d in str(abs(n))]
    prod = 1
    for d in digits:
        prod *= d
    return prod


def digital_root_sequence(n: int):
    """
    Yield the digital root sequence for n, repeatedly summing its decimal
    digits until a single-digit is reached, including the starting value.
    """
    seq = [abs(n)]
    while seq[-1] >= 10:
        seq.append(sum(int(d) for d in str(seq[-1])))
    return seq


def filter_maximal_intersections(classes, intersection_label_to_atomic):
    """
    For all present intersections, only keep the most specific (maximal) ones.
    Suppress smaller intersections (their atomic components are a proper subset).
    Preserves order and details.
    """
    labels = [item['label'] for item in classes]
    keep = set(labels)
    # Find all intersection labels present
    present_inters = [lbl for lbl in labels if lbl in intersection_label_to_atomic]
    # Compare all pairs of intersection labels present
    for a in present_inters:
        atoms_a = set(intersection_label_to_atomic[a])
        for b in present_inters:
            if a == b:
                continue
            atoms_b = set(intersection_label_to_atomic[b])
            # If a is a proper subset of b, and both present, suppress a
            if atoms_a < atoms_b:  # strictly smaller
                keep.discard(a)
    # Now remove all atomic labels that are part of any kept intersection
    for inter in list(keep):
        if inter in intersection_label_to_atomic:
            for atom in intersection_label_to_atomic[inter]:
                keep.discard(atom)
    # Rebuild the class list, preserving original order
    return [item for item in classes if item['label'] in keep]


def format_aliquot_sequence(
    seq,
    highlight_idx=None,
    aborted=False,
    skipped=False,
    max_steps=200,
    show_sequence=True
):
    """
    Format the aliquot sequence for display.

    Returns:
        aliquot_wrapped: wrapped sequence string (colorized as before)
        status_lines: list of status/warning strings (colorized)
        sequence_length: integer, the sequence length (steps)
    """
    indent = 24

    if len(seq) < 1:
        return "", [], 0

    if skipped:
        return f"{Fore.YELLOW}[Skipped due to time constraints]{Style.RESET_ALL}", [], 0

    terminal_width = get_terminal_width()

    seq_str_parts = []
    truncated = False

    ALIQUOTLIST_LIMIT = getattr(settings, "ALIQUOTLIST_LIMIT", 50)

    display_seq = seq
    if (
        show_sequence
        and ALIQUOTLIST_LIMIT is not None
        and len(seq) > ALIQUOTLIST_LIMIT
    ):
        display_seq = seq[:ALIQUOTLIST_LIMIT]
        truncated = True

    for idx, val in enumerate(display_seq):
        sval = str(val)
        if highlight_idx and (idx == highlight_idx[0] or idx == highlight_idx[1]):
            sval = f"{Fore.MAGENTA + Style.BRIGHT}{sval}{Style.RESET_ALL}"
        seq_str_parts.append(sval)

    seq_raw = " → ".join(seq_str_parts)

    # Status lines (except sequence length)
    status_lines = []
    if truncated:
        status_lines.append(
            f"{Fore.YELLOW}[Long sequence truncated to the first {ALIQUOTLIST_LIMIT} elements]{Style.RESET_ALL}"
        )
    if aborted:
        status_lines.append(
            f"{Fore.GREEN + Style.BRIGHT}Possible Open Sequence"
            f"{Style.RESET_ALL} detected; maximum step limit ({max_steps}) reached."
        )

    # Sequence length
    if highlight_idx:
        steps = highlight_idx[1]
    else:
        steps = len(seq)

    # Wrapping
    aliquot_wrapped = ""
    if show_sequence:
        wrapper = textwrap.TextWrapper(
            width=terminal_width,
            initial_indent=' ' * indent,
            subsequent_indent=' ' * indent
        )
        aliquot_wrapped = wrapper.fill(seq_raw)[indent:]
    return aliquot_wrapped, status_lines, steps


def format_prime_factors(n: int, *,
                         factors: dict[int, int] | None = None,
                         return_factors: bool = False):
    """
    Format the prime factorization of n as a string.
    If `factors` is provided, it is used (no refactor). If `return_factors`
    is True, also return the factor dict that was used.

    Examples:
    >>> format_prime_factors(60)
    '2^2 * 3 * 5'
    >>> s, facs = format_prime_factors(60, return_factors=True)
    >>> s, facs
    ('2^2 * 3 * 5', {2: 2, 3: 1, 5: 1})
    """
    if n == 0:
        return ("0", {}) if return_factors else "0"
    if abs(n) == 1:
        return (str(n), {}) if return_factors else str(n)

    if factors is None:
        factors = factorint(abs(n))

    parts = []
    for p in sorted(factors):
        exp = factors[p]
        parts.append(f"{p}^{exp}" if exp > 1 else f"{p}")

    result = " * ".join(parts)
    if n < 0:
        result = f"-1 * {result}"

    return (result, factors) if return_factors else result


def get_terminal_width(default=80):
    """
    Return the terminal's character width if detected, else the default
    value (80 by default).
    """
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


def get_terminal_height(default=24):
    """
    Get the number of terminal lines.
    """
    try:
        return shutil.get_terminal_size().lines
    except Exception:
        return default


def get_ordinal_suffix(n):
    """
    Return the ordinal representation of an integer (e.g., 1st, 2nd, 3rd, 4th...).
    """
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def import_classifier_functions(package_folder):
    """
    Import all classifier functions from .py files in a package folder.
    `package_folder` is a filesystem path to a directory containing __init__.py.
    """
    classifier_functions = {}
    folder = os.path.abspath(package_folder)
    package = os.path.basename(folder)  # 'classifiers' (not './classifiers')
    parent = os.path.dirname(folder)

    # Ensure parent on sys.path so we can import 'classifiers'
    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        importlib.import_module(package)
    except Exception as e:
        print(f"Failed to import package '{package_folder}': {e}")
        traceback.print_exc()
        sys.exit(1)

    for filename in os.listdir(folder):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        modname = filename[:-3]
        module_name = f"{package}.{modname}"  # e.g., classifiers.primes
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"Failed to import {module_name}: {e}")
            traceback.print_exc()
            sys.exit(1)

        for name, obj in inspect.getmembers(module):
            if (inspect.isfunction(obj) or inspect.isbuiltin(obj)) and hasattr(obj, "label"):
                classifier_functions[name] = obj

    return classifier_functions


def intersection_details_from_atomic(atomic_labels, atomic_results):
    """
    Nicely aligns intersection atomic details.
    'Details:' is printed at col 5; after that, atomic details start.
    Each subsequent atomic property is aligned with the first one,
    so all 'Label:'s align, regardless of their length.
    """
    lines = []
    for label in atomic_labels:
        res = atomic_results.get(label)
        if isinstance(res, tuple) and res[0]:
            details_str = res[1]
            if details_str:
                details = details_str.replace('\n', ' ')
                lines.append(f"{label}: {details}")

    if not lines:
        return None

    # First line: no extra space, it's directly after 'Details: '
    first_line = lines[0]
    # All other lines: indent with exactly the length of details_prefix
    continued_lines = [line for line in lines[1:]]
    return "\n".join([first_line] + continued_lines)


def load_oeis_bfile(filename):
    """
    General-purpose loader for OEIS b-files.
    Returns a set of numbers from the given file.
    Each line should have the format: n a(n)
    """
    numbers_list = []
    numbers_set = set()
    try:
        with open(filename) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        val = int(parts[1])
                        numbers_list.append(val)
                        numbers_set.add(val)
    except Exception as e:
        print(f"Error loading OEIS file: {e}")
    return numbers_set, numbers_list


def mobius_and_radical(factors: dict[int, int]) -> tuple[int, int, bool]:
    """
    Return (μ(n), rad(n), squarefree) from a prime factor dict.
    μ(n)=0 if any exponent>1; else (-1)^ω(n).  rad(n)=∏p.
    """
    rad = 1
    squarefree = True
    for p, e in factors.items():
        rad *= p
        if e > 1:
            squarefree = False
    mu = 0 if not squarefree else (-1 if (len(factors) % 2) else 1)
    return mu, rad, squarefree


def multiplicative_persistence_sequence(n: int) -> list[int]:
    """
    Return the multiplicative persistence sequence for |n|.
    Repeatedly replace x with the product of its decimal digits until x < 10.
    Example: 39 → [39, 27, 14, 4] (persistence = 3)
    """
    x = abs(n)
    seq = [x]
    while x >= 10:
        prod = 1
        for ch in str(x):
            prod *= int(ch)
        x = prod
        seq.append(x)
    return seq


def multiplicative_persistence(n: int) -> int:
    """Number of steps in multiplicative_persistence_sequence(n)."""
    return max(0, len(multiplicative_persistence_sequence(n)) - 1)


def natural_sort_key(s):
    # Split string into parts: digits as ints, non-digits as lower-case strings
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def omega_stats(n: int):
    """
    Returns (Ω(n), ω(n)) for integer n.
    Ω(n): number of prime factors (with multiplicity)
    ω(n): number of distinct prime factors
    """
    factors = factorint(abs(n))
    if n == 0 or abs(n) == 1:
        return 0, 0
    omega_big = sum(factors.values())   # Ω(n)
    omega_small = len(factors)          # ω(n)
    return omega_big, omega_small


def parity(n: int):
    """
    Returns the parity (even/odd of a number)
    """
    if n % 2 == 0:
        return "Even"
    return "Odd"


# --- Number-stat helpers (factorization-aware) ---

def repetend_info_base10(n: int, factors: dict[int, int] | None = None):
    """
    Info about decimal expansion of 1/n in base 10.

    Returns:
      ('terminates', k)               if 1/n terminates with exactly k digits
      ('repeats', period, preperiod)  otherwise (period length, preperiod length)

    For n = 2^a * 5^b * m, preperiod = max(a, b); if m=1 it terminates.
    The repetend length is the multiplicative order of 10 modulo m, which divides λ(m).
    """
    if n <= 0:
        return None

    if factors is None:
        factors = factorint(n)

    a = factors.get(2, 0)
    b = factors.get(5, 0)
    preperiod = max(a, b)

    # strip 2s & 5s from n to get m
    m = n // (2**a * 5**b)
    if m == 1:
        return ('terminates', preperiod)

    # compute λ(m) from factorization of m, then find the smallest d | λ(m) with 10^d ≡ 1 (mod m)
    from math import lcm
    fac_m = {p: e for p, e in factors.items() if p not in (2, 5)}
    if not fac_m:
        # should not happen because m>1 and only 2,5 were removed
        return ('terminates', preperiod)

    # λ for each p^e (odd p): (p-1)p^{e-1}
    lam = 1
    for p, e in fac_m.items():
        lam_pa = (p - 1) * (p ** (e - 1))
        lam = lcm(lam, lam_pa)

    fac_lam = factorint(lam)
    for d in _divisors_from_factorization(fac_lam):
        if pow(10, d, m) == 1:
            return ('repeats', d, preperiod)
    return ('repeats', lam, preperiod)


def strip_ansi(s):
    """
    Strip ansi sequences from text, used when printing to a file.
    """
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', s)


def suppress_atomic_components(classes, intersection_label_to_atomic):
    """
    If an intersection label is present, remove its atomic components from the list.
    Do NOT remove the intersection label itself.
    """
    labels = {item['label'] for item in classes}
    atomic_to_remove = set()
    # Only suppress atomic components of intersections that are present
    for inter, atoms in intersection_label_to_atomic.items():
        if inter in labels:
            atomic_to_remove.update(atoms)
    return [
        item for item in classes
        if (item['label'] not in atomic_to_remove)
        or (item['label'] in intersection_label_to_atomic)
    ]


def totient_with_details(n: int, factors: dict[int, int]):
    """
    Return φ(n) and a compact explanation string.
    Example: φ(2^2)=2, φ(3)=2, φ(5)=4 ⇒ 2*2*4=16
    """
    if n == 1:
        return 1, "φ(1)=1"

    parts = []
    phi = 1
    mults = []  # for the "2*2*4" bit

    for p, a in sorted(factors.items()):
        # φ(p^a) = p^(a-1) * (p-1)
        phi_pa = (p - 1) * (p ** (a - 1))
        parts.append(f"φ({p}^{a})={phi_pa}" if a > 1 else f"φ({p})={phi_pa}")
        mults.append(str(phi_pa))
        phi *= phi_pa

    return phi, f"{', '.join(parts)} ⇒ {'*'.join(mults)}={phi}"


def validate_output_setting(output_file: str | None) -> str | None:
    """
    Validate output setting.
    - None / "" => ok (screen only)
    - "." / "./" / trailing "/" => ok (per-number directory mode)
    - path/to/file => must not be in forbidden base names or extensions
    Returns the (possibly normalized) output_file, or raises ValueError.
    """
    FORBIDDEN_FILENAMES = {
        ".gitignore",
        "b002093.txt",
        "b004394.txt",
        "b005114.txt",
        "b104272.txt",
        "LICENSE",
        "requirements.txt",
        # Windows reserved device names (case-insensitive on Windows)
        "con", "prn", "aux", "nul",
        "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
        "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
    }

    FORBIDDEN_EXTENSIONS = {".py", ".md"}

    if not output_file:
        return output_file  # screen only

    # directory/per-number modes are allowed as-is
    if output_file in (".", "./") or output_file.endswith("/"):
        return output_file

    # single file mode: check basename & extension
    basename = os.path.basename(output_file)
    name_no_ext, ext = os.path.splitext(basename)
    ext = ext.lower()

    # On Windows, device names are forbidden regardless of extension
    base_lower = basename.lower()
    name_lower = name_no_ext.lower()
    if base_lower in FORBIDDEN_FILENAMES or name_lower in FORBIDDEN_FILENAMES:
        raise ValueError(f"Forbidden output filename: {basename}")

    if ext in FORBIDDEN_EXTENSIONS:
        raise ValueError(f"Forbidden output file extension: {ext}")

    return output_file
