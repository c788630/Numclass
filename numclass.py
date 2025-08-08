"""
Number Classifier - Mathematical Classifications and Curiosities

Author:      Marcel M W van Dinteren <m.vandinteren1@chello.nl>
Date:        2025-08-05
Version:     1.0.0

Description:
    Classifies an integer n according to various mathematical properties
    and curiosities. Outputs classification, statistics, and special sequences.

Requirements: See requirements.txt

usage: numclass.py [-h] [--output OUTPUT] [--quiet] [--no-details] [--debug] [number]

Number Classifier - Mathematical Classifications and Curiosities

positional arguments:
  number           Number to classify (no number to run in interactive mode)

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Output file or directory (see user/settings.py for more info)
  --quiet          Suppress screen output (for quiet file output)
  --no-details     Do not show explanation/details for results
  --debug          Debug mode (including timings)

License:
    Code: CC BY-NC 4.0 - © 2024 Marcel M W van Dinteren
          https://creativecommons.org/licenses/by-nc/4.0/
    Data: CC BY-NC 3.0 - From OEIS Foundation Inc.
          https://oeis.org/copyright.html
          - Data from OEIS A004394 (https://oeis.org/A004394/b004394.txt)
          - Data from OEIS A104272 (https://oeis.org/A104272/b104272.txt)
          - Modified: shortened and/or with comments.

"""

__version__ = "1.0.0"

import importlib.util
import sys
import time

try:
    from sympy import (isprime, divisors, totient)
    from sympy.functions.combinatorial.numbers import reduced_totient
except ImportError:
    print("Please install sympy: pip install sympy")
    exit(1)

try:
    from user import settings
except ImportError:
    class _EmptySettings:
        pass
    settings = _EmptySettings()

import os
if os.name != "nt":
    import termios
    import tty
else:
    import msvcrt

from collections import OrderedDict, defaultdict
from colorama import init, Fore, Style
from decorators import TEST_LIMITS
from output_manager import OutputManager
from utility import (clear_screen, digital_root_sequence, natural_sort_key,
                     omega_stats, parity, digit_sum, digit_product,
                     get_terminal_width, get_terminal_height,
                     intersection_details_from_atomic, strip_ansi,
                     suppress_atomic_components, filter_maximal_intersections,
                     format_prime_factors, compute_aliquot_sequence,
                     format_aliquot_sequence, import_classifier_functions)

# Load classifier functions
TEST_FUNCTIONS_ALL = import_classifier_functions("./classifiers")
LABEL_TO_CATEGORY = {}
LABEL_TO_DESCRIPTION = {}
LABEL_TO_OEIS = {}

for fn in list(TEST_FUNCTIONS_ALL.values()):
    if hasattr(fn, 'label'):
        label = fn.label
        TEST_FUNCTIONS_ALL[label] = fn
        LABEL_TO_CATEGORY[label] = getattr(fn, 'category', 'Uncategorized')
        LABEL_TO_DESCRIPTION[label] = getattr(fn, 'description', '')
        LABEL_TO_OEIS[label] = getattr(fn, 'oeis', None)

INTERSECTION_RULES = {}
user_path = os.path.join("user", "intersections.py")
if os.path.exists(user_path):
    spec = importlib.util.spec_from_file_location("user_intersections", user_path)
    user_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_mod)
    INTERSECTION_RULES = getattr(user_mod, "INTERSECTION_RULES", [])

INTERSECTION_LABELS = {rule[1] for rule in INTERSECTION_RULES}
intersection_label_to_atomic = {rule[1]: tuple(rule[0]) for rule in INTERSECTION_RULES}
for label in INTERSECTION_LABELS:
    fn = TEST_FUNCTIONS_ALL.get(label)
    main_label = intersection_label_to_atomic[label][0]
    intersection_cat = LABEL_TO_CATEGORY.get(main_label, "Intersection")
    LABEL_TO_CATEGORY[label] = intersection_cat
    LABEL_TO_DESCRIPTION[label] = f"(Intersection) {', '.join(intersection_label_to_atomic[label])}"
    LABEL_TO_OEIS[label] = getattr(fn, 'oeis', None) if fn else None
    if fn and callable(fn):
        TEST_FUNCTIONS_ALL[label] = fn

# "atomic" means: all primary (non-intersection) labels
atomic_labels = set(LABEL_TO_CATEGORY.keys())

SCREEN_HEADER = (f"{Fore.YELLOW + Style.BRIGHT}Number Classifier - Mathematical "
                 f"Classifications and Curiosities{Style.RESET_ALL}")


def clear_line():
    sys.stdout.write('\r\x1b[2K')
    sys.stdout.flush()


def paginate_output(lines, page_lines, om=None):
    """
    Paginate a list of lines to the terminal height (with [Enter]/q prompt).
    """
    total = len(lines)
    i = 0
    while i < total:
        end = min(i + page_lines, total)
        for line in lines[i:end]:
            om.write(line)
        i = end
        if i < total:
            prompt = "\033[94m[Enter] to continue, 'Q' to quit help...\033[0m"
            om.write(prompt, end='')
            # Wait for key
            key = ''
            if os.name == 'nt':
                key = msvcrt.getch()
                if key == b'\r':
                    key = '\n'
                if isinstance(key, bytes):
                    try:
                        key = key.decode('utf-8')
                    except UnicodeDecodeError:
                        key = ''
            else:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            # Clear continue line
            sys.stdout.write('\r\x1b[2K')
            sys.stdout.flush()
            if key.lower() == 'q':
                break


def get_keypress():
    if os.name == 'nt':
        key = msvcrt.getch()
        if key == b'\r':
            return '\n'
        try:
            return key.decode('utf-8')
        except UnicodeDecodeError:
            return ''
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def dedupe_with_details(classes):
    """
    For each label, if multiple are present, keep the one with details (if available).
    Preserves input order for first occurrence of each label.
    """
    label_to_item = {}
    for item in classes:
        label = item['label']
        # prefer details, or first occurrence
        if label not in label_to_item or (item['details'] and not label_to_item[label]['details']):
            label_to_item[label] = item
    # Preserve the original order for the first occurrence of each label
    seen = set()
    deduped = []
    for item in classes:
        label = item['label']
        if label not in seen:
            deduped.append(label_to_item[label])
            seen.add(label)
    return deduped


def format_details(details, start_col=13):
    """
    Formats the details string so the first line starts at column 'start_col',
    wraps to (terminal width - start_col), and all subsequent lines are
    indented at start_col and wrap to full terminal width with that indent.
    """
    import textwrap

    width = get_terminal_width()
    first_line_width = width - start_col
    indent = " " * start_col

    lines = details.splitlines()
    if not lines:
        return ""

    # First line: wraps after 'Details: ' to (width - start_col)
    wrapper_first = textwrap.TextWrapper(
        width=first_line_width,
        initial_indent="",
        subsequent_indent=""
    )
    # Subsequent lines: full width, but always indented
    wrapper_rest = textwrap.TextWrapper(
        width=width,
        initial_indent=indent,
        subsequent_indent=indent
    )

    # Wrap the first logical line for after 'Details: '
    first, *rest = lines
    first_wrapped = wrapper_first.wrap(first)
    out_lines = []
    if first_wrapped:
        # First line after the label is returned as the first item
        out_lines.append(first_wrapped[0])
        # Any wraps from the first logical line get proper indent
        out_lines.extend(indent + line for line in first_wrapped[1:])

    # Handle any further logical lines, also wrapped/indented
    for logical in rest:
        wrapped = wrapper_rest.wrap(logical)
        out_lines.extend(wrapped)

    return "\n".join(out_lines)


def classify_number(
    n: int,
    debug: bool = False,
    allow_slow_calculations: bool = False,
    om=None
) -> dict:
    """
    Classify n with all enabled classifiers (category and classifier-level),
    skip any disabled by category, and only report 'skipped' for true skips/errors.
    """

    # Get enabled categories and classifier-level disables
    CATEGORIES = getattr(settings, "CATEGORIES", {})
    ENABLED_CATEGORIES = {cat for cat, enabled in CATEGORIES.items() if enabled}
    CLASSIFIERS = getattr(settings, "CLASSIFIERS", {})

    # Only run classifiers in enabled categories
    active_labels = [
        label for label in atomic_labels
        if LABEL_TO_CATEGORY.get(label) in ENABLED_CATEGORIES
    ]

    temp_results = []
    skipped = []
    tested_labels = set()
    atomic_results = {}

    if om:
        om.write("")

    for idx, label in enumerate(active_labels):

        if label in tested_labels:
            continue

        progress = f"\r{Fore.CYAN}Evaluating: {label} (Press Ctrl-C to skip.){Style.RESET_ALL}".ljust(get_terminal_width())

        if om:
            om.write_screen(progress, end="")

        # SKIP IF DISABLED IN CLASSIFIERS (category disables already handled above)
        lbl = label.upper().replace(" ", "_")
        if not CLASSIFIERS.get(lbl, True):
            if debug:
                om.write(f"Skipping {label} (disabled in settings)")
            skipped.append(f"{label}: disabled in settings")
            continue

        fn = TEST_FUNCTIONS_ALL.get(label)
        if not fn:
            continue

        lim = TEST_LIMITS.get(fn.__name__) if 'TEST_LIMITS' in globals() else None
        if not allow_slow_calculations and lim is not None and n > lim:
            if om and debug:
                om.write(f"Skipping {label}: n > {lim}")
            skipped.append(f"{label}: n > {lim}")
            continue

        if debug:
            start = time.perf_counter()
        try:
            result = fn(n)
        except KeyboardInterrupt:
            skipped.append(f"{label}: aborted by user (Ctrl-C)")
            continue
        except Exception as e:
            if om and debug:
                duration = time.perf_counter() - start
                om.write(f"{duration:.4f}s")
            skipped.append(f"{label}: error {e}")
            continue

        if debug:
            duration = time.perf_counter() - start
            if om and duration > 0.5:
                om.write(f"{Fore.RED + Style.BRIGHT}{duration:.4f}s{Style.RESET_ALL}")
            elif om:
                om.write(f"{duration:.4f}s")

        if isinstance(result, tuple):
            matched, details = result
            if matched:
                temp_results.append({'label': label, 'details': details})
                atomic_results[label] = (True, details)
                tested_labels.add(label)
        elif result is True:
            temp_results.append({'label': label, 'details': None})
            atomic_results[label] = (True, None)
            tested_labels.add(label)
        elif result is None:
            skipped.append(f"{label}: no result")
        else:
            atomic_results[label] = (False, None)

    # ---- Clear the progress line at the end ----
    if om:
        om.write_screen("\r" + " " * get_terminal_width() + "\r", end="")
    if om and debug:
        om.write("")

    # --- Intersections & grouping (unchanged) ---
    found_labels = set(item['label'] for item in temp_results)

    while True:
        added = False
        sorted_rules = sorted(INTERSECTION_RULES, key=lambda rule: -len(rule[0]))
        for base_labels, intersection_label in sorted_rules:
            if all(lbl in found_labels for lbl in base_labels):
                if intersection_label not in found_labels:
                    inherited_details = intersection_details_from_atomic(base_labels, atomic_results)
                    temp_results.append({'label': intersection_label, 'details': inherited_details})
                    found_labels.add(intersection_label)
                    added = True
        if not added:
            break

    classes = suppress_atomic_components(temp_results, intersection_label_to_atomic)
    classes = filter_maximal_intersections(classes, intersection_label_to_atomic)
    # Dedupe classes so each label appears only once, preferring with details
    classes = dedupe_with_details(classes)

    # Only group into enabled categories, in alphabetical order
    enabled_category_list = sorted(ENABLED_CATEGORIES, key=str.lower)
    grouped_results = OrderedDict((cat, []) for cat in enabled_category_list)

    for item in classes:
        lbl = item['label']
        cat = LABEL_TO_CATEGORY.get(lbl)
        if cat in ENABLED_CATEGORIES:
            grouped_results[cat].append({'label': lbl, 'details': item.get('details')})
    return {'value': n, 'classes': classes, 'grouped': grouped_results, 'skipped': skipped}


def print_statistics(n: int, show_details: bool = True, om=None):
    """
    Pretty print all statistics for n.
    output_manager: if None, prints to screen.
    """
    ALIGN_WIDTH = 22

    # -- Number statistics
    om.write(f"Number: {Fore.YELLOW + Style.BRIGHT}{n}{Style.RESET_ALL}")
    om.write(f"\n{Fore.CYAN + Style.BRIGHT}Number statistics:{Style.RESET_ALL}")

    om.write(f"  {'Parity:':<{ALIGN_WIDTH}}{parity(n)}")
    prime_status = "Yes" if isprime(n) else "No"
    om.write(f"{'  Prime:':<24}{prime_status}")
    om.write(f"  {'Number of digits:':<{ALIGN_WIDTH}}{len(str(abs(n)))}")
    om.write(f"  {'Sum of digits:':<{ALIGN_WIDTH}}{digit_sum(n)}")
    om.write(f"  {'Product of digits:':<{ALIGN_WIDTH}}{digit_product(n)}")
    dr_seq = digital_root_sequence(n)
    dr_seq_str = " → ".join(str(x) for x in dr_seq)
    om.write(f"  {'Digital root of |n|:':<{ALIGN_WIDTH}}{dr_seq[-1]} (sequence: {dr_seq_str})")
    if n != 0 and not isprime(n):
        om.write(f"  {'Prime factorization:':<{ALIGN_WIDTH}}{format_prime_factors(n)}")
    if n > 1:
        big_omega, little_omega = omega_stats(n)
        label = "Prime factor counts:"
        om.write(f"  {label:<{ALIGN_WIDTH}}Ω(n) = {big_omega}, ω(n) = {little_omega}")
    if n > 0:
        label = "Euler's totient φ(n):"
        om.write(f"  {label:<{ALIGN_WIDTH}}{totient(n)}")
        label = "Carmichael λ(n):"
        om.write(f"  {label:<{ALIGN_WIDTH}}{reduced_totient(n)}")

    # -- Divisor statistics
    if n != 0:
        om.write(f"\n{Fore.CYAN + Style.BRIGHT}Divisor statistics of |n|:{Style.RESET_ALL}")
        divs = sorted(divisors(n))
        num_divs = len(divs)
        sum_divs = sum(divs)
        aliquot = sum_divs - n
        max_divs = getattr(settings, "DIVISORLIST_LIMIT", 100)
        aliquot_seq = getattr(settings, "ALIQUOT_SEQUENCE", True)
        show_div = getattr(settings, "SHOW_DIVISORS", True)
        s = ", ".join(str(d) for d in divs)
        s_wrapped = format_details(s, start_col=24)
        om.write(f"  {'Number of divisors:':<22}{num_divs}")
        if show_div and ((max_divs is None) or (num_divs <= max_divs)):
            om.write(f"  {'Divisors:':<22}{s_wrapped.strip()}")
        om.write(f"  {'Sum of divisors:':<22}{sum_divs}")
        if n > 0 and aliquot_seq:
            om.write(f"  {'Aliquot sum:':<22}{aliquot}")

            MAX_ALIQUOT_STEPS = getattr(settings, "MAX_ALIQUOT_STEPS", 50)
            seq, highlight_idx, aborted, skipped = compute_aliquot_sequence(n, max_steps=MAX_ALIQUOT_STEPS, om=om)

            aliquot_wrapped, status_lines, seq_len = format_aliquot_sequence(
                seq, highlight_idx, aborted, skipped, max_steps=MAX_ALIQUOT_STEPS)

            if aliquot_wrapped:
                om.write("  Aliquot sequence:     " + aliquot_wrapped)
            for line in status_lines:
                om.write(f"{' ' * 24}{line}")
            if seq_len != 0 and seq_len < MAX_ALIQUOT_STEPS:
                om.write(f"  Sequence length:      {seq_len}")


def print_classifications(n: int, results: dict, show_details: bool = True, om=None):

    grouped = results.get('grouped')

    # Fetch enabled categories from settings (preserving order or alpha)
    enabled_categories = [
        cat for cat, enabled in getattr(settings, "CATEGORIES", {}).items() if enabled
    ]
    if not enabled_categories:
        enabled_categories = sorted(grouped.keys())

    first = True
    for cat in sorted(enabled_categories, key=str.lower):
        items = grouped.get(cat, [])
        if not items:
            continue
        if not first:
            om.write()
        om.write(f"{Fore.CYAN + Style.BRIGHT}{cat}:{Style.RESET_ALL}")
        first = False
        for item in sorted(items, key=lambda x: natural_sort_key(x['label'])):
            lbl = item['label']
            if lbl == "Prime number":
                continue   # <--- Hide common prime number test (only needed for intersections)
            desc = LABEL_TO_DESCRIPTION.get(lbl, "")
            om.write(f"{Fore.WHITE}  - {lbl}:{Style.RESET_ALL} {desc}")
            if show_details and item.get('details'):
                om.write(f"{Fore.GREEN + Style.BRIGHT}    Details: {format_details(item['details'])}{Style.RESET_ALL}")

    # Skipped (as before)
    def skip_sort_key(skip_msg):
        return skip_msg.split(":")[0] if ":" in skip_msg else skip_msg

    if results.get('skipped'):
        om.write(f"\n{Fore.RED + Style.BRIGHT}Skipped classifications due to constraints or settings:{Style.RESET_ALL}")
        for lbl in sorted(results['skipped'], key=skip_sort_key):
            om.write(f"{Fore.RED + Style.BRIGHT}  - {lbl}{Style.RESET_ALL}")


def get_cat_to_labels_and_total():
    """
    Returns (cat_to_labels: dict, total: int), skipping intersections.
    """
    cat_to_labels = defaultdict(list)
    for label, cat in LABEL_TO_CATEGORY.items():
        if cat == "Intersection":
            continue
        cat_to_labels[cat].append(label)
    total = sum(len(v) for v in cat_to_labels.values())
    return cat_to_labels, total


def show_classifier_list(om=None):
    """
    Display all classifier categories/labels with descriptions, paginated.
    """

    init(autoreset=True)  # Enables color on Windows

    # Build help text lines
    def build_help_lines():
        cat_to_labels, total = get_cat_to_labels_and_total()
        
        lines = []
        lines.append(f"\n{Fore.YELLOW}{Style.BRIGHT}{total} classifiers available, listed by category:{Style.RESET_ALL}\n")
        
        for cat in sorted(cat_to_labels):
            lines.append(f"{Fore.CYAN}{Style.BRIGHT}{cat}:{Style.RESET_ALL}")
            for label in sorted(cat_to_labels[cat], key=natural_sort_key):
                desc = LABEL_TO_DESCRIPTION.get(label, "")
                lines.append(f"  - {Style.BRIGHT}{label}{Style.RESET_ALL}: {desc}")
            lines.append("")  # Blank line between categories
        return lines

    term_height = get_terminal_height()
    help_lines = build_help_lines()
    paginate_output(help_lines, term_height - 2, om)


def show_intro_help(om=None):
    """
    Show introduction/help for numclass: features, usage, and menu options.
    """
    init(autoreset=True)

    cat_to_labels, total = get_cat_to_labels_and_total()

    intro_lines = [
        "",
        f"{Fore.GREEN}Welcome to numclass v{__version__}{Style.RESET_ALL}",
        f"{'-'*79}",
        f"{Fore.LIGHTWHITE_EX}Explore the fascinating world of integer classifications.{Style.RESET_ALL}",
        "",
        f"Discover the hidden mathematical properties of any integer — from the most",
        f"famous sequences to delightful oddities.",
        "",
        f" • Find out if a number is Prime, Perfect, Abundant, Deficient, ... for classic math fans.",
        " • Amicable, Keith, Cake, Untouchable, ... for the curious.",
        f" • Enjoy playful categories: Fun number, Cyclops number, Repdigit, Evil, ",
        "   and mmany more from pop culture, computing, science fiction, internet lore, and memes.",
        "",
        f"{Fore.YELLOW}Features:{Style.RESET_ALL}",
        f" • {total} classifications available.",
        f" • Shows detailed explanations, decompositions, and references.",
        f" • Handy links to the On-Line Encyclopedia of Integer Sequences (OEIS).",
        "",
        f"{Fore.MAGENTA + Style.BRIGHT}Usage:{Style.RESET_ALL}",
        f" • Enter an integer and press enter, numclass shows number and divisor ",
        f"   statistics and all its classifications.",
        f" • Commas are allowed as thousand separators. Press Ctrl-C to skip long calculations.",
        " • You can customize your experience in user/settings.py.",
        "",
        f"{Fore.CYAN}Tips:{Style.RESET_ALL}",
        f" • For command-line options, run: python numclass.py --help",
        f" • numclass adapts to your terminal window size, try resizing for the best view!",
        "",
        f"{Fore.GREEN}Next steps:{Style.RESET_ALL}",
        "",
    ]
    help_lines = [
        f"  {Style.BRIGHT}L{Style.RESET_ALL} - List all classifications by category",
        f"  {Style.BRIGHT}R{Style.RESET_ALL} - Show OEIS references",
        f"  {Style.BRIGHT}E{Style.RESET_ALL} - Show some example numbers you can try",
        f"  {Style.BRIGHT}Q{Style.RESET_ALL} - Quit help and start classifying integers",
        "",
    ]

    clear_screen()
    om.write(SCREEN_HEADER)
    # Show intro
    for line in intro_lines:
        om.write(line)

    # Wait for a menu command
    while True:
        for line in help_lines:
            om.write(line)
        om.write(f"{Fore.LIGHTBLUE_EX}Enter L/R/E/Q : {Style.RESET_ALL}", end="")
        key = get_keypress().strip().lower()
        clear_line()
        if key == 'l':
            show_classifier_list(om)
        elif key == 'r':
            show_oeis_references(om)
        elif key == 'e':
            show_example_inputs(om)
        elif key == 'q' or key == '':
            break
        else:
            om.write("Invalid choice. Please type L, R, E, Q.\n")
            continue
    clear_screen()
    print(SCREEN_HEADER)


def show_oeis_references(om=None):
    """
    Display all classifiers with OEIS references (paginated), URLs aligned.
    """
    from collections import defaultdict

    # Collect labels by OEIS sequence
    oeis_to_labels = defaultdict(list)
    for label, oeis in LABEL_TO_OEIS.items():
        if oeis:
            oeis_to_labels[oeis].append(label)

    lines = [
        f"{Fore.YELLOW}Classifiers with OEIS references:{Style.RESET_ALL}",
        ""
    ]

    # First, build the prefix strings to compute max width
    prefix_label_list = []
    for oeis in sorted(oeis_to_labels, key=lambda s: (s or "")):
        labels = ", ".join(sorted(oeis_to_labels[oeis], key=str.lower))
        prefix = f"  {Fore.GREEN}{oeis}{Style.RESET_ALL}: {labels}"
        prefix_label_list.append((prefix, oeis))

    # Find max prefix width (without color codes for alignment)
    max_prefix_width = max(len(strip_ansi(prefix)) for prefix, _ in prefix_label_list)

    # Add padded lines
    for (prefix, oeis) in prefix_label_list:
        url = f"https://oeis.org/{oeis}"
        pad = max_prefix_width - len(strip_ansi(prefix))
        lines.append(f"{prefix}{' ' * pad}{Fore.LIGHTBLACK_EX} {url}{Style.RESET_ALL}")

    lines.append("")
    paginate_output(lines, get_terminal_height() - 2, om=om)


def show_example_inputs(om=None):
    """
    Display some fun/example numbers to try, with short explanations.
    """
    examples = [
        "",
        "42          (Sum of Three cubes, Highly abundant, fun number, ...)",
        "89          (Disarium, Fibonacci, Palindromic prime, Thick prime, ...)",
        "153         (Narcissistic, Octal interpretable, Harshad triangular, ...)",
        "163         (Lucky Euler number",
        "276         (Aliquot open sequence, Erdős-Woods, Fermat Pseudoprime, ...)",
        "561         (Carmichael, Fermat en Euler-Jacobi pseudoprimes, Hexagonal...)",
        "796         (Boring number... say what?, ...)",
        "1089        (digit-reversal constant, perfect square, sum of palindromes, ...)",
        "1260        (Vampire, Super abundant, Pronic, Highly composite, ...)",
        "1729        (Ramanujan Taxicab number, Sphenic, ...)",
        "2357        (Smarandache Wellin, Chen prime, Proth prime, Sexy prime, ...)",
        "2856        (Large repeating Aliquot sequence, Untouchable, ...)",
        "6174        (Kaprekar constant, Sum of Palindromes, Sum of squares and cubes, ...)",
        "8128        (Perfect, Happy, Semiprime, Squarefree, ...)",
        "47176870    (Busy Beaver, ...)",
        "169808691   (Strobogrammatic, Cyclops, Self, Lychrel candidate, ...)",
        "193764528   (Pandigital, Odious, ...)",
        "459818240   (Triperfect, Practical, Evil, Semiperfect, ...)",
        "74596893730427      (Keith prime, Gaussian prime, Isolated prime, ... )",
        "52631578947368421   (Cyclic, ...)",
        "192137918101841817  (Motzkin, ...)",
        "1000111100101100100010011111001000101  (Binary interpretable, Deficient, ...)",
        "6086555670238378989670371734243169622657830773351885970528324860512791691264 (Sublime)",

    ]
    om.write("\nExample numbers to try:")
    for ex in examples:
        om.write(f"  {ex}")
    om.write("")


def cli_main():
    import argparse

    parser = argparse.ArgumentParser(description="Number Classifier - Mathematical Classifications and Curiosities")
    parser.add_argument("number", nargs="?", type=int, help="Number to classify (no number to run in interactive mode)")
    parser.add_argument("--output", default=None, help="Output file or directory (see user/settings.py for more info)")
    parser.add_argument("--quiet", action="store_true", help="Suppress screen output (for quiet file output)")
    parser.add_argument("--no-details", action="store_true", help="Do not show explanation/details for results")
    parser.add_argument("--debug", action="store_true", help="Debug mode (including timings)")

    args = parser.parse_args()

    def make_output_manager(n=None):
        return OutputManager(output_file=args.output or settings.OUTPUT_FILE, quiet=args.quiet, number=n)

    # CLI (number provided)
    if args.number is not None:
        om = make_output_manager(args.number)
        n = args.number
        try:
            print_statistics(n, show_details=not args.no_details, om=om)
            results = classify_number(n, debug=args.debug, om=om)
            print_classifications(n, results, show_details=not args.no_details, om=om)
        finally:
            om.close()
        return

    # Interactive mode
    clear_screen()
    print(SCREEN_HEADER)

    while True:
        try:
            user_input = input("\nEnter an integer (H=Help, Q=Quit): ").strip().lower()

            if user_input == 'q' or user_input == '':
                break

            if user_input == 'h':
                om = OutputManager(output_file=None, quiet=False, number="help")
                show_intro_help(om)
                continue

            try:
                n = int(user_input.replace(",", ""))
            except ValueError:
                print("Invalid input. Please enter a valid integer or 'H' for help.")
                continue

            om = make_output_manager(n)
            try:
                clear_screen()
                print_statistics(n, show_details=not args.no_details, om=om)
                results = classify_number(n, debug=args.debug, om=om)
                print_classifications(n, results, show_details=not args.no_details, om=om)
            finally:
                if om:
                    om.close()

        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"Fatal error: {e}", file=sys.stderr)


if __name__ == "__main__":
    cli_main()
