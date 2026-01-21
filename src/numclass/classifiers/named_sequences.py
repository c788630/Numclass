# -----------------------------------------------------------------------------
#  named_sequences.py
#  Named sequences test functions
# -----------------------------------------------------------------------------

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from numclass.context import NumCtx
from numclass.fmt import abbr_int_fast
from numclass.registry import classifier
from numclass.utility import CFG, check_oeis_bfile, fib_upto, fibonacci_index_if_member, zeckendorf_decomposition
from numclass.workspace import workspace_dir

CATEGORY = "Named Sequences"


@dataclass
class _UlamState:
    seq: list[int]                       # U(1), U(2), ...
    seen: set[int]                       # set of values in seq
    cur_max: int                         # largest candidate tested
    pos: dict[int, int]                  # value -> 0-based index in seq
    repr: dict[int, tuple[int, int]]     # value -> (a, b) with a + b = value
    warmed: bool = False


def _new_ulam_state() -> _UlamState:
    return _UlamState(
        seq=[1, 2],
        seen={1, 2},
        cur_max=2,
        pos={1: 0, 2: 1},
        repr={},
        warmed=False,
    )


ULAM = _new_ulam_state()


def _warm_ulam_from_series(series: list[int]) -> None:
    """
    Initialize ULAM from a full b-file series (U(1)..U(k)).
    Only runs once per process.
    """
    if ULAM.warmed or not series:
        return

    ULAM.seq = list(series)
    ULAM.seen = set(series)
    ULAM.pos = {v: i for i, v in enumerate(series)}
    ULAM.repr.clear()
    ULAM.cur_max = series[-1]

    ULAM.warmed = True


def _ensure_ulam_repr_for(n: int) -> None:
    """
    If n is an Ulam number known in ULAM.seq but has no stored (a, b),
    find its unique representation and cache it in ULAM.repr.
    """
    if n not in ULAM.seen or n in ULAM.repr:
        return

    seq = ULAM.seq
    seen = ULAM.seen
    half = n // 2

    pair: tuple[int, int] | None = None
    count = 0

    for x in seq:
        if x > half:
            break
        y = n - x
        if y != x and y in seen:
            count += 1
            if count > 1:
                # not unique → don't store anything
                pair = None
                break
            pair = (x, y)

    if pair is not None:
        ULAM.repr[n] = pair


def _ensure_ulam_upto_with_budget(target: int, time_budget_s: float) -> bool:
    """
    Advance the Ulam sequence toward `target`, but stop after ~time_budget_s.
    Returns True if we reached `target`, False if we ran out of time first.
    """
    if target <= ULAM.cur_max:
        return True

    start = perf_counter()
    deadline = start + time_budget_s

    seq = ULAM.seq
    seen = ULAM.seen

    while ULAM.cur_max < target:
        if perf_counter() >= deadline:
            return False

        candidate = ULAM.cur_max + 1
        count = 0
        half = candidate // 2
        pair: tuple[int, int] | None = None

        for x in seq:
            if x > half:
                break
            y = candidate - x
            if y != x and y in seen:
                count += 1
                if count > 1:
                    pair = None
                    break
                pair = (x, y)

        if count == 1:
            seq.append(candidate)
            seen.add(candidate)
            ULAM.pos[candidate] = len(seq) - 1
            if pair is not None:
                ULAM.repr[candidate] = pair

        ULAM.cur_max = candidate

    return True


@classifier(
    label="Busy Beaver number",
    description="The most steps a computer program with n states can take before halting.",
    oeis="A060843",
    category=CATEGORY
)
def is_busy_beaver_number(n: int) -> tuple[bool, str]:
    """
    The highest number of calculation steps for a Turing machine with n states.
    The sequence grows faster than any computable function.
    """
    KNOWN = [None, 1, 6, 21, 107, 47176870]  # 1-based indexing
    try:
        index = KNOWN.index(n)
        # Calculate number of Turing machines for index states
        # (standard 2-symbol)
        num_machines = (4 * index + 2) ** (2 * index)
        details = (
            f"Maximum {n} calculation step{'s' if index != 1 else ''} "
            f"for a {index}-state Turing machine "
            f"({num_machines:,} possible machines)"
        )
        return True, details
    except ValueError:
        return False, None


@classifier(
    label="Carol number",
    description="n = (2^k−1)^2 − 2 for some k.",
    oeis="A093112",
    category=CATEGORY
)
def is_carol_number(n: int) -> tuple[bool, str | None]:
    """
    Returns (True, details) if n is a Carol number, else (False, None).

    Uses bit-length heuristics and shift arithmetic:
      n = (2^k−1)^2 − 2 = 2^(2k) − 2^(k+1) − 1
    so bit_length(n) ≈ 2k and we only need to test a few k around that.
    """
    # Small / negative cases
    if n < -1:
        return False, None
    if n == -1:
        # k = 1: (2^1 - 1)^2 - 2 = -1
        return True, f"{n} = (2^1 - 1)^2 - 2 (k = 1)"
    if n < 7:
        # smallest positive Carol number is 7 (k=2)
        return False, None

    # From here: n >= 7
    bl = n.bit_length()
    # For Carol numbers with k >= 2, bit_length(n) ≈ 2k
    approx_k = bl // 2

    # Search only in a tiny window around approx_k
    for delta in (-2, -1, 0, 1, 2):
        k = approx_k + delta
        if k < 1:
            continue

        # Carol(k) = 2^(2k) − 2^(k+1) − 1   (fast via shifts)
        carol = (1 << (2 * k)) - (1 << (k + 1)) - 1

        if carol == n:
            details = f"{n} = (2^{k} - 1)^2 - 2 (k = {k})"
            return True, details

        # Optional tiny optimization: if we are on or above approx_k and
        # the candidate already exceeds n, we can break early.
        if carol > n and delta >= 0:
            break

    return False, None


@classifier(
    label="Cullen number",
    description="Number of the form n·2^n + 1 for some integer n ≥ 1.",
    oeis="A002064",
    category=CATEGORY
)
def is_cullen_number(n: int) -> tuple[bool, str | None]:
    """
    Check if n is a Cullen number: n = k·2^k + 1 for some integer k ≥ 1.

    We solve m = n - 1 = k·2^k by binary search on k, using that
    f(k) = k·2^k is strictly increasing for k ≥ 1.
    """
    if n < 3:
        # Smallest Cullen number is 3 = 1·2^1 + 1
        return False, None

    m = n - 1
    if m <= 0:
        return False, None

    # k·2^k grows very fast. For k ≥ bit_length(m), k·2^k already exceeds m,
    # so k must lie in [1, bit_length(m)).
    bl = m.bit_length()
    lo, hi = 1, bl  # search in [lo, hi)

    while lo < hi:
        k = (lo + hi) // 2
        mk = k << k  # k * 2^k

        if mk == m:
            # Found exact k
            return True, f"{n} = {k}·2^{k} + 1 (Cullen number with k = {k})."

        if mk < m:
            lo = k + 1
        else:
            hi = k

    return False, None


@classifier(
    label="Erdős-Woods number",
    description="n is length of an interval where every number shares a factor with the first or last.",
    oeis="A059756",
    category=CATEGORY,
    limit=331_777
)
def is_erdos_woods_number(k: int, *, abbreviate_output: bool = True) -> tuple[bool, str | None]:
    """
    Simple checker: uses membership + prints a witness interval if embedded.
    Returns (True, details) or (False, None)
    """

    # --- Helpers ---
    def _erdos_woods_toml_path() -> Path:
        """
        Prefer user's workspace data; fall back to packaged default if you ship one.
        """
        ws = workspace_dir() / "data" / "erdos_woods.toml"
        if ws.exists():
            return ws
        return ws  # if no fallback, this will raise later with a clear error

    @lru_cache(maxsize=1)  # load data once
    def _load_erdos_woods() -> tuple[set[int], dict[int, int]]:
        path = _erdos_woods_toml_path()
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        ew = data.get("erdos_woods", {})

        # Coerce
        ew_set = {int(x) for x in ew.get("set", [])}
        starts_raw = ew.get("min_starts", {})
        ew_min_starts = {int(k): int(v) for k, v in starts_raw.items()}
        return ew_set, ew_min_starts

    ERDOS_WOODS_SET, EW_MIN_STARTS = _load_erdos_woods()

    def _witness_pair(n: int) -> tuple[int, int] | None:
        a = EW_MIN_STARTS.get(n)
        return (a, a + n) if a is not None else None

    def _abbr(x: int, head: int = 6, tail: int = 6, threshold: int = 35) -> str:
        s = str(x)
        if len(s) <= threshold:
            return s
        return f"{s[:head]}…{s[-tail:]} ({len(s)} digits)"

    def _interval_str(n: int) -> str | None:
        pair = _witness_pair(n)
        if pair is None:
            return None
        a, b = pair
        if not abbreviate_output:
            return f"[{a}, {b}]"
        return f"[{_abbr(a)}, {_abbr(b)}]"

    # --- Computation ---
    if k < 1:
        return False, "k must be >= 1"

    interval = _interval_str(k)
    if interval is not None:
        return True, f"Witness interval (minimal): {interval}"

    if k in ERDOS_WOODS_SET:
        return True, (
            f"{k} is a known Erdős–Woods number (minimal witness not embedded)."
        )

    return False, None


@classifier(
    label="Factorial number",
    description="n = k! for some integer k ≥ 0 (reverse factorial check).",
    oeis="A000142",
    category=CATEGORY,
)
def is_factorial_number(n: int) -> tuple[bool, str | None]:
    if n < 0:
        return False, None
    if n == 1:
        return True, "1 = 0! = 1!"
    k, steps = reverse_factorial_index(n)
    if k is not None:
        # Pretty division witness: n ÷ 2 ÷ 3 ÷ ... ÷ k = 1
        if steps:
            chain = " ÷ ".join(map(str, steps))
            detail = f"{n} = {k}!  (reverse: {n} ÷ {chain} = 1)"
        else:
            detail = f"{n} = {k}!"
        return True, detail
    return False, None


@classifier(
    label="Fibonacci number",
    description="Term of the Fibonacci sequence (F0=0, F1=1, Fₙ=Fₙ₋₁+Fₙ₋₂).",
    oeis="A000045",
    category=CATEGORY
)
def is_fibonacci_number(n: int) -> tuple[bool, str | None]:
    if n < 0:
        return False, None

    k = fibonacci_index_if_member(n)
    if k is None:
        return False, None

    # Optional: show a capped sequence preview (avoid giant details)
    seq = fib_upto(n)  # already stops at <= n
    details = f"{n} is Fibonacci number F({k}): sequence: {seq}"
    return True, details


@classifier(
    label="Hamming number",
    description="Numbers with no prime factor greater than 5 (also called regular numbers)",
    oeis="A051037",
    category=CATEGORY
)
def is_hamming_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a Hamming number (all prime factors ≤ 5).
    """
    if n < 1:
        return False, None
    original_n = int(n)
    for p in [2, 3, 5]:
        while n % p == 0 and n > 1:
            n //= p
    if n == 1:
        return True, f"{abbr_int_fast(original_n)} has no prime factors greater than 5."
    else:
        return False, None


@classifier(
    label="Keith number",
    description="Appears in its own digit-sequence sum recurrence.",
    oeis="A007629",
    category=CATEGORY,
    limit=10**20-1
)
def is_keith_number(n: int) -> tuple[bool, str]:
    """
    Returns (True, details) if n is a Keith number, else False.
    Details show the digit recurrence sequence leading to n.
    """
    @lru_cache(maxsize=128)
    def keith_test(n):
        if n < 10:
            return False, None  # No single-digit Keith numbers

        seq = [int(d) for d in str(n)]
        k = len(seq)
        steps = list(seq)
        while True:
            next_term = sum(steps[-k:])
            steps.append(next_term)
            if next_term == n:
                details = f"Keith sequence: {', '.join(str(x) for x in steps)}"
                return True, details
            if next_term > n:
                return False, None
    return keith_test(n)


@classifier(
    label="Lucas number",
    description="L₀=2, L₁=1, and Lₙ = Lₙ₋₁ + Lₙ₋₂ for n≥2.",
    oeis="A000032",
    category=CATEGORY,
)
def is_lucas_number(n: int) -> tuple[bool, str | None]:
    if n < 0:
        return False, None

    # Small cases
    if n == 2:
        return True, "2 is Lucas number L(0): sequence: [2]"
    if n == 1:
        return True, "1 is Lucas number L(1): sequence: [2, 1]"
    if n == 0:
        return False, None  # Lucas numbers are 2,1,3,4,7,... so 0 not included

    a, b = 2, 1  # L0, L1
    k = 1
    seq = [2, 1]

    while b < n:
        a, b = b, a + b
        k += 1
        seq.append(b)

    if b == n:
        return True, f"{n} is Lucas number L({k}): sequence: {seq}"
    return False, None


@classifier(
    label="Lucky number",
    description="Number remaining after repeatedly removing every k-th number (k=2, 3, …) from the natural numbers.",
    oeis="A000959",
    category=CATEGORY,
    limit=10**6-1
)
def is_lucky_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a lucky number.
    Details: index if n is lucky
    """
    if n < 1:
        return False, None
    numbers = list(range(1, max(n, 10000) + 1, 2))  # Start with odd numbers
    idx = 1
    # Start sieving from the second element (1-based index in mathematical description)
    while idx < len(numbers):
        step = numbers[idx]
        if step <= 0:
            break
        # Remove every step-th element (1-based counting!)
        # The deletion must always use the current list length and indices.
        del numbers[step-1::step]
        idx += 1
    if n in numbers:
        lucky_idx = numbers.index(n)
        details = f"{n} is Lucky number L({lucky_idx})."
        return True, details
    return False, None


@classifier(
    label="Padovan number",
    description="Defined by P(n)=P(n-2)+P(n-3), starting 1,1,1.",
    oeis="A000931",
    category=CATEGORY
)
def is_padovan_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a Padovan number and provide details.
    The Padovan sequence is defined by P(0) = P(1) = P(2) = 1,
    and P(n) = P(n-2) + P(n-3) for n > 2.
    """
    if n < 0:
        return False, None
    seq = [1, 1, 1]
    idx = 2
    while seq[-1] < n:
        next_val = seq[-2] + seq[-3]
        seq.append(next_val)
        idx += 1
    if seq[-1] == n:
        details = f"{n} is Padovan number P({idx}): sequence: {seq[:idx+1]}"
        return True, details
    return False, None


@classifier(
    label="Pell number",
    description="P(n)=2P(n−1)+P(n−2), P(0)=0,P(1)=1.",
    oeis="A000129",
    category=CATEGORY
)
def is_pell_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a Pell number and provide details.
    The Pell numbers are an integer sequence defined by the recurrence
    P(n) = 2×P(n−1) + P(n−2), with P(0) = 0, P(1) = 1.
    Supports negative numbers: P(-k) = (-1)^k * P(k)
    """
    if n < 0:
        return False, None
    pell = [0, 1]
    abs_n = abs(n)
    # Generate Pell numbers up to |n|
    while abs(pell[-1]) < abs_n:
        pell.append(2 * pell[-1] + pell[-2])

    # Check positive Pell numbers
    if n in pell:
        idx = pell.index(n)
        details = f"{n} is Pell number P({idx}): sequence: {pell[:idx+1]}"
        return True, details

    # Check negative indices: P(-k) = (-1)^k * P(k)
    for k, val in enumerate(pell):
        neg_val = ((-1) ** k) * val
        if n == neg_val:
            details = (
                f"{n} is Pell number P(-{k}) = {neg_val}; "
                f"P(-k) = (-1)^{k} * P({k}) = {neg_val}; "
                f"sequence: {[((-1) ** i) * pell[i] for i in range(k+1)]}"
            )
            return True, details

    return False, None


@classifier(
    label="Taxicab number",
    description="Can be written as a sum of two positive cubes in at least two ways. (Also called a Hardy-Ramanujan number.)",
    oeis="A011541",
    category=CATEGORY,
    limit=100973305
)
def is_taxicab_number(n) -> tuple[bool, str]:
    """
    Checks if n os a Taxicab number
    Details: for n as taxicab number (up to 2, 3, 4, 5 ways, below 100973304).
    """

    # Taxicab numbers, up to taxicab(5), n <= 100973304
    # Format: n : (ways, [ "a³+b³", ... ] )
    TAXICAB_DETAILS = {
        1729: (2, ["1³+12³", "9³+10³"]),
        4104: (2, ["2³+16³", "9³+15³"]),
        13832: (2, ["2³+24³", "18³+20³"]),
        20683: (2, ["10³+27³", "19³+24³"]),
        32832: (2, ["4³+32³", "18³+30³"]),
        39312: (2, ["2³+34³", "15³+33³"]),
        40033: (2, ["9³+34³", "16³+33³"]),
        46683: (3, ["3³+36³", "10³+37³", "27³+30³"]),
        64232: (2, ["17³+39³", "26³+36³"]),
        65728: (2, ["12³+40³", "31³+33³"]),
        110656: (2, ["4³+48³", "18³+46³"]),
        110808: (2, ["27³+41³", "36³+40³"]),
        134379: (2, ["9³+50³", "25³+44³"]),
        149389: (2, ["17³+54³", "32³+53³"]),
        165464: (2, ["23³+55³", "34³+54³"]),
        216027: (2, ["6³+59³", "19³+58³"]),
        216125: (3, ["5³+60³", "17³+58³", "30³+55³"]),
        262656: (2, ["2³+64³", "16³+62³"]),
        314496: (2, ["12³+68³", "32³+68³"]),
        320264: (2, ["9³+69³", "32³+68³"]),
        327763: (2, ["13³+70³", "29³+68³"]),
        373464: (2, ["14³+74³", "44³+62³"]),
        402597: (2, ["17³+77³", "36³+75³"]),
        439101: (2, ["27³+80³", "45³+76³"]),
        515375: (2, ["15³+80³", "44³+77³"]),
        525824: (2, ["24³+80³", "36³+80³"]),
        558441: (2, ["11³+82³", "54³+77³"]),
        593047: (2, ["19³+84³", "60³+77³"]),
        684019: (2, ["30³+89³", "65³+84³"]),
        704977: (2, ["41³+88³", "48³+89³"]),
        805688: (2, ["12³+92³", "72³+80³"]),
        842751: (2, ["39³+92³", "56³+87³"]),
        885248: (2, ["56³+88³", "62³+86³"]),
        886464: (2, ["24³+96³", "60³+88³"]),
        920673: (2, ["33³+94³", "72³+87³"]),
        955016: (2, ["28³+98³", "84³+88³"]),
        984067: (2, ["51³+98³", "66³+97³"]),
        998001: (2, ["99³+0³", "70³+91³"]),
        87539319: (3, ["167³+436³", "228³+423³", "255³+414³"]),
        100973304: (5, ["2³+464³", "228³+454³", "167³+485³", "131³+502³", "119³+508³"]),
    }
    if n in TAXICAB_DETAILS:
        ways, pairs = TAXICAB_DETAILS[n]
        min_order = {2: "smallest number as the sum of two cubes in two ways",
                     3: "smallest number as the sum of two cubes in three ways",
                     4: "smallest number as the sum of two cubes in four ways",
                     5: "smallest number as the sum of two cubes in five ways"}
        label = min_order.get(ways, f"in {ways} ways")
        cubes = ", ".join(pairs)
        details = f"{n} = {cubes} ({label})"
        return True, details
    return False, None


@classifier(
    label="Tribonacci number",
    description="Sum of preceding three numbers in its sequence, starting 0,0,1.",
    oeis="A000073",
    category=CATEGORY
)
def is_tribonacci_number(n: int) -> tuple[bool, str]:
    """
    Check if n is a Tribonacci number, including negative indices.
    T(0)=0, T(1)=0, T(2)=1, T(n)=T(n-1)+T(n-2)+T(n-3) for n>2
    For negative index: T(-n) = -T(-n+1) + T(-n+3)
    """
    if n < 0:
        return False, None
    # Positive indices
    pos = [0, 0, 1]
    idx = 2
    while abs(pos[-1]) < abs(n):
        next_val = pos[-1] + pos[-2] + pos[-3]
        pos.append(next_val)
        idx += 1
    if n in pos:
        found_idx = pos.index(n)
        details = f"{n} is Tribonacci number T({found_idx}): sequence: {pos[:found_idx+1]}"
        return True, details
    # Negative indices
    neg = [0, 0, 1]
    for i in range(3, 40):  # 40 is arbitrary, adjust if you want
        neg_val = -neg[-1] + neg[-3]
        neg.append(neg_val)
        if neg_val == n:
            details = f"{n} is Tribonacci number T(-{i}): negative sequence: {neg}"
            return True, details
    return False, None


@classifier(
    label="Ulam number",
    description="Term in the Ulam sequence (1,2,…): smallest unique sum of two distinct earlier terms.",
    oeis="A002858",
    category=CATEGORY,
)
def is_ulam_number(n: int) -> tuple[bool, str | None]:
    if n < 1:
        return False, None

    # Load b-file (once per process) and check membership
    found, idx_file, series, idx_set = check_oeis_bfile("b002858.txt", n)

    # Warm state from loaded series (only once per session)
    if series:
        _warm_ulam_from_series(series)

    # Time budget from config (seconds)
    timebudget = CFG("CLASSIFIER.ULAM.TIME_BUDGET_S", 0.5)

    # If b-file contains `n`, we are DONE (fast path)
    if found:
        # Prefer the true OEIS index from the file; fall back to list index if needed.
        if idx_file is not None:
            k = idx_file
        elif idx_set is not None:
            k = idx_set + 1  # series index is 0-based, U(1) = series[0]
        else:
            k = None

        _ensure_ulam_repr_for(n)
        if n in ULAM.repr:
            a, b = ULAM.repr[n]
            if k is not None:
                return True, f"{n} is Ulam number U({k}): {a} + {b}."
            return True, f"{n} is an Ulam number: {a} + {b}."
        else:
            if k is not None:
                return True, f"{n} is Ulam number U({k})."
            return True, f"{n} is an Ulam number."

    # Otherwise use time-budgeted generator starting at current ULAM.cur_max
    ok = _ensure_ulam_upto_with_budget(n, time_budget_s=timebudget)

    if not ok and n > ULAM.cur_max:
        limit = ULAM.cur_max
        # Let the caller / framework handle this as a "skipped due to timeout"
        raise TimeoutError(
            f"|n| > {limit} (computed within the {timebudget:.2f}s time budget)"
        )

    # If the generator (possibly partially) reached it:
    if n in ULAM.seen:
        idx = ULAM.pos[n] + 1  # pos is 0-based; U(1) = seq[0]
        _ensure_ulam_repr_for(n)
        if n in ULAM.repr:
            a, b = ULAM.repr[n]
            return True, f"{n} is Ulam number U({idx}): {a} + {b}."
        else:
            return True, f"{n} is Ulam number U({idx})."

    return False, None


def _k2k_equals(m: int) -> int | None:
    """
    Find k>=1 with k*2^k == m, or return None.
    Uses monotone binary search with hi := bit_length(m).
    """
    if m <= 0:
        return None
    lo, hi = 1, m.bit_length()  # k < log2(m) + 1 always holds if m=k*2^k
    while lo <= hi:
        mid = (lo + hi) // 2
        val = mid << mid  # mid * 2^mid
        if val == m:
            return mid
        if val < m:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


@classifier(
    label="Woodall number",
    description="Prime p = k×2^k − 1 for some k ≥ 1",
    oeis="A003261",
    category=CATEGORY,
)
def is_woodall_number(n: int) -> tuple[bool, str | None]:
    if n < 1:
        return False, None
    k = _k2k_equals(n + 1)
    if k is not None:
        return True, f"n + 1 = {n+1} = {k}×2^{k}"
    return False, None


def reverse_factorial_index(n: int) -> tuple[int | None, list[int]]:
    """
    Return (k, steps) where k is such that n = k! (k >= 0), or (None, steps) if not.
    'steps' is the list of divisors we divided by: [2, 3, ..., k].
    """
    if n < 0:
        return None, []
    m = abs(n)
    if m == 1:
        # ambiguous case: 1 = 0! = 1!
        return 1, []  # report 1! by default (you can mention 0! in details)
    if m in (0,):
        return None, []
    k = 2
    steps: list[int] = []
    while m % k == 0:
        steps.append(k)
        m //= k
        k += 1
    if m == 1:
        return k - 1, steps
    return None, steps


@classifier(
    label="2-term Zeckendorf number",
    description="Zeckendorf decomposition uses exactly two Fibonacci terms: n = F_a + F_b with a ≥ b+2.",
    category=CATEGORY,
)
def is_2_term_zeckendorf(n: int) -> tuple[bool, str | None]:
    if n < 0:
        return False, None
    terms, idxs, _bits = zeckendorf_decomposition(n)
    if len(terms) != 2:
        return False, None
    return True, f"{n} = {terms[0]} + {terms[1]} (F{idxs[0]} + F{idxs[1]})"


@classifier(
    label="Sparse Zeckendorf representation",
    description="Zeckendorf decomposition uses few Fibonacci terms (weight ≤ K).",
    category=CATEGORY,
)
def is_sparse_zeckendorf(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n < 0:
        return False, None

    K = int(CFG("CLASSIFIER.ZECKENDORF.SPARSE_MAX_TERMS", 3))
    if K <= 0:
        return False, None  # avoid silly config

    terms, idxs, _bits = zeckendorf_decomposition(n)
    if len(terms) == 0:
        return False, None

    if len(terms) <= K:
        sum_str = " + ".join(map(str, terms))
        idx_str = " + ".join(f"F{j}" for j in idxs)
        return True, f"weight {len(terms)} ≤ {K}: {n} = {sum_str} ({idx_str})"
    return False, None
