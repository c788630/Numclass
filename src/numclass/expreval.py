import ast
import csv
import io
import math
import operator as op
import random
import re
from dataclasses import dataclass
from decimal import Decimal, getcontext
from importlib.resources import files as pkg_files
from pathlib import Path

from colorama import Fore, Style

try:
    from numclass.transform import TransformNotApplicable, try_transform_to_int
except Exception:
    try_transform_to_int = None
    TransformNotApplicable = None

from numclass.utility import CFG, UserInputError, clear_screen, dec_digits
from numclass.workspace import workspace_dir

# ---- simple number parsing helpers (keep original behavior) ----
_THIN_SPACES = ("\u2009", "\u202F", "\u00A0")  # thin, narrow no-break, no-break
_SEP_CLASS = r"[ ,._\u00A0\u2009\u202F]"      # spaces/commas/dots/underscores & NBSP variants
_GROUPED_RE = re.compile(rf"^[+-]?\d{{1,3}}(?:{_SEP_CLASS}\d{{3}})+$")

# ---- allowed operators (safe subset) ----
_ALLOWED_BINOPS = {
    ast.Add:      op.add,
    ast.Sub:      op.sub,
    ast.Mult:     op.mul,
    ast.FloorDiv: op.floordiv,
    ast.Mod:      op.mod,
    ast.Pow:      op.pow,
    ast.LShift:   op.lshift,
    ast.RShift:   op.rshift,
    ast.BitAnd:   op.and_,
    ast.BitXor:   op.xor,
    ast.BitOr:    op.or_,         # NORMAL bitwise OR restored
    # NOTE: concatenation is handled by a ':' pre-pass, not here
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_MAX_NODES = 256  # sanity guard
_MAX_DIGITS = CFG("BEHAVIOUR.MAX_DIGITS", 100_000)


class _IntExprError(Exception):
    pass


_COMPLEX_INPUT_RE = re.compile(
    r"""
    ^\s*
    [+-]?\d+(?:\.\d+)?      # optional real part
    \s*[+-]\s*
    \d+(?:\.\d+)?[ij]       # imaginary part with i or j
    \s*$
    |
    ^\s*
    [+-]?\d+(?:\.\d+)?[ij]  # pure imaginary: 2i, -3.5j, etc.
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


_SCI_NOTATION_TOKEN = re.compile(
    r"""
    (?<![\w.])          # not immediately after a word char or dot
    ([+\-]?)            # optional sign
    (\d+)               # mantissa (digits)
    [eE]
    ([+\-]?\d+)         # exponent (optional sign + digits)
    (?![\w.])           # not immediately before a word char or dot
    """,
    re.VERBOSE,
)

_SEP_RX = re.compile(r"[\s\-_–—]+", re.UNICODE)


def _read_datafile_text(name: str) -> str:
    """
    Load a data file from workspace data/ first, else packaged numclass.data.
    Returns text or raises UserInputError if not found.
    """
    name = Path(name).name  # sanitize, disallow paths
    up = Path(workspace_dir()) / "data" / name
    if up.is_file():
        return up.read_text(encoding="utf-8")

    try:
        return pkg_files("numclass.data").joinpath(name).read_text(encoding="utf-8")
    except FileNotFoundError as err:
        raise UserInputError(f"Special-input data file not found: data/{name}") from err


def _norm_special_key(s: str) -> str:
    """
    Normalize for exact-match special inputs:
    - lowercase
    - strip
    - remove whitespace and common separators: space, tab, hyphen, underscore, en/em dash
    """
    s = (s or "").strip().lower()
    return _SEP_RX.sub("", s)


@dataclass(frozen=True)
class SpecialRow:
    key: str
    handler: str
    index: str
    description: str
    digits: str
    extra: str


_SPECIAL_CACHE: dict[str, SpecialRow] | None = None
_SPECIAL_CACHE_MTIME: float | None = None


def _load_special_inputs() -> dict[str, SpecialRow]:
    name = "special_inputs.tsv"
    text = _read_datafile_text(name)

    table: dict[str, SpecialRow] = {}
    r = csv.reader(io.StringIO(text), delimiter="\t")

    _EXPECTED_COLS = 6

    for row in r:
        if not row:
            continue
        if str(row[0]).lstrip().startswith("#"):
            continue

        if len(row) < _EXPECTED_COLS:
            raise UserInputError(
                f"Bad row in {name} (expected ≥6 tab-delimited fields, got {len(row)}): {row!r}"
            )

        fields = [str(x).strip() for x in row[:6]]
        sr = SpecialRow(*fields)
        table[_norm_special_key(sr.key)] = sr

    return table


class SpecialInputHandled(Exception):
    """Raised to short-circuit normal integer parsing/evaluation."""
    def __init__(self, output: str):
        super().__init__(output)
        self.output = output


LOG10_2 = Decimal(2).ln() / Decimal(10).ln()


def mersenne_leading_digits(p: int, k: int = 20) -> int:
    # precision: k digits + safety margin
    getcontext().prec = k + 10

    x = Decimal(p) * LOG10_2
    f = x - x.to_integral_value(rounding="ROUND_FLOOR")
    return int((Decimal(10) ** (f + (k - 1))).to_integral_value(rounding="ROUND_FLOOR"))


def mersenne_trailing_digits(p: int, k: int = 20) -> int:
    """
    Last k decimal digits of 2^p - 1, computed with modular exponentiation.
    """
    mod = 10 ** k
    return (pow(2, p, mod) - 1) % mod


def abbr_mersenne_number(p: int, *, k: int = 20) -> str:
    ELLIPSIS = CFG("FORMATTING.ELLIPSIS", "…")
    lead = mersenne_leading_digits(p, k)
    tail = mersenne_trailing_digits(p, k)
    return f"{lead:0{k}d}{ELLIPSIS}{tail:0{k}d}"


def _special_handler(key: str, handler: str, index: str, description: str, digits: str, extra: str) -> str:
    """
    Single special handler taking the 6 arguments from the file.
    Returns a preformatted output string to print, OR raises UserInputError.
    """
    h = (handler or "").strip().lower()

    if h == "egg":
        # if egg, raise a user-facing message
        raise UserInputError(description or "Easter egg.")

    if h in ("mersenne_exact", "mersenne"):
        # index = Mersenne index (e.g. 32), digits = decimal digits
        # exponent p can be extracted from key like "2**756839-1"
        p = ""
        m = re.match(r"^\s*2\*\*(\d+)-1\s*$", (key or "").replace(" ", ""))
        if m:
            p = m.group(1)

        digs = (digits or "").strip()

        lines = []

        if p:
            clear_screen()
            p_int = int(p)
            abbr_number = abbr_mersenne_number(p_int, k=20)  # "first10…last10"
            lines.append(f"{Fore.CYAN + Style.BRIGHT}Number statistics:{Style.RESET_ALL}")
            lines.append(f"  Input:                {key}")
            lines.append(f"  Number:               {Fore.YELLOW + Style.BRIGHT}{abbr_number}{Style.RESET_ALL}")
            lines.append(f"  Digits:               Count={digs}")
            lines.append("  Parity:               Odd")
            lines.append("  Prime:                Yes")
            lines.append("")
            lines.append(f"{Fore.CYAN + Style.BRIGHT}Primes and Prime-related Numbers{Style.RESET_ALL}")
            lines.append("- Mersenne prime: Prime of form 2^p−1 where p is prime.")
            lines.append(f"{Fore.GREEN + Style.BRIGHT}  Details: {description}.{Style.RESET_ALL}")

        return "\n".join(lines)

    # Unknown handler keyword
    raise UserInputError(f"Unknown special handler '{handler}' for key '{key}'.")


def looks_like_complex(s: str) -> bool:
    return bool(_COMPLEX_INPUT_RE.match(s))


def _concat(a: int, b: int) -> int:
    """Decimal concatenation: 12:34 -> 1234. Requires non-negative integers.

    Also enforces BEHAVIOUR.MAX_DIGITS on the resulting concatenation, so
    a:b:c cannot silently exceed the configured digit limit even if each
    segment individually is small enough.
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise _IntExprError("concatenation ':' requires integers")
    if a < 0 or b < 0:
        raise _IntExprError("concatenation ':' requires non-negative integers")

    out = a * (10 ** dec_digits(b)) + b

    # Enforce global digit limit on concatenated result
    if dec_digits(abs(out)) > _MAX_DIGITS:
        raise UserInputError(
            f"number has more than {_MAX_DIGITS} decimal digits. "
            "Increase the limit in the profile or pass a smaller value."
        )

    return out


# ---- top-level ':' splitter ----
def _split_top_level_colon(expr: str) -> list[str]:
    """Split on top-level ':' (not inside parentheses). Whitespace is kept per segment."""
    parts = []
    level = 0
    start = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            level += 1
        elif ch == ')':
            level = max(0, level - 1)
        elif ch == ':' and level == 0:
            parts.append(expr[start:i].strip())
            start = i + 1
    parts.append(expr[start:].strip())
    return [p for p in parts if p != ""]  # ignore empty segments like '1::2'


def _would_exceed_digit_limit_for_pow(base: int, exp: int, limit: int = _MAX_DIGITS) -> bool:
    """
    Cheap lower bound on the decimal digits of base**exp.
    If this lower bound already exceeds `limit`, we reject.

    We use that for any |base| >= 2:
        base**exp >= 2**exp
    so digits(base**exp) >= digits(2**exp).

    digits(2**exp) ~= exp * log10(2) + 1.
    We approximate log10(2) by 30103 / 100000 (> actual), which gives a
    safe lower bound when we compare against `limit`.
    """
    if exp <= 0:
        # exp = 0 → 1 digit, exp < 0 is handled elsewhere
        return False

    b = abs(base)
    if b <= 1:
        # 0**exp (exp>0) or ±1**exp are at most 1 digit
        return False

    # Lower bound: digits(2**exp)
    # Using integer math: floor(exp * 0.30103) + 1 with 0.30103 ≈ 30103/100000.
    digits_lb = 1 + (exp * 30103) // 100000
    return digits_lb > limit


def _eval_int_expr(expr: str) -> int:
    """
    Evaluate a *safe* integer expression.

    Allowed: integers (incl. underscores), parentheses,
             + - * // % **, << >>, & ^ |, unary +/-,
             and the special __fact__(...) wrapper for factorial.

    Disallowed: names, general calls (except __fact__), attributes,
                subscripts, lists, etc.
    Negative exponents are rejected (to avoid floats).

    Additionally:
      - Scientific notation like 1e3, 2E5, -3e10 is rewritten to
        exact integer expressions using powers of 10.
      - BEHAVIOUR.MAX_DIGITS is enforced on literals and on the
        final result of the expression.
    """
    # First, rewrite base-10 scientific notation into exact integer math
    expr = _rewrite_scientific_notation(expr)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        # Just "not a valid integer expression"; the caller will turn this
        # into a generic invalid-input message.
        raise _IntExprError("invalid integer expression") from e

    # Sanity guard on AST size
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise _IntExprError("expression too large")

    def _eval_constant(val) -> int:
        """Handle literal constants (int, small float) with digit-limit enforcement."""

        # --- Plain integers ---
        if isinstance(val, int):
            if dec_digits(abs(val)) > _MAX_DIGITS:
                raise UserInputError(
                    f"number has more than {_MAX_DIGITS} decimal digits. "
                    "Increase the limit in the profile or pass a smaller value."
                )
            return val

        # --- Floats (e.g. 1.5, 0.1) are not allowed in integer expressions ---
        if isinstance(val, float):
            raise _IntExprError(
                "non-integer values are not allowed in integer expressions"
            )

        # Anything else (complex, etc.) is not supported here
        raise _IntExprError("unsupported constant type in integer expression")

    def _eval(node, *, modulus: int | None = None) -> int:
        if isinstance(node, ast.Expression):
            return _eval(node.body, modulus=modulus)

        # Python 3.8+: literals
        if isinstance(node, ast.Constant):
            return _eval_constant(node.value)

        # (Compat for very old Python versions where ast.Num exists)
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return _eval_constant(node.n)

        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](
                _eval(node.operand, modulus=modulus)
            )

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            # --- Power: handle explicitly so we can apply digit limits / modular pow ---
            if op_type is ast.Pow:
                base = _eval(node.left, modulus=modulus)
                exp = _eval(node.right, modulus=modulus)

                if not isinstance(base, int) or not isinstance(exp, int):
                    raise _IntExprError("power '**' requires integer operands")
                if exp < 0:
                    raise UserInputError(
                        "negative exponents are not allowed in integer expressions"
                    )

                # In modular context, use Python's pow with modulus: does not
                # construct base**exp, so no digit-limit check needed here.
                if modulus is not None:
                    return pow(base, exp, modulus)

                # Plain context: enforce profile-driven digit limit
                if _would_exceed_digit_limit_for_pow(base, exp, _MAX_DIGITS):
                    raise UserInputError(
                        f"number has more than {_MAX_DIGITS} decimal digits. "
                        "Increase the limit in the profile or pass a smaller value."
                    )

                return pow(base, exp)

            # --- Modulo: make right-hand side the modulus and evaluate left mod that ---
            if op_type is ast.Mod:
                # Evaluate modulus (right) in normal context
                m = _eval(node.right, modulus=None)
                if not isinstance(m, int):
                    raise _IntExprError("modulus must be an integer")
                if m == 0:
                    raise _IntExprError("modulus by zero is not allowed")
                # Evaluate left in a "modulus-aware" context
                left_val = _eval(node.left, modulus=m)
                return left_val % m

            # --- All other binary operators: normal evaluation ---
            if op_type in _ALLOWED_BINOPS:
                left = _eval(node.left, modulus=modulus)
                right = _eval(node.right, modulus=modulus)
                return _ALLOWED_BINOPS[op_type](left, right)

        # --- Support our synthetic factorial call: __fact__(...) ---
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == _FAKE_FACT:
                if node.keywords:
                    raise _IntExprError("factorial does not take keyword arguments")
                if len(node.args) != 1:
                    raise _IntExprError("factorial takes exactly one argument")
                val = _eval(node.args[0], modulus=modulus)
                if not isinstance(val, int):
                    raise _IntExprError("factorial argument must be an integer")
                if val < 0:
                    raise _IntExprError("factorial requires non-negative integer")
                # We rely on the *final* digit-limit check after evaluation
                # to reject insanely large n! values if they exceed MAX_DIGITS.
                return math.factorial(val)

            raise _IntExprError("function calls are not allowed")

        raise _IntExprError(f"unsupported syntax: {type(node).__name__}")

    # Evaluate the expression root
    value = _eval(tree.body, modulus=None)

    if not isinstance(value, int):
        # Should not happen given our node handling, but keep it defensive
        raise _IntExprError("expression did not evaluate to an integer")

    # Final global digit-limit guard on the result of the whole expression
    if dec_digits(abs(value)) > _MAX_DIGITS:
        raise UserInputError(
            f"number has more than {_MAX_DIGITS} decimal digits. "
            "Increase the limit in the profile or pass a smaller value."
        )

    return value


def _parse_int_literal(text: str) -> int | None:
    """Accepts: 42  -7  1_000_000  0xFF  0b1010  123.456.789  123 456 789
       Rejects: 3.14  1,23  12.34.56  0xG1"""

    if text is None:
        return None

    s = text.strip()

    if not s:
        return None

    for ch in _THIN_SPACES:
        s = s.replace(ch, " ")

    if s.lower().startswith(("0x", "0b", "0o")):
        try:
            return int(s.replace("_", ""), 0)
        except ValueError:
            return None

    if re.fullmatch(r"[+-]?\d[\d_]*", s):
        try:
            return int(s.replace("_", ""))
        except ValueError:
            return None

    if _GROUPED_RE.match(s):
        compact = re.sub(_SEP_CLASS, "", s)
        try:
            return int(compact)
        except ValueError:
            return None

    return None


def _replace_random_operator(expr: str) -> str:
    """
    Replace each '?' occurrence with a random integer.
      - '?'            -> randint(0, 999_999)
      - '?N'           -> randint(0, N)       (N: integer literal, grouped digits ok)
      - '?(expr)'      -> randint(0, eval(expr)) where expr is a safe int expression
                         (no '?' allowed inside expr)
    This runs BEFORE colon splitting and AST evaluation, so the result becomes a literal.
    """
    i, n = 0, len(expr)
    out = []

    def _rand_up_to(bound: int | None) -> str:
        if bound is None:
            bound = 999_999
        if not isinstance(bound, int):
            raise _IntExprError("random bound must be an integer")
        if bound < 0:
            raise _IntExprError("random bound must be ≥ 0")
        return str(random.randint(0, bound))

    while i < n:
        ch = expr[i]
        if ch != '?':
            out.append(ch)
            i += 1
            continue

        # Look ahead for '(...)'
        j = i + 1
        while j < n and expr[j].isspace():
            j += 1

        if j < n and expr[j] == '(':
            # find matching ')', respecting nesting
            level = 0
            k = j
            while k < n:
                if expr[k] == '(':
                    level += 1
                elif expr[k] == ')':
                    level -= 1
                    if level == 0:
                        break
                k += 1
            if k >= n or level != 0:
                raise _IntExprError("unbalanced parentheses after '?'")

            inner = expr[j+1:k]
            if '?' in inner:
                raise _IntExprError("random bound must not contain '?'")
            # Safe-evaluate the bound (no ':' concat here; normal ops are fine)
            bound = _eval_int_expr(inner.strip()) if inner.strip() else None
            out.append(_rand_up_to(bound))
            i = k + 1
            continue

        # Else: try '?N' with an integer literal (allow separators and bases)
        m = re.match(
            r"(0[xbo][0-9A-Fa-f_]+|\d[\d_]*)",   # no leading sign; no spaces allowed
            expr[i+1:]
            )
        if m:
            token = m.group(1)
            bound = _parse_int_literal(token)  # already handles bases/underscores
            if bound is None:
                raise _IntExprError("invalid numeric bound after '?'")
            out.append(_rand_up_to(bound))
            i += 1 + m.end()  # skip '?' + matched literal
            continue

        # Bare '?' (default range)
        out.append(_rand_up_to(None))
        i += 1

    return "".join(out)


_FAKE_FACT = "__fact__"


def _rewrite_factorial(expr: str) -> str:
    """
    Rewrite postfix factorial 'x!' into '__fact__(x)'.

    Handles:
        5!
        (3+2)!
        1:23!      -> 1:__fact__(23)

    Does NOT allow:
        !5         (no prefix factorial)
        3!!        (no double/nested factorial)
        (3!)!      (no factorial of factorial)

    The goal is to let Python's AST handle precedence/grouping,
    while we only transform the postfix '!' into a safe call.
    """
    out: list[str] = []
    n = len(expr)
    pos = 0  # start of the next chunk to copy

    i = 0
    while i < n:
        if expr[i] != '!':
            i += 1
            continue

        # We found a '!' at position i; find the operand to its left.
        j = i - 1
        # Skip spaces just before '!'
        while j >= 0 and expr[j].isspace():
            j -= 1
        if j < 0:
            raise _IntExprError("factorial '!' requires a left operand")

        # Case 1: operand ends in ')': need to find matching '('
        if expr[j] == ')':
            level = 0
            k = j
            while k >= 0:
                if expr[k] == ')':
                    level += 1
                elif expr[k] == '(':
                    level -= 1
                    if level == 0:
                        break
                k -= 1
            if k < 0 or level != 0:
                raise _IntExprError("unbalanced parentheses before '!'")
            operand_start = k

        else:
            # Case 2: operand is a "token" (digits/letters/underscore/dot),
            # e.g.  123,  0xFF,  abc,  1_000.000
            if not (expr[j].isalnum() or expr[j] in "._"):
                # e.g. '3!!' (second '!' sees previous '!' as operand)
                raise _IntExprError("factorial '!' has invalid left operand")

            k = j
            while k >= 0 and (expr[k].isalnum() or expr[k] in "._"):
                k -= 1
            operand_start = k + 1

        # Prevent rewriting inside an already-rewritten chunk:
        if operand_start < pos:
            # This catches patterns like (3!)! or 3!!, which we don't support.
            raise _IntExprError("nested factorial '!' is not supported")

        # Copy everything before the operand (that we haven't copied yet)
        out.append(expr[pos:operand_start])

        # The operand text is expr[operand_start : j+1]
        operand_text = expr[operand_start:j+1]
        out.append(f"{_FAKE_FACT}({operand_text})")

        # Move past the '!' and continue scanning
        pos = i + 1
        i = i + 1

    # Copy any remaining tail after the last '!'
    out.append(expr[pos:])
    return "".join(out)


_SCI_NOTATION = re.compile(r"^([0-9]+)[eE]([+-]?[0-9]+)$")


def _rewrite_scientific_notation(expr: str) -> str:
    """
    Rewrite base-10 scientific notation tokens like '1e3', '2E5', '-3e10'
    into exact integer expressions using powers of 10:

        1e3   -> 10**3
        2e5   -> 2*10**5
        -3e4  -> (-3)*10**4

    Negative exponents are rejected as "not integer"; anything that doesn't
    match the pattern is left unchanged and will be handled by the normal
    evaluator (floats, junk, etc.).
    """

    def repl(m: re.Match) -> str:
        sign, mant, exp_str = m.group(1), m.group(2), m.group(3)

        # Parse exponent
        try:
            exp = int(exp_str)
        except ValueError:
            raise _IntExprError("invalid exponent in scientific notation") from None

        if exp < 0:
            # 1e-3 etc. are not integers
            raise _IntExprError(
                "scientific notation with negative exponent is not an integer"
            )

        # 0eN is always 0
        if int(mant) == 0:
            return "0"

        full_mant = (sign or "") + mant

        # 1eN -> 10**N
        if full_mant == "1":
            return f"10**({exp})"

        # general case: A e B -> (A)*10**(B)
        return f"({full_mant})*10**({exp})"

    return _SCI_NOTATION_TOKEN.sub(repl, expr)


def _eval_specials(expr: str) -> int:
    """
    Support 'a:b:c' as decimal concatenation, '?' as random number,
    and postfix '!' as factorial (rewritten to __fact__(...)).

    Each segment is evaluated with the safe AST evaluator, then
    concatenated left-to-right.
    """
    # First expand randoms, then rewrite factorial syntax
    expr = _replace_random_operator(expr)
    expr = _rewrite_factorial(expr)

    segments = _split_top_level_colon(expr)
    if len(segments) <= 1:
        return _eval_int_expr(expr)
    values: list[int] = []
    for seg in segments:
        v = _eval_int_expr(seg)
        if not isinstance(v, int) or v < 0:
            raise _IntExprError("concatenation ':' requires non-negative integer segments")
        values.append(v)
    out = values[0]
    for v in values[1:]:
        out = _concat(out, v)
    return out


# ---- public entry point ----
def _parse_int_or_expr(s: str) -> int | None:
    # 1) special-input short-circuit (exact match, spaces removed)
    special = _load_special_inputs().get(_norm_special_key(s))
    if special is not None:
        out = _special_handler(
            special.key,
            special.handler,
            special.index,
            special.description,
            special.digits,
            special.extra,
        )
        # eggs raise UserInputError; other specials short-circuit:
        raise SpecialInputHandled(out)

    # 2) try literal parser first (keeps grouped digits/bases)
    n = _parse_int_literal(s)
    if n is not None:
        return n

    # 3) optional transform hook
    if try_transform_to_int is not None:
        try:
            return try_transform_to_int(s)
        except TransformNotApplicable:
            pass

    # 4) Complex numbers
    if looks_like_complex(s):
        raise UserInputError(
            "NumClass does number theory on integers only (ℤ, not ℂ)."
        )

    # 5) try colon-aware expression evaluator
    try:
        return _eval_specials(s)
    except _IntExprError:
        return None
