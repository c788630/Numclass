# transform.py
# To transform any input into integers for NumClass classification

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


class TransformNotApplicable(Exception):
    """Raised when a transform does not apply to the input."""
    pass


# ---------------------------------
# Transformer registry / dispatcher
# ---------------------------------

def _try_roman(text: str) -> int:
    return roman_to_int(text, strict=True)


def _try_klingon(text: str) -> int:
    try:
        return klingon_to_int(text)
    except KlingonNumberError as e:
        msg = str(e)
        # If it doesn't even look like Klingon, treat as "not applicable"
        if "Unknown token/word" in msg or "Empty Klingon" in msg:
            raise TransformNotApplicable from None
        # Otherwise, it looked Klingon-ish but was invalid
        raise ValueError(msg) from None


_TRANSFORMS: list[Callable[[str], int]] = [
    _try_klingon,
    _try_roman,
]


def try_transform_to_int(text: str) -> int:
    """
    Try all registered transforms. Return the first int produced.
    Raise TransformNotApplicable if none apply.
    Raise ValueError if a transform applies but input is invalid.
    """
    last_value_error: ValueError | None = None

    for fn in _TRANSFORMS:
        try:
            return fn(text)
        except TransformNotApplicable:
            continue
        except ValueError as e:
            # "applies but invalid": keep the message in case nothing else matches
            last_value_error = e

    if last_value_error is not None:
        # Input looked like (say) Roman/Klingon but was invalid => surface error
        raise last_value_error

    raise TransformNotApplicable


# -----------------------------
# Demo: Roman numerals
# -----------------------------

_ROMAN_TOKEN_RE = re.compile(r"^[MDCLXVI]+$", re.IGNORECASE)

_ROMAN_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

# Strict-ish Roman numeral validation pattern for 1..3999
# (common canonical form: no overlines)
_ROMAN_STRICT_RE = re.compile(
    r"^M{0,3}"
    r"(CM|CD|D?C{0,3})"
    r"(XC|XL|L?X{0,3})"
    r"(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)


def roman_to_int(text: str, *, strict: bool = True) -> int:
    s = (text or "").strip()
    if not s:
        raise TransformNotApplicable

    # Allow "roman:XIV" style if you want; optional:
    if s.lower().startswith("roman:"):
        s = s.split(":", 1)[1].strip()

    # If it contains non-roman letters, it's not Roman numerals
    if not _ROMAN_TOKEN_RE.match(s):
        raise TransformNotApplicable

    s = s.upper()

    if strict and not _ROMAN_STRICT_RE.match(s):
        # Canonical range 1..3999
        raise ValueError(f"Invalid Roman numeral (canonical 1..3999): {text!r}")

    total = 0
    prev = 0
    for ch in reversed(s):
        v = _ROMAN_VALUES[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v

    if total <= 0:
        raise ValueError(f"Invalid Roman numeral: {text!r}")

    return total


# Demo: Transforming Klingon language into integers.
# ---------------------------------------------------------------------------
# Supports:
#   0..9: pagh, wa’, cha’, wej, loS, vagh, jav, Soch, chorgh, Hut
#   scales: maH(10), vatlh(100), SaD(1000), netlh(10k), bIp(100k), ’uy’(1M)
#   additive structure: <digit><scale> [<rest smaller>] ... + optional trailing <digit>
# Examples:
#   "loSmaH cha’" -> 42
#   "wa’maH wa’" -> 11
#   "cha’SaD wa’vatlh vagh" -> 2105
#   "javSaD wa’vatlh SochmaH loS" -> 6174
# ---------------------------------------------------------------------------

_DIGITS = {
    "pagh": 0,
    "wa'": 1, "wa’": 1,
    "cha'": 2, "cha’": 2,
    "wej": 3,
    "los": 4, "loS": 4,  # allow both; we'll normalize case anyway
    "vagh": 5,
    "jav": 6,
    "soch": 7, "Soch": 7,
    "chorgh": 8,
    "hut": 9, "Hut": 9,
}

# canonical scales (Klingon is case-sensitive in spelling, but we normalize)
_SCALES = {
    "mah": 10,         # maH
    "vatlh": 100,
    "sad": 1_000,      # SaD
    "netlh": 10_000,
    "bip": 100_000,    # bIp
    "uy'": 1_000_000,  # ’uy’
    "’uy’": 1_000_000,
}


@dataclass(frozen=True)
class KlingonNumberError(ValueError):
    msg: str


def _normalize_klingon(s: str) -> str:
    # - unify apostrophes
    # - keep spaces (we will tokenize)
    # - strip weird punctuation around tokens
    s = s.strip()
    # unify curly apostrophes to plain apostrophe
    s = s.replace("’", "'").replace("‘", "'")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|'")  # simple-ish; we postprocess


def _tokenize_klingon(s: str) -> list[str]:
    s = _normalize_klingon(s)

    # Split on whitespace first, then further split glued forms like "loSmaH"
    raw_parts = s.split()  # avoids empty parts from multiple spaces
    out: list[str] = []

    for raw_part in raw_parts:
        # peel non-letter chars
        cleaned = re.sub(r"[^A-Za-z']+", "", raw_part)
        if not cleaned:
            continue

        lower = cleaned.lower()

        # direct digit word?
        if lower in _DIGITS:
            out.append(lower)
            continue

        # direct scale word?
        if lower in _SCALES:
            out.append(lower)
            continue

        # try split glued "<digit><scale>", e.g. "losmah", "chasad", "wa'vatlh"
        for dword in sorted(_DIGITS, key=len, reverse=True):
            dl = dword.lower()
            if not lower.startswith(dl):
                continue
            rest = lower[len(dl):]
            if rest in _SCALES:
                out.append(dl)
                out.append(rest)
                break
        else:
            # no break => no match
            raise KlingonNumberError(f"Unknown token/word: {cleaned!r}")

    return out


def klingon_to_int(text: str) -> int:
    """
    Parse a Klingon number expression into an integer.

    Rules (practical):
      - "pagh" alone means 0.
      - For each scale (10, 100, 1000, ...), you can have at most one multiplier digit.
      - Expression is additive across descending scales.
      - A trailing digit without scale is allowed (units).
      - No negatives, no fractions.

    Raises KlingonNumberError on invalid input.
    """
    tokens = _tokenize_klingon(text)
    if not tokens:
        raise KlingonNumberError("Empty Klingon number")

    if tokens == ["pagh"]:
        return 0

    # Disallow "pagh" mixed with other stuff (keeps it simple/clean)
    if "pagh" in tokens:
        raise KlingonNumberError("pagh (0) must be used alone")

    # Parse by consuming (digit scale) pairs, then optional trailing digit.
    # We enforce descending scales (so you can't do "maH SaD" nonsense).
    i = 0
    total = 0
    last_scale = float("inf")
    used_scales: set[int] = set()

    def peek() -> str | None:
        return tokens[i] if i < len(tokens) else None

    def take() -> str:
        nonlocal i
        t = tokens[i]
        i += 1
        return t

    while i < len(tokens):
        t = peek()

        # trailing unit digit
        if t in _DIGITS:
            unit = _DIGITS[take()]
            if i != len(tokens):
                # If there's more tokens after a bare digit, it must be a scale immediately.
                # (But glued forms already split into digit+scale.)
                nxt = peek()
                if nxt in _SCALES:
                    # digit + scale pair handled below by rewinding one step
                    i -= 1
                else:
                    raise KlingonNumberError(f"Unexpected digit {t!r} before {nxt!r}")
            else:
                total += unit
                break

        # digit + scale pair
        if i + 1 <= len(tokens) - 1 and tokens[i] in _DIGITS and tokens[i + 1] in _SCALES:
            d = _DIGITS[take()]
            scale_word = take()
            scale = _SCALES[scale_word]

            if scale >= last_scale:
                raise KlingonNumberError("Scales must be in descending order (e.g., SaD then vatlh then maH)")
            if scale in used_scales:
                raise KlingonNumberError(f"Scale used more than once: {scale_word!r}")

            if d == 0:
                raise KlingonNumberError(f"0 cannot multiply a scale ({scale_word!r}); use pagh alone for 0")

            total += d * scale
            used_scales.add(scale)
            last_scale = scale
            continue

        # bare scale (like "maH") without multiplier is not allowed in this strict parser
        if t in _SCALES:
            raise KlingonNumberError(f"Scale {t!r} needs a preceding digit (e.g., wa'maH)")

        raise KlingonNumberError(f"Unexpected token: {t!r}")

    return total
