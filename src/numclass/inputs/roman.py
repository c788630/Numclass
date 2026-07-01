import re

from numclass.transform import TransformNotApplicable

_ROMAN_TOKEN_RE = re.compile(r"^[MDCLXVI]+$", re.IGNORECASE)
_ROMAN_STRICT_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)

_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def transform(text: str) -> int:
    s = (text or "").strip()
    if not s:
        raise TransformNotApplicable

    if s.lower().startswith("roman:"):
        s = s.split(":", 1)[1].strip()

    if not _ROMAN_TOKEN_RE.match(s):
        raise TransformNotApplicable

    if not _ROMAN_STRICT_RE.match(s):
        raise ValueError(f"Invalid Roman numeral (canonical 1..3999): {text!r}")

    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        v = _VALUES[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v

    return total
