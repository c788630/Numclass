import re
from dataclasses import dataclass

from numclass.transform import TransformNotApplicable

# ---------------------------------------------------------------------------
# Klingon number parser (tlhIngan Hol) -> int
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


def _tokenize_klingon(s: str) -> list[str]:
    s = _normalize_klingon(s)

    if not s:
        raise KlingonNumberError("Empty Klingon number")

    # Important: this is a full-input parser, not an extractor.
    # Digits/operators/punctuation mean: not a Klingon number.
    if re.search(r"[^A-Za-z'\s]", s):
        raise KlingonNumberError(f"Unknown token/word: {s!r}")

    raw_parts = s.split()
    out: list[str] = []

    for part in raw_parts:
        lower = part.lower()

        if lower in _DIGITS:
            out.append(lower)
            continue

        if lower in _SCALES:
            out.append(lower)
            continue

        for dword in sorted(_DIGITS.keys(), key=len, reverse=True):
            dl = dword.lower()
            if not lower.startswith(dl):
                continue

            rest = lower[len(dl):]
            if rest in _SCALES:
                out.append(dl)
                out.append(rest)
                break
        else:
            raise KlingonNumberError(f"Unknown token/word: {part!r}")

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


def transform(text: str) -> int:
    try:
        return klingon_to_int(text)
    except KlingonNumberError as e:
        msg = str(e)
        if "Unknown token/word" in msg or "Empty Klingon" in msg:
            raise TransformNotApplicable from None
        raise ValueError(msg) from None
