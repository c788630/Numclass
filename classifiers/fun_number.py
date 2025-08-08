# -----------------------------------------------------------------------------
# Fun numbers test functions
# -----------------------------------------------------------------------------

from decorators import classifier
from typing import Tuple
from user.fun_numbers import FUN_NUMBERS

CATEGORY = "Fun numbers"


@classifier(
    label="Fun number",
    description="Numbers that are famous or iconic in pop culture, computing, sci-fi, internet humor, or memes.",
    oeis=None,
    category=CATEGORY
)
def is_fun_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a fun number, else False.
    Detects iconic numbers from pop culture, computing, sci-fi, internet humor,
    or memes.
    """
    if n in FUN_NUMBERS:
        return True, f"{n} is {FUN_NUMBERS[n]}"
    return False, None