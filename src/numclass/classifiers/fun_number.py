# -----------------------------------------------------------------------------
# Fun numbers test functions
# -----------------------------------------------------------------------------

from __future__ import annotations

from numclass.dataio import load_fun_numbers
from numclass.registry import classifier

CATEGORY = "Fun numbers"

_FUN_MAP = load_fun_numbers()


@classifier(
    label="Fun number",
    description=("Numbers that are famous or iconic in pop culture, "
                 "computing, sci-fi, internet humor, or memes."),
    category=CATEGORY,
)
def is_fun_number(n: int, ctx=None):
    # Use column 2 (description) from fun_numbers.tsv as Details
    desc = _FUN_MAP.get(int(n))
    if desc:
        return True, desc
    return False, None
