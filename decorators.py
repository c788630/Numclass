# -----------------------------------------------------------------------------
#  DECORATORS
# -----------------------------------------------------------------------------

import sys
from functools import wraps

TEST_LIMITS = {}

def classifier(label=None, description=None, oeis=None, category=None):
    """
    Decorator for number classifier functions.
    If category is not given, uses the CATEGORY variable from the module.
    """
    def wrapper(fn):
        mod = sys.modules[fn.__module__]
        fn.category = category or getattr(mod, "CATEGORY", "Uncategorized")
        fn.label = label or fn.__name__
        fn.description = description or ""
        fn.oeis = oeis
        return fn
    return wrapper


def limited_to(max_n: int):
    """
    Decorator factory to limit a function to n <= max_n.
    """
    def decorator(fn):
        TEST_LIMITS[fn.__name__] = max_n

        @wraps(fn)
        def wrapper(n, *args, **kwargs):
            return fn(n, *args, **kwargs)
        return wrapper

    return decorator
