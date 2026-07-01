from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

import numclass.inputs


class TransformNotApplicable(Exception):
    """Raised when a transform does not apply to the input."""


def get_transforms() -> list[str]:
    return [fn.__module__ for fn in _TRANSFORMS]


def _load_transforms() -> list[Callable[[str], int]]:
    transforms: list[Callable[[str], int]] = []

    for modinfo in pkgutil.iter_modules(numclass.inputs.__path__):
        if modinfo.name.startswith("_"):
            continue

        module = importlib.import_module(f"{numclass.inputs.__name__}.{modinfo.name}")
        fn = getattr(module, "transform", None)

        if callable(fn):
            transforms.append(fn)

    return transforms


_TRANSFORMS = _load_transforms()


def try_transform_to_int(text: str) -> int:
    last_value_error: ValueError | None = None

    for fn in _TRANSFORMS:
        try:
            return fn(text)
        except TransformNotApplicable:
            continue
        except Exception as e:
            if e.__class__.__name__ == "TransformNotApplicable":
                continue
            if isinstance(e, ValueError):
                last_value_error = e
                continue
            raise

    if last_value_error is not None:
        raise last_value_error

    raise TransformNotApplicable
