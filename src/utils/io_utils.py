"""Small, dependency-light helpers for safe input/output handling."""

from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Optional, Union


PathLike = Union[str, Path]


def normalize_id(value: Any) -> Optional[str]:
    """Return a stable string representation of an identifier.

    Numeric values such as ``1`` and ``1.0`` become ``"1"``. String values
    are stripped but otherwise preserved, so a meaningful value such as
    ``"001"`` keeps its leading zeroes.

    Missing numeric values and empty strings return ``None``. Other values are
    converted to strings without attempting to reinterpret their contents.
    """

    if value is None:
        return None

    # Handle strings before numeric conversion: "001" is a valid string ID.
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, Integral):
        return str(int(value))

    if isinstance(value, Real):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return None
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return format(numeric_value, ".15g")

    return str(value).strip() or None


def ensure_directory(path: PathLike) -> Path:
    """Create *path* and its parents if needed, then return it as a ``Path``."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert a value to a finite float, returning ``default`` on failure."""

    if value is None:
        return default

    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    return converted if math.isfinite(converted) else default


__all__ = ["ensure_directory", "normalize_id", "safe_float"]
