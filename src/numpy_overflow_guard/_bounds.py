"""dtype range helpers shared by the overflow checkers."""

from __future__ import annotations

from typing import Any

import numpy as np

#: Narrow integer dtype -> a native dtype wide enough to hold any product,
#: sum, or shift of the narrow dtype's full range without itself overflowing.
#: int64/uint64 have no wider native counterpart in NumPy, so those keys are
#: absent; callers fall back to Python's arbitrary-precision ``object`` dtype.
_WIDER: dict[str, "np.dtype[Any]"] = {
    np.dtype(np.int8).str: np.dtype(np.int64),
    np.dtype(np.int16).str: np.dtype(np.int64),
    np.dtype(np.int32).str: np.dtype(np.int64),
    np.dtype(np.uint8).str: np.dtype(np.uint64),
    np.dtype(np.uint16).str: np.dtype(np.uint64),
    np.dtype(np.uint32).str: np.dtype(np.uint64),
}

# Conservative (not tight) magnitude bounds under which a product of two
# int64/uint64 values is *guaranteed* not to overflow, used only to choose
# which lanes need an exact recheck -- never to decide overflow itself.
INT64_MUL_SAFE_BOUND = 3_000_000_000  # < floor(sqrt(2**63 - 1))
UINT64_MUL_SAFE_BOUND = 4_000_000_000  # < floor(sqrt(2**64 - 1))


def is_integer_dtype(dtype: "np.dtype[Any]") -> bool:
    return dtype.kind in "iu"


def bounds(dtype: "np.dtype[Any]") -> tuple[int, int]:
    """The inclusive ``(min, max)`` representable by an integer dtype."""
    info = np.iinfo(dtype)
    return int(info.min), int(info.max)


def wider_dtype(dtype: "np.dtype[Any]") -> "np.dtype[Any] | None":
    """A native dtype at least twice as wide, or ``None`` if already 64-bit."""
    return _WIDER.get(dtype.str)
