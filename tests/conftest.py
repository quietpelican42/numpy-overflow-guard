from __future__ import annotations

from typing import Any

import numpy as np

SIGNED_DTYPES = [np.int8, np.int16, np.int32, np.int64]
UNSIGNED_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
ALL_INT_DTYPES = SIGNED_DTYPES + UNSIGNED_DTYPES


def fits(value: int, dtype: "np.dtype[Any]") -> bool:
    info = np.iinfo(dtype)
    return bool(info.min <= value <= info.max)
