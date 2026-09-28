"""sum / prod / cumsum / cumprod: default accumulator vs. explicit dtype."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_overflow_guard import IntegerOverflowError, checked_array


def test_default_sum_uses_wide_accumulator_like_numpy() -> None:
    # Matches plain NumPy: a.sum() on an int8 array accumulates in the
    # platform default integer type, not int8, so this does not overflow.
    a = checked_array([100, 100, 100], dtype=np.int8)
    assert int(a.sum()) == 300


def test_sum_with_explicit_narrow_dtype_overflows() -> None:
    a = checked_array([100, 100, 100], dtype=np.int8)
    with pytest.raises(IntegerOverflowError) as excinfo:
        a.sum(dtype=np.int8)
    assert excinfo.value.op == "add.reduce"


def test_cumsum_with_explicit_narrow_dtype_reports_first_index() -> None:
    a = checked_array([100, 100, 100], dtype=np.int8)
    with pytest.raises(IntegerOverflowError) as excinfo:
        a.cumsum(dtype=np.int8)
    # 100 (ok) + 100 -> 200 doesn't fit int8: first bad prefix is index 1.
    assert excinfo.value.index == (1,)


def test_prod_with_explicit_narrow_dtype_overflows() -> None:
    a = checked_array([20, 20], dtype=np.int8)
    with pytest.raises(IntegerOverflowError):
        a.prod(dtype=np.int8)


def test_cumprod_with_explicit_narrow_dtype_overflows() -> None:
    # Prefix products 5, 25, 125 all fit int8; 625 (the 4th) does not.
    a = checked_array([5, 5, 5, 5], dtype=np.int8)
    with pytest.raises(IntegerOverflowError) as excinfo:
        a.cumprod(dtype=np.int8)
    assert excinfo.value.index == (3,)


def test_sum_int64_explicit_dtype_overflows() -> None:
    a = checked_array([2**62, 2**62, 2**62], dtype=np.int64)
    with pytest.raises(IntegerOverflowError):
        a.sum(dtype=np.int64)


def test_sum_int64_small_values_do_not_overflow() -> None:
    a = checked_array([1, 2, 3, 4], dtype=np.int64)
    assert int(a.sum(dtype=np.int64)) == 10


def test_sum_unsigned_explicit_dtype_overflows() -> None:
    a = checked_array([200, 200], dtype=np.uint8)
    with pytest.raises(IntegerOverflowError):
        a.sum(dtype=np.uint8)
