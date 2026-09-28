"""``.astype()`` narrowing casts."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_overflow_guard import IntegerOverflowError, checked_array


def test_narrowing_cast_overflows() -> None:
    a = checked_array([300], dtype=np.int64)
    with pytest.raises(IntegerOverflowError) as excinfo:
        a.astype(np.int8)
    assert excinfo.value.op == "astype"


def test_narrowing_cast_within_range_is_fine() -> None:
    a = checked_array([100, -100], dtype=np.int64)
    assert a.astype(np.int8).tolist() == [100, -100]


def test_narrowing_cast_boundary_values_are_fine() -> None:
    a = checked_array([127, -128], dtype=np.int64)
    assert a.astype(np.int8).tolist() == [127, -128]


def test_narrowing_cast_one_past_boundary_overflows() -> None:
    a = checked_array([128], dtype=np.int64)
    with pytest.raises(IntegerOverflowError):
        a.astype(np.int8)
    b = checked_array([-129], dtype=np.int64)
    with pytest.raises(IntegerOverflowError):
        b.astype(np.int8)


def test_widening_cast_never_overflows() -> None:
    a = checked_array([127, -128], dtype=np.int8)
    assert a.astype(np.int64).tolist() == [127, -128]


def test_signed_to_unsigned_narrowing_detects_negative_values() -> None:
    a = checked_array([-1], dtype=np.int64)
    with pytest.raises(IntegerOverflowError):
        a.astype(np.uint8)


def test_float_cast_is_not_checked() -> None:
    # astype's overflow check only applies between two integer dtypes.
    a = checked_array([300], dtype=np.int64)
    assert a.astype(np.float64).tolist() == [300.0]
