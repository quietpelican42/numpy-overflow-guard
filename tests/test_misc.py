"""NEP 50 promotion, error attributes, and general ndarray-like behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_overflow_guard import IntegerOverflowError, checked_array


def test_mixed_int64_uint64_promotes_to_float_and_is_not_guarded() -> None:
    # NumPy 2's NEP 50 promotion has no integer type that covers both
    # ranges, so it silently promotes to float64. That's a NumPy design
    # choice (and a precision-loss risk of its own), not an integer
    # overflow, so this package does not intervene.
    a = checked_array([1], dtype=np.int64)
    b = checked_array([1], dtype=np.uint64)
    result = a + b
    assert result.dtype == np.float64
    assert result.tolist() == [2.0]


def test_small_signed_unsigned_mix_promotes_and_is_checked() -> None:
    # int8 + uint8 -> int16 per NEP 50, which is wide enough that this
    # particular pair can never overflow.
    a = checked_array([127], dtype=np.int8)
    b = checked_array([1], dtype=np.uint8)
    result = a + b
    assert result.dtype == np.int16
    assert result.tolist() == [128]


def test_error_attributes() -> None:
    a = checked_array([2**62, 2**62], dtype=np.int64)
    with pytest.raises(IntegerOverflowError) as excinfo:
        a * 4
    err = excinfo.value
    assert err.op == "multiply"
    assert err.index == (0,)
    assert err.dtype == np.dtype(np.int64)
    assert isinstance(err, FloatingPointError)


def test_indexing_and_slicing_preserved() -> None:
    a = checked_array([1, 2, 3, 4], dtype=np.int32)
    assert a[1] == 2
    assert a[1:3].tolist() == [2, 3]
    assert isinstance(a[1:3], checked_array)


def test_comparison_ops_not_guarded() -> None:
    a = checked_array([1, 2, 3], dtype=np.int8)
    assert (a > 1).tolist() == [False, True, True]


def test_plain_ndarray_arithmetic_with_checked_array_operand_is_guarded() -> None:
    # Overflow checking should trigger even when the *other* operand is a
    # plain ndarray, since NEP 13 dispatches on either side.
    a = np.array([127], dtype=np.int8)
    b = checked_array([1], dtype=np.int8)
    with pytest.raises(IntegerOverflowError):
        a + b


def test_repr_and_str_behave_like_ndarray() -> None:
    a = checked_array([1, 2, 3], dtype=np.int32)
    assert "1" in repr(a) and "2" in repr(a) and "3" in repr(a)
