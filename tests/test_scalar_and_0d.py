"""Regression tests for 0-d / scalar-result overflow checks.

A fully-reduced operation -- a 1-D @ 1-D matmul (vector dot product), a
0-d ``checked_array`` (a scalar wrapped as an array), or a `.sum()` over a
whole array -- can produce a bare Python ``bool``/``int`` or a NumPy scalar
rather than a proper ndarray from the int64/uint64 "exact recompute with
Python ints" fallback path. Neither of those has ``.any()`` or ``.ravel()``,
which ``_ops.first_bad_index`` (and, before it can even get there, the
`np.nonzero()` candidate-selection calls used by multiply/power/left_shift)
assumed unconditionally. This file pins down every path that was found to
be affected.
"""

from __future__ import annotations

import numpy as np
import pytest

from numpy_overflow_guard import IntegerOverflowError, checked_array


# -- the reviewer's original crash: 1-D @ 1-D (vector dot) on int64 --------


def test_1d_matmul_int64_overflow_reported_crash() -> None:
    a = checked_array([9223372036854775807, 9223372036854775807], dtype=np.int64)
    b = checked_array([-1636887089, -6805489360418517580], dtype=np.int64)
    with pytest.raises(IntegerOverflowError) as excinfo:
        a @ b
    assert excinfo.value.index == ()
    assert excinfo.value.op == "matmul"


def test_1d_dot_int64_overflow() -> None:
    a = checked_array([9223372036854775807, 9223372036854775807], dtype=np.int64)
    b = checked_array([-1636887089, -6805489360418517580], dtype=np.int64)
    with pytest.raises(IntegerOverflowError):
        a.dot(b)


def test_1d_matmul_int64_safe_case_does_not_crash_or_raise() -> None:
    a = checked_array([9223372036854775807, 1], dtype=np.int64)
    b = checked_array([0, 1], dtype=np.int64)
    assert int(a @ b) == 1


def test_1d_matmul_uint64_overflow() -> None:
    a = checked_array([np.iinfo(np.uint64).max, np.iinfo(np.uint64).max], dtype=np.uint64)
    b = checked_array([1, 1], dtype=np.uint64)
    with pytest.raises(IntegerOverflowError):
        a @ b


# -- 0-d checked_array (a scalar wrapped as a 0-d array) --------------------


def test_0d_multiply_int64_overflow() -> None:
    a = checked_array(np.array(3_037_000_500, dtype=np.int64))
    b = checked_array(np.array(3_037_000_500, dtype=np.int64))
    with pytest.raises(IntegerOverflowError) as excinfo:
        a * b
    assert excinfo.value.index == ()


def test_0d_multiply_int64_safe() -> None:
    a = checked_array(np.array(2, dtype=np.int64))
    b = checked_array(np.array(3, dtype=np.int64))
    assert int(a * b) == 6


def test_0d_multiply_narrow_dtype_overflow() -> None:
    # Narrow dtypes go through the promote-to-wide-native-dtype path, not
    # the object/nonzero path, but 0-d inputs are worth pinning down too.
    a = checked_array(np.array(100, dtype=np.int8))
    with pytest.raises(IntegerOverflowError):
        a * 100


def test_0d_power_int64_overflow() -> None:
    a = checked_array(np.array(10, dtype=np.int64))
    with pytest.raises(IntegerOverflowError):
        a**30


def test_0d_power_int64_safe() -> None:
    a = checked_array(np.array(2, dtype=np.int64))
    assert int(a**10) == 1024


def test_0d_left_shift_int64_overflow() -> None:
    a = checked_array(np.array(1, dtype=np.int64))
    with pytest.raises(IntegerOverflowError):
        a << 70


def test_0d_left_shift_int64_safe() -> None:
    a = checked_array(np.array(1, dtype=np.int64))
    assert int(a << 40) == 1 << 40


def test_0d_add_and_subtract_do_not_crash() -> None:
    a = checked_array(np.array(np.iinfo(np.int64).max, dtype=np.int64))
    b = checked_array(np.array(1, dtype=np.int64))
    with pytest.raises(IntegerOverflowError):
        a + b
    assert int(checked_array(np.array(5, dtype=np.int64)) - checked_array(np.array(3, dtype=np.int64))) == 2


def test_0d_negative_and_absolute_do_not_crash() -> None:
    a = checked_array(np.array(np.iinfo(np.int64).min, dtype=np.int64))
    with pytest.raises(IntegerOverflowError):
        -a
    with pytest.raises(IntegerOverflowError):
        abs(a)


# -- scalar (bare Python int / 0-d) operand mixed with a real array --------


def test_array_times_python_int_scalar_int64() -> None:
    a = checked_array([3_037_000_500, 1], dtype=np.int64)
    with pytest.raises(IntegerOverflowError) as excinfo:
        a * 3_037_000_500
    assert excinfo.value.index == (0,)


def test_array_times_python_int_scalar_int64_safe() -> None:
    a = checked_array([2, 3], dtype=np.int64)
    assert (a * 4).tolist() == [8, 12]


def test_python_int_times_array_operand_order() -> None:
    # __array_ufunc__ dispatches the same way regardless of operand order.
    a = checked_array([3_037_000_500, 1], dtype=np.int64)
    with pytest.raises(IntegerOverflowError):
        3_037_000_500 * a


def test_full_reduction_sum_does_not_crash_on_single_element() -> None:
    # np.add.reduce on a 1-element object array returns a bare Python int,
    # not a 0-d ndarray -- confirm the reduction path already normalises
    # this correctly (it wraps with np.asarray before comparing).
    a = checked_array([5], dtype=np.int64)
    assert int(a.sum(dtype=np.int64)) == 5


def test_full_reduction_sum_int64_overflow_two_elements() -> None:
    a = checked_array([np.iinfo(np.int64).max, 1], dtype=np.int64)
    with pytest.raises(IntegerOverflowError):
        a.sum(dtype=np.int64)
