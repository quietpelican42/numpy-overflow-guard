"""Property tests: every guarded op raises iff the exact Python-int result
does not fit the operand dtype, across every integer dtype NumPy has."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from conftest import ALL_INT_DTYPES, fits
from numpy_overflow_guard import IntegerOverflowError, checked_array

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

_SETTINGS = settings(max_examples=60, deadline=None)


def _int_arrays(dtype: type, n: int = 6) -> st.SearchStrategy[Any]:
    info = np.iinfo(dtype)
    return arrays(dtype=dtype, shape=n, elements=st.integers(min_value=int(info.min), max_value=int(info.max)))


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_add_exact(dtype: type, data: st.DataObject) -> None:
    a = data.draw(_int_arrays(dtype))
    b = data.draw(_int_arrays(dtype))
    exact = [int(x) + int(y) for x, y in zip(a.tolist(), b.tolist())]
    expect_overflow = any(not fits(v, np.dtype(dtype)) for v in exact)
    ca, cb = checked_array(a), checked_array(b)
    if expect_overflow:
        with pytest.raises(IntegerOverflowError):
            ca + cb
    else:
        result = ca + cb
        assert result.tolist() == (a + b).tolist()


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_subtract_exact(dtype: type, data: st.DataObject) -> None:
    a = data.draw(_int_arrays(dtype))
    b = data.draw(_int_arrays(dtype))
    exact = [int(x) - int(y) for x, y in zip(a.tolist(), b.tolist())]
    expect_overflow = any(not fits(v, np.dtype(dtype)) for v in exact)
    ca, cb = checked_array(a), checked_array(b)
    if expect_overflow:
        with pytest.raises(IntegerOverflowError):
            ca - cb
    else:
        result = ca - cb
        assert result.tolist() == (a - b).tolist()


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_multiply_exact(dtype: type, data: st.DataObject) -> None:
    a = data.draw(_int_arrays(dtype))
    b = data.draw(_int_arrays(dtype))
    exact = [int(x) * int(y) for x, y in zip(a.tolist(), b.tolist())]
    expect_overflow = any(not fits(v, np.dtype(dtype)) for v in exact)
    ca, cb = checked_array(a), checked_array(b)
    if expect_overflow:
        with pytest.raises(IntegerOverflowError):
            ca * cb
    else:
        result = ca * cb
        assert result.tolist() == (a * b).tolist()


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_negative_exact(dtype: type, data: st.DataObject) -> None:
    a = data.draw(_int_arrays(dtype))
    exact = [-int(x) for x in a.tolist()]
    expect_overflow = any(not fits(v, np.dtype(dtype)) for v in exact)
    ca = checked_array(a)
    if expect_overflow:
        with pytest.raises(IntegerOverflowError):
            -ca
    else:
        assert (-ca).tolist() == (-a).tolist()


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_absolute_exact(dtype: type, data: st.DataObject) -> None:
    a = data.draw(_int_arrays(dtype))
    exact = [abs(int(x)) for x in a.tolist()]
    expect_overflow = any(not fits(v, np.dtype(dtype)) for v in exact)
    ca = checked_array(a)
    if expect_overflow:
        with pytest.raises(IntegerOverflowError):
            abs(ca)
    else:
        assert abs(ca).tolist() == np.abs(a).tolist()


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_power_exact(dtype: type, data: st.DataObject) -> None:
    info = np.iinfo(dtype)
    a = data.draw(
        arrays(dtype=dtype, shape=5, elements=st.integers(min_value=int(info.min), max_value=int(info.max)))
    )
    exponent = data.draw(st.integers(min_value=0, max_value=5))
    exact = [int(x) ** exponent for x in a.tolist()]
    expect_overflow = any(not fits(v, np.dtype(dtype)) for v in exact)
    ca = checked_array(a)
    if expect_overflow:
        with pytest.raises(IntegerOverflowError):
            ca**exponent
    else:
        assert (ca**exponent).tolist() == exact


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_left_shift_exact(dtype: type, data: st.DataObject) -> None:
    info = np.iinfo(dtype)
    a = data.draw(
        arrays(dtype=dtype, shape=5, elements=st.integers(min_value=int(info.min), max_value=int(info.max)))
    )
    shift = data.draw(st.integers(min_value=0, max_value=dtype(0).itemsize * 8 + 2))
    exact = [int(x) << shift for x in a.tolist()]
    expect_overflow = any(not fits(v, np.dtype(dtype)) for v in exact)
    ca = checked_array(a)
    if expect_overflow:
        with pytest.raises(IntegerOverflowError):
            ca << shift
    else:
        assert (ca << shift).tolist() == exact


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_1d_matmul_exact(dtype: type, data: st.DataObject) -> None:
    # 1-D @ 1-D (vector dot product) contracts to a scalar, which is the
    # shape that exposed the int64/uint64 object-dtype fallback returning a
    # bare Python bool/int instead of something array-like.
    a = data.draw(_int_arrays(dtype, n=4))
    b = data.draw(_int_arrays(dtype, n=4))
    exact = sum(int(x) * int(y) for x, y in zip(a.tolist(), b.tolist()))
    expect_overflow = not fits(exact, np.dtype(dtype))
    ca, cb = checked_array(a), checked_array(b)
    if expect_overflow:
        with pytest.raises(IntegerOverflowError) as excinfo:
            ca @ cb
        assert excinfo.value.index == ()
        with pytest.raises(IntegerOverflowError):
            ca.dot(cb)
    else:
        assert int(ca @ cb) == exact
        assert int(ca.dot(cb)) == exact


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_scalar_operand_mix_exact(dtype: type, data: st.DataObject) -> None:
    # A checked_array combined with a bare Python int (not another
    # checked_array/ndarray) on either side of the operator -- this drove
    # narrow-dtype arrays through np.broadcast_arrays against a 0-d operand,
    # and int64/uint64 through the nonzero()-based candidate selection with
    # a 0-d shape on one side.
    info = np.iinfo(dtype)
    a = data.draw(_int_arrays(dtype, n=3))
    scalar = data.draw(st.integers(min_value=int(info.min), max_value=int(info.max)))
    ca = checked_array(a)

    exact_add = [int(x) + scalar for x in a.tolist()]
    if any(not fits(v, np.dtype(dtype)) for v in exact_add):
        with pytest.raises(IntegerOverflowError):
            ca + scalar
        with pytest.raises(IntegerOverflowError):
            scalar + ca
    else:
        assert (ca + scalar).tolist() == exact_add
        assert (scalar + ca).tolist() == exact_add

    exact_mul = [int(x) * scalar for x in a.tolist()]
    if any(not fits(v, np.dtype(dtype)) for v in exact_mul):
        with pytest.raises(IntegerOverflowError):
            ca * scalar
        with pytest.raises(IntegerOverflowError):
            scalar * ca
    else:
        assert (ca * scalar).tolist() == exact_mul
        assert (scalar * ca).tolist() == exact_mul


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES), data=st.data())
def test_0d_scalar_array_exact(dtype: type, data: st.DataObject) -> None:
    # A 0-d checked_array (a scalar wrapped as an array, as opposed to a
    # bare Python int operand) hits the same np.nonzero()-on-0-d hazard for
    # multiply, since a 0-d array does not auto-promote the way a Python
    # int operand does going into np.broadcast_arrays.
    info = np.iinfo(dtype)
    x = data.draw(st.integers(min_value=int(info.min), max_value=int(info.max)))
    y = data.draw(st.integers(min_value=int(info.min), max_value=int(info.max)))
    a = checked_array(np.array(x, dtype=dtype))
    b = checked_array(np.array(y, dtype=dtype))

    exact = int(x) * int(y)
    if not fits(exact, np.dtype(dtype)):
        with pytest.raises(IntegerOverflowError) as excinfo:
            a * b
        assert excinfo.value.index == ()
    else:
        assert int(a * b) == exact


# -- negative controls: values comfortably inside range never raise --------


@_SETTINGS
@given(dtype=st.sampled_from(ALL_INT_DTYPES))
def test_small_values_never_raise(dtype: type) -> None:
    a = checked_array(np.array([0, 1, 2, 3], dtype=dtype))
    b = checked_array(np.array([1, 1, 1, 1], dtype=dtype))
    assert (a + b).tolist() == [1, 2, 3, 4]
    assert (a * b).tolist() == [0, 1, 2, 3]


def test_int8_min_boundary_add_no_overflow() -> None:
    # -128 + 1 = -127: fits, must not raise.
    a = checked_array([-128], dtype=np.int8)
    assert (a + 1).tolist() == [-127]


def test_int8_max_boundary_sub_no_overflow() -> None:
    # 127 - 1 = 126: fits, must not raise.
    a = checked_array([127], dtype=np.int8)
    assert (a - 1).tolist() == [126]


def test_negate_int_min_overflows_int8() -> None:
    # -(-128) == 128, which does not fit in int8. This is the classic
    # two's-complement trap: negation is not always safe even though it
    # looks like the identity op.
    a = checked_array([-128], dtype=np.int8)
    with pytest.raises(IntegerOverflowError):
        -a


def test_negate_int_min_overflows_every_signed_width() -> None:
    dtypes: list[Any] = [np.int8, np.int16, np.int32, np.int64]
    for dtype in dtypes:
        min_val = int(np.iinfo(dtype).min)
        a = checked_array([min_val], dtype=dtype)
        with pytest.raises(IntegerOverflowError):
            -a


def test_negate_min_plus_one_does_not_overflow() -> None:
    # -(-127) == 127 fits int8 -- a near neighbour of the trap above that
    # must NOT raise, guarding against an off-by-one in the check.
    a = checked_array([-127], dtype=np.int8)
    assert (-a).tolist() == [127]


def test_abs_int_min_overflows() -> None:
    a = checked_array([-128], dtype=np.int8)
    with pytest.raises(IntegerOverflowError):
        abs(a)


def test_abs_min_plus_one_does_not_overflow() -> None:
    a = checked_array([-127], dtype=np.int8)
    assert abs(a).tolist() == [127]
