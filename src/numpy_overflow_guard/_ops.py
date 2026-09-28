"""Exact overflow detectors, one per guarded operation.

Every function here returns a boolean mask (same shape as the operation's
output) that is ``True`` wherever the *exact* mathematical result does not
fit in the operation's dtype. None of these decide overflow from a float
approximation: narrow dtypes are checked by promoting to a native dtype wide
enough to hold any result exactly; int64/uint64 (which have no wider native
dtype) fall back to Python's arbitrary-precision ints for the lanes that
could plausibly overflow. A float-based magnitude estimate is used only to
pick *which* lanes get that exact recheck -- never to decide overflow itself.

Array types here are annotated as ``Any``: NumPy's own stubs do not model
"an ndarray of some integer dtype" precisely enough to be useful, and this
module never hands one of these values back to a caller uncontrolled --
``_array.py`` is the public, precisely-typed boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._bounds import (
    INT64_MUL_SAFE_BOUND,
    UINT64_MUL_SAFE_BOUND,
    bounds,
    wider_dtype,
)


def first_bad_index(mask: Any) -> tuple[int, ...] | None:
    """The first ``True`` position in ``mask``, in C (row-major) order."""
    if not mask.any():
        return None
    flat_index = int(np.flatnonzero(mask.ravel())[0])
    return tuple(int(i) for i in np.unravel_index(flat_index, mask.shape))


def _no_overflow(shape: tuple[int, ...]) -> Any:
    return np.zeros(shape, dtype=np.bool_)


# --------------------------------------------------------------------------
# add / subtract: exact sign-comparison test, no promotion needed.
# --------------------------------------------------------------------------


def check_add(a: Any, b: Any, result: Any, dtype: np.dtype[Any]) -> Any:
    a, b = np.broadcast_arrays(a, b)
    if dtype.kind == "i":
        sign_a = a >= 0
        sign_b = b >= 0
        sign_r = result >= 0
        return (sign_a == sign_b) & (sign_r != sign_a)
    # Unsigned wraparound: the wrapped sum of a + b can only be < a if it wrapped.
    return result < a


def check_subtract(a: Any, b: Any, result: Any, dtype: np.dtype[Any]) -> Any:
    a, b = np.broadcast_arrays(a, b)
    if dtype.kind == "i":
        sign_a = a >= 0
        sign_b = b >= 0
        sign_r = result >= 0
        return (sign_a != sign_b) & (sign_r != sign_a)
    # Unsigned: a - b is only representable when b <= a.
    return b > a


# --------------------------------------------------------------------------
# negative / absolute: only the dtype minimum (signed) is unrepresentable.
# --------------------------------------------------------------------------


def check_negative(a: Any, dtype: np.dtype[Any]) -> Any:
    if dtype.kind == "i":
        min_val, _ = bounds(dtype)
        return a == min_val
    # Unsigned: -0 is fine, -x for any x > 0 is not representable.
    return a != 0


def check_absolute(a: Any, dtype: np.dtype[Any]) -> Any:
    if dtype.kind == "i":
        min_val, _ = bounds(dtype)
        return a == min_val
    return _no_overflow(np.shape(a))


# --------------------------------------------------------------------------
# multiply
# --------------------------------------------------------------------------


def check_multiply(a: Any, b: Any, dtype: np.dtype[Any]) -> Any:
    a, b = np.broadcast_arrays(a, b)
    wide = wider_dtype(dtype)
    lo, hi = bounds(dtype)
    if wide is not None:
        exact = a.astype(wide) * b.astype(wide)
        return (exact < lo) | (exact > hi)

    # int64 / uint64: no wider native dtype. Use a conservative magnitude
    # bound (computed in float64, so it can be loose but never unsafe) to
    # find the lanes that could possibly overflow, then settle only those
    # exactly with Python ints.
    bound = INT64_MUL_SAFE_BOUND if dtype.kind == "i" else UINT64_MUL_SAFE_BOUND
    mag_a = np.abs(a.astype(np.float64))
    mag_b = np.abs(b.astype(np.float64))
    candidates = (mag_a > bound) | (mag_b > bound)
    overflow = _no_overflow(a.shape)
    if candidates.any():
        idx = np.nonzero(candidates)
        exact = a[idx].astype(object) * b[idx].astype(object)
        bad = (exact < lo) | (exact > hi)
        overflow[idx] = bad.astype(np.bool_)
    return overflow


# --------------------------------------------------------------------------
# power (integer exponents only -- NumPy itself rejects negative exponents
# on integer dtypes before we ever see the call).
# --------------------------------------------------------------------------


def check_power(a: Any, exponent: Any, dtype: np.dtype[Any]) -> Any:
    a, exponent = np.broadcast_arrays(a, exponent)
    lo, hi = bounds(dtype)
    width_bits = dtype.itemsize * 8

    safe_base = (a >= -1) & (a <= 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        needed_bits = np.abs(exponent.astype(np.float64)) * np.log2(
            np.abs(a.astype(np.float64)) + 1.0
        )
    margin = 8  # generous, to absorb float64 log2 imprecision
    candidates = (~safe_base) & (needed_bits > (width_bits - margin))

    overflow = _no_overflow(a.shape)
    if candidates.any():
        idx = np.nonzero(candidates)
        exact = a[idx].astype(object) ** exponent[idx].astype(object)
        bad = (exact < lo) | (exact > hi)
        overflow[idx] = bad.astype(np.bool_)
    return overflow


# --------------------------------------------------------------------------
# left_shift
# --------------------------------------------------------------------------


def check_left_shift(a: Any, shift: Any, dtype: np.dtype[Any]) -> Any:
    a, shift = np.broadcast_arrays(a, shift)
    lo, hi = bounds(dtype)
    nonzero = shift != 0
    overflow = _no_overflow(a.shape)
    if nonzero.any():
        idx = np.nonzero(nonzero)
        exact = a[idx].astype(object) << shift[idx].astype(object)
        bad = (exact < lo) | (exact > hi)
        overflow[idx] = bad.astype(np.bool_)
    return overflow


# --------------------------------------------------------------------------
# reduce / accumulate (sum, prod, cumsum, cumprod)
# --------------------------------------------------------------------------


def check_reduce_or_accumulate(
    ufunc: np.ufunc,
    method: str,
    arr: Any,
    dtype: np.dtype[Any],
    extra_kwargs: dict[str, Any],
) -> Any:
    lo, hi = bounds(dtype)
    wide = wider_dtype(dtype)
    # The caller's own `dtype=` (if any) selected the *narrow* accumulator
    # we're checking; reusing it for the recomputation would just reproduce
    # the same wraparound instead of an exact reference value.
    exact_kwargs = {k: v for k, v in extra_kwargs.items() if k != "dtype"}
    if wide is not None:
        wide_arr = arr.astype(wide)
        exact = getattr(ufunc, method)(wide_arr, dtype=wide, **exact_kwargs)
    else:
        obj_arr = arr.astype(object)
        exact = np.asarray(getattr(ufunc, method)(obj_arr, **exact_kwargs))
    return (exact < lo) | (exact > hi)


# --------------------------------------------------------------------------
# matmul (also reached via .dot(), see _array.py)
# --------------------------------------------------------------------------


def check_matmul(a: Any, b: Any, dtype: np.dtype[Any], result_shape: tuple[int, ...]) -> Any:
    lo, hi = bounds(dtype)
    wide = wider_dtype(dtype)
    if wide is not None:
        exact = a.astype(wide) @ b.astype(wide)
        return (exact < lo) | (exact > hi)

    # int64 / uint64: matmul accumulates k products per output element. A
    # conservative bound on the largest possible |sum of products| tells us,
    # cheaply, whether an exact (and much slower) object-dtype matmul is
    # even necessary.
    k = a.shape[-1] if a.ndim else 1
    mag_a = float(np.max(np.abs(a.astype(np.float64)))) if a.size else 0.0
    mag_b = float(np.max(np.abs(b.astype(np.float64)))) if b.size else 0.0
    bound_val = mag_a * mag_b * k
    limit = hi if dtype.kind == "u" else min(hi, -lo)
    if bound_val < limit * 0.9:
        return _no_overflow(result_shape)

    exact = a.astype(object) @ b.astype(object)
    return (exact < lo) | (exact > hi)
