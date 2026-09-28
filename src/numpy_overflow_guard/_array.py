"""``checked_array``: an ndarray subclass that guards integer overflow."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import _ops
from ._bounds import bounds, is_integer_dtype
from ._errors import IntegerOverflowError
from ._mode import Mode, current_mode

# ufunc name -> arity handled directly in __array_ufunc__.
_ONE_INPUT = frozenset({"negative", "absolute"})
_TWO_INPUT_ELEMENTWISE = frozenset({"add", "subtract", "multiply", "power", "square", "left_shift"})
_MATMUL = frozenset({"matmul"})
_REDUCE_ACCUMULATE = frozenset({"add", "multiply"})


def _to_plain(value: Any) -> Any:
    """Strip the ``checked_array`` wrapper so the native ufunc runs unguarded."""
    if isinstance(value, checked_array):
        return np.asarray(value).view(np.ndarray)
    return value


class checked_array(np.ndarray):
    """A NumPy ``ndarray`` that raises on silent integer overflow.

    Behaves exactly like a regular array -- arithmetic, reductions
    (``sum``, ``prod``, ``cumsum``, ``cumprod``), ``matmul``/``@``/``.dot()``,
    and narrowing ``.astype()`` calls all go through the same dtype-range
    checks as everywhere else in this package. On overflow it raises
    :class:`~numpy_overflow_guard.IntegerOverflowError` naming the first
    offending index, or -- with ``mode="warn"`` -- emits a ``RuntimeWarning``
    and returns NumPy's own (wrapped) result.

    Args:
        data: Anything accepted by ``numpy.asarray``.
        dtype: Passed through to ``numpy.asarray``.
        mode: ``"raise"`` (default) or ``"warn"``. When omitted, the mode
            active via :class:`~numpy_overflow_guard.checked` (or ``"raise"``
            if none is active) is looked up fresh on every operation.
    """

    _overflow_mode: Mode | None

    def __new__(
        cls,
        data: Any,
        dtype: Any = None,
        *,
        mode: Mode | None = None,
    ) -> "checked_array":
        arr = np.asarray(data, dtype=dtype)
        obj = arr.view(cls)
        obj._overflow_mode = mode
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        self._overflow_mode = getattr(obj, "_overflow_mode", None)

    @property
    def mode(self) -> Mode:
        return self._overflow_mode if self._overflow_mode is not None else current_mode()

    def _report(self, overflow: Any, op: str, dtype: np.dtype[Any]) -> None:
        if not np.any(overflow):
            return
        index = _ops.first_bad_index(overflow)
        message = f"{op} overflows {dtype} at output index {index}"
        if self.mode == "warn":
            warnings.warn(message, category=RuntimeWarning, stacklevel=4)
        else:
            raise IntegerOverflowError(message, op=op, index=index, dtype=dtype)

    # -- ufunc dispatch (NEP 13) --------------------------------------------

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        plain_inputs = [_to_plain(i) for i in inputs]
        out = kwargs.pop("out", None)
        if out is not None:
            kwargs["out"] = tuple(_to_plain(o) for o in out)

        result = getattr(ufunc, method)(*plain_inputs, **kwargs)

        name = ufunc.__name__
        result_dtype = getattr(result, "dtype", None)
        if result_dtype is None or not is_integer_dtype(result_dtype):
            # NumPy itself promoted away from an integer result (for example
            # int64 + uint64 -> float64, per NEP 50): nothing to guard.
            return self._wrap(result)

        if method == "__call__":
            if name in _ONE_INPUT:
                a = np.asarray(plain_inputs[0])
                checker = _ops.check_negative if name == "negative" else _ops.check_absolute
                overflow = checker(a, result_dtype)
                self._report(overflow, name, result_dtype)
            elif name in _TWO_INPUT_ELEMENTWISE:
                a = np.asarray(plain_inputs[0])
                b = np.asarray(plain_inputs[1]) if len(plain_inputs) > 1 else a
                if name == "add":
                    overflow = _ops.check_add(a, b, np.asarray(result), result_dtype)
                elif name == "subtract":
                    overflow = _ops.check_subtract(a, b, np.asarray(result), result_dtype)
                elif name == "multiply":
                    overflow = _ops.check_multiply(a, b, result_dtype)
                elif name == "square":
                    overflow = _ops.check_multiply(a, a, result_dtype)
                elif name == "power":
                    overflow = _ops.check_power(a, b, result_dtype)
                else:  # left_shift
                    overflow = _ops.check_left_shift(a, b, result_dtype)
                self._report(overflow, name, result_dtype)
            elif name in _MATMUL:
                a = np.asarray(plain_inputs[0])
                b = np.asarray(plain_inputs[1])
                overflow = _ops.check_matmul(a, b, result_dtype, np.shape(result))
                self._report(overflow, name, result_dtype)
        elif method in ("reduce", "accumulate") and name in _REDUCE_ACCUMULATE:
            arr = np.asarray(plain_inputs[0])
            overflow = _ops.check_reduce_or_accumulate(ufunc, method, arr, result_dtype, dict(kwargs))
            self._report(overflow, f"{name}.{method}", result_dtype)

        return self._wrap(result)

    def _wrap(self, result: Any) -> Any:
        if isinstance(result, np.ndarray) and result.dtype != np.dtype(object):
            wrapped = result.view(checked_array)
            wrapped._overflow_mode = self._overflow_mode
            return wrapped
        return result

    # -- operations that bypass __array_ufunc__ entirely --------------------

    def dot(self, b: Any, out: Any = None) -> Any:
        # ndarray.dot() calls into BLAS directly and never reaches
        # __array_ufunc__; np.matmul does, and matches .dot()'s semantics
        # for the 1-D/2-D cases this package documents support for.
        return np.matmul(self, b, out=out) if out is not None else np.matmul(self, b)

    def astype(self, dtype: Any, *args: Any, **kwargs: Any) -> "checked_array":
        target_dtype = np.dtype(dtype)
        result = super().astype(target_dtype, *args, **kwargs)
        if is_integer_dtype(self.dtype) and is_integer_dtype(target_dtype):
            lo, hi = bounds(target_dtype)
            src = np.asarray(self).view(np.ndarray)
            overflow = (src < lo) | (src > hi)
            self._report(overflow, "astype", target_dtype)
        if isinstance(result, checked_array):
            result._overflow_mode = self._overflow_mode
        return result  # type: ignore[return-value]
