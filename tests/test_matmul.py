"""matmul, ``@``, and ``.dot()``."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_overflow_guard import IntegerOverflowError, checked_array


def test_matmul_overflows_int8() -> None:
    a = checked_array([[100, 100], [1, 1]], dtype=np.int8)
    b = checked_array([[100, 100], [1, 1]], dtype=np.int8)
    with pytest.raises(IntegerOverflowError):
        a @ b


def test_dot_matches_matmul_behaviour() -> None:
    a = checked_array([[100, 100], [1, 1]], dtype=np.int8)
    b = checked_array([[100, 100], [1, 1]], dtype=np.int8)
    with pytest.raises(IntegerOverflowError):
        a.dot(b)


def test_matmul_safe_case() -> None:
    a = checked_array([[1, 2], [3, 4]], dtype=np.int8)
    b = checked_array([[5, 6], [7, 8]], dtype=np.int8)
    result = a @ b
    assert result.tolist() == (np.asarray(a).astype(np.int64) @ np.asarray(b).astype(np.int64)).tolist()


def test_matmul_int64_overflows() -> None:
    n = 2**62
    a = checked_array([[n, n]], dtype=np.int64)
    b = checked_array([[2], [2]], dtype=np.int64)
    with pytest.raises(IntegerOverflowError):
        a @ b


def test_matmul_int64_safe_case_matches_python() -> None:
    a = checked_array([[1, 2, 3]], dtype=np.int64)
    b = checked_array([[4], [5], [6]], dtype=np.int64)
    result = a @ b
    assert result.tolist() == [[32]]


def test_vector_dot_product() -> None:
    a = checked_array([1, 2, 3], dtype=np.int32)
    b = checked_array([4, 5, 6], dtype=np.int32)
    assert int(a @ b) == 32
