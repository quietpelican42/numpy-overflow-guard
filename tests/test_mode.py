"""The ``checked()`` context manager / decorator and ``mode="warn"``."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from numpy_overflow_guard import IntegerOverflowError, checked, checked_array


def test_default_mode_raises() -> None:
    a = checked_array([127], dtype=np.int8)
    with pytest.raises(IntegerOverflowError):
        a + 1


def test_per_instance_warn_mode() -> None:
    a = checked_array([127], dtype=np.int8, mode="warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = a + 1
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    # NumPy's own wrapped value is still returned in warn mode.
    assert result.tolist() == [-128]


def test_context_manager_sets_warn_mode() -> None:
    a = checked_array([127], dtype=np.int8)
    with checked(mode="warn"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            a + 1
        assert len(caught) == 1
    # Outside the context, raising resumes.
    with pytest.raises(IntegerOverflowError):
        a + 1


def test_checked_as_decorator() -> None:
    a = checked_array([127], dtype=np.int8)

    @checked(mode="warn")
    def bump() -> Any:
        return a + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = bump()
    assert len(caught) == 1
    assert result.tolist() == [-128]


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        checked(mode="explode")  # type: ignore[arg-type]


def test_per_instance_mode_overrides_context() -> None:
    # An explicit mode= at construction wins over the ambient context.
    a = checked_array([127], dtype=np.int8, mode="raise")
    with checked(mode="warn"):
        with pytest.raises(IntegerOverflowError):
            a + 1
