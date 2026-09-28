"""The ``checked()`` context manager / decorator that selects overflow handling."""

from __future__ import annotations

import contextvars
from contextlib import ContextDecorator
from typing import Literal

Mode = Literal["raise", "warn"]

_VALID_MODES = ("raise", "warn")

_mode_var: contextvars.ContextVar[Mode] = contextvars.ContextVar(
    "numpy_overflow_guard_mode", default="raise"
)


def current_mode() -> Mode:
    """The overflow-handling mode in effect for the current context."""
    return _mode_var.get()


class checked(ContextDecorator):
    """Set the overflow-handling mode for :class:`checked_array` operations.

    Use as a context manager or a decorator. ``checked_array`` instances that
    were not given an explicit ``mode=`` at construction time look up the mode
    active at the time each operation runs, via a `contextvars.ContextVar`, so
    it composes correctly across threads and ``asyncio`` tasks.

    Example:
        >>> from numpy_overflow_guard import checked, checked_array
        >>> a = checked_array([1, 2, 3], dtype="int8")
        >>> with checked(mode="warn"):
        ...     a * 100  # doctest: +SKIP
    """

    def __init__(self, mode: Mode = "raise") -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES!r}, got {mode!r}")
        self.mode: Mode = mode
        self._token: contextvars.Token[Mode] | None = None

    def __enter__(self) -> "checked":
        self._token = _mode_var.set(self.mode)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._token is not None:
            _mode_var.reset(self._token)
            self._token = None

    def _recreate_cm(self) -> "checked":
        # Each decorated call gets its own instance, so concurrent/recursive
        # calls to a @checked()-decorated function do not share `_token`.
        return type(self)(self.mode)
