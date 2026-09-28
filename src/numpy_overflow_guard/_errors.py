"""Exception type raised on detected integer overflow."""

from __future__ import annotations

from typing import Any


class IntegerOverflowError(FloatingPointError):
    """A guarded integer operation would silently overflow its dtype.

    This mirrors the exception NumPy itself raises for *scalar* integer
    overflow under ``np.errstate(over="raise")`` (also a ``FloatingPointError``
    subclass), but it fires for array operations too -- something NumPy does
    not do (see the README's "Why" section for the underlying issues).

    Attributes:
        op: Name of the operation that overflowed, e.g. ``"add"`` or
            ``"multiply"``.
        index: Index (as a tuple, matching ``numpy`` fancy-index conventions)
            of the first offending element in the operation's output, or
            ``None`` when not applicable.
        dtype: The dtype whose range was exceeded.
    """

    def __init__(
        self,
        message: str,
        *,
        op: str,
        index: tuple[int, ...] | None,
        dtype: Any,
    ) -> None:
        super().__init__(message)
        self.op = op
        self.index = index
        self.dtype = dtype
