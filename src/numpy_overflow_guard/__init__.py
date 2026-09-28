"""Catch silent integer overflow in NumPy array arithmetic.

``np.errstate(over="raise")`` and ``np.seterr(over="raise")`` only catch
overflow in scalar operations. Array operations wrap around silently -- see
the project README for the measured proof and the relevant NumPy issues.
This package adds array-level checking on top.
"""

from ._array import checked_array
from ._errors import IntegerOverflowError
from ._mode import checked, current_mode

__all__ = ["checked", "checked_array", "current_mode", "IntegerOverflowError"]

__version__ = "0.1.0"
