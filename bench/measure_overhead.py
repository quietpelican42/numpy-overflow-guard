"""Measure the real overhead of checked_array vs. plain ndarray.

Run with: .venv/bin/python bench/measure_overhead.py
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from numpy_overflow_guard import checked_array

SIZES = [100, 10_000, 1_000_000]
REPEATS = 7


def _time(fn: Callable[[], object], repeats: int = REPEATS) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def bench_op(name: str, dtype: type, make_plain: Callable[[int], object], op: Callable[[object], object]) -> None:
    print(f"\n{name} ({np.dtype(dtype)}):")
    print(f"{'size':>10} {'plain (ms)':>12} {'checked (ms)':>14} {'overhead':>10}")
    for size in SIZES:
        plain = make_plain(size)
        checked = checked_array(plain, dtype=dtype)
        t_plain = _time(lambda: op(plain)) * 1000
        t_checked = _time(lambda: op(checked)) * 1000
        overhead = t_checked / t_plain if t_plain > 0 else float("nan")
        print(f"{size:>10} {t_plain:>12.4f} {t_checked:>14.4f} {overhead:>9.1f}x")


def main() -> None:
    rng = np.random.default_rng(0)

    def make_int64(size: int) -> object:
        return rng.integers(-1_000_000, 1_000_000, size=size, dtype=np.int64)

    def make_int8_small(size: int) -> object:
        # Small enough that a*a and a running sum never overflow int8,
        # regardless of array size -- this benchmark measures overhead,
        # not correctness (that's what the test suite is for).
        return rng.integers(-5, 5, size=size, dtype=np.int8)

    def make_int8_zeros(size: int) -> object:
        return np.zeros(size, dtype=np.int8)

    bench_op("add", np.int64, make_int64, lambda a: a + a)  # type: ignore[operator]
    bench_op("multiply (int64, small magnitude -- fast path)", np.int64, make_int64, lambda a: a * a)  # type: ignore[operator]
    bench_op("multiply (int8, promoted to int64 and back)", np.int8, make_int8_small, lambda a: a * a)  # type: ignore[operator]
    bench_op("sum (dtype=int8, narrow accumulator)", np.int8, make_int8_zeros, lambda a: a.sum(dtype=np.int8))  # type: ignore[attr-defined]
    bench_op("sum (dtype=int64, object fallback)", np.int64, make_int64, lambda a: a.sum(dtype=np.int64))  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
