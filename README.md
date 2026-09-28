# numpy-overflow-guard

Detects silent integer overflow in NumPy array arithmetic, which `np.errstate` and `np.seterr` do not catch.

```python
from numpy_overflow_guard import checked_array

balances = checked_array([2**62, 2**62], dtype="int64")
balances * 4  # raises IntegerOverflowError instead of silently returning [0, 0]
```

## Why

NumPy's overflow reporting (`np.seterr`/`np.errstate(over=...)`) only instruments
*scalar* integer operations. Array operations wrap around with no warning and no
exception, even when every other error mode is set to `"raise"`.

Measured: `python3 -c "import numpy as np; ..."` (numpy 2.5.3)

```pycon
>>> import numpy as np
>>> with np.errstate(all="raise"):
...     np.array([2**62, 2**62], dtype=np.int64) * 4
...
array([0, 0])
>>> with np.errstate(all="raise"):
...     np.int64(2**62) * np.int64(4)
...
FloatingPointError: overflow encountered in scalar multiply
```

Same operation, same dtype, same `errstate`. The scalar path raises; the array
path returns `[0, 0]` and continues running. NumPy's own documentation for
`seterr` notes the scalar-only scope, and several open issues describe the gap
directly:

- [numpy/numpy#8987](https://github.com/numpy/numpy/issues/8987): *"Integer overflow warning applies to scalars but not arrays"* -- `np.uint8(0) - np.uint8(1)` warns; `np.uint8(0)[...] - np.uint8(1)[...]` (the array path) does not.
- [numpy/numpy#21219](https://github.com/numpy/numpy/issues/21219): `np.prod` on a plain Python list of ints "silently creates garbage values" by wrapping in `int32`, with no warning.
- [numpy/numpy#5657](https://github.com/numpy/numpy/issues/5657): `absolute()` on `INT_MIN` returns a negative number unchanged, rather than converting or erroring.
- [numpy/numpy#17982](https://github.com/numpy/numpy/issues/17982): a request to even *document* that signed-integer overflow behavior is wraparound, since "signed integer overflow is undefined behavior in C" and NumPy does not commit to it across compilers.
- [numpy/numpy#28837](https://github.com/numpy/numpy/issues/28837): an open 2025 feature request for saturating arithmetic, i.e. the problem is still unresolved upstream.

Searched for prior art before writing this (`gh search repos`/`code` for "numpy
overflow guard", "checked array overflow numpy", "safe integer numpy array
overflow", "np.errstate overflow array", `__array_ufunc__` combined with
"overflow"): nothing that checks NumPy *array* integer overflow exactly turned
up.

## Installation

```sh
pip install numpy-overflow-guard
```

Python >= 3.11. Runtime dependency: `numpy` (only). NumPy 2's [NEP
50](https://numpy.org/neps/nep-0050-scalar-promotion.html) promotion rules are
what make the mixed signed/unsigned behavior below exact.

## Usage

### Basic arithmetic

```python
from numpy_overflow_guard import checked_array

a = checked_array([100, 2, 3], dtype="int8")
a + 100          # IntegerOverflowError: add overflows int8 at output index (0,)
```

Indexing a single element out of a `checked_array` returns a plain NumPy
scalar (`np.int8`, etc.), same as it would for a plain `ndarray` -- that
scalar is no longer guarded by this package, though NumPy's own
`np.seterr(over="raise")` already covers scalar overflow (see "Why").

### Reductions: mind the accumulator dtype

`sum()`/`cumsum()`/`prod()`/`cumprod()` behave exactly like plain NumPy: with
no explicit `dtype=`, NumPy accumulates in a wide default integer type, so
these essentially never overflow. Ask for the narrow dtype explicitly and the
guard applies to *that* accumulator:

```python
a = checked_array([100, 100, 100], dtype="int8")
a.sum()                 # 300 -- default accumulator is wide, no overflow
a.sum(dtype="int8")     # IntegerOverflowError: add.reduce overflows int8 ...
a.cumsum(dtype="int8")  # ... at output index (1,) -- the first bad prefix
```

### matmul, `@`, and `.dot()`

```python
A = checked_array([[100, 100], [1, 1]], dtype="int8")
A @ A          # IntegerOverflowError: matmul overflows int8 at output index (0, 0)
A.dot(A)       # same check -- .dot() is redirected through the same path
```

### Narrowing casts

```python
big = checked_array([300], dtype="int64")
big.astype("int8")   # IntegerOverflowError: astype overflows int8 ...
```

### Warn instead of raise

```python
from numpy_overflow_guard import checked

a = checked_array([127], dtype="int8", mode="warn")
a + 1   # RuntimeWarning, returns NumPy's own wrapped value: array([-128])

with checked(mode="warn"):        # or as a decorator: @checked(mode="warn")
    checked_array([127], dtype="int8") + 1
```

## Reference

### `checked_array(data, dtype=None, *, mode=None)`

An `np.ndarray` subclass. Everything that works on a plain array works here --
indexing, slicing, comparisons, `repr`. The guarded operations are: `+`, `-`,
`*`, unary `-`, `abs()`, `**`, `<<`, `@`/`.dot()`/`np.matmul`, and the
reduce/accumulate forms of `+`/`*` (`sum`, `prod`, `cumsum`, `cumprod`).
Everything else passes straight through to NumPy.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | anything `numpy.asarray` accepts | -- | Array data |
| `dtype` | dtype-like | `None` | Forwarded to `numpy.asarray` |
| `mode` | `"raise"` \| `"warn"` \| `None` | `None` | Per-instance override. `None` means "look up the ambient mode from `checked()`, default `raise`", every time an operation runs |

### `checked(mode="raise")`

A context manager and decorator (`contextlib.ContextDecorator`) that sets the
ambient overflow mode for any `checked_array` operation performed within it
(unless that instance was given its own `mode=`). Backed by a
`contextvars.ContextVar`, so it composes correctly across threads and asyncio
tasks, and a decorated function gets a fresh instance per call.

### `IntegerOverflowError`

A `FloatingPointError` subclass -- the same base class NumPy itself uses for
scalar overflow, so `except FloatingPointError` still catches it.

| Attribute | Type | Description |
| --- | --- | --- |
| `op` | `str` | Operation name, e.g. `"multiply"`, `"add.reduce"`, `"astype"` |
| `index` | `tuple[int, ...] \| None` | Index of the first offending element in the operation's output |
| `dtype` | `numpy.dtype` | The dtype whose range was exceeded |

## How it works

`checked_array` implements [NEP 13](https://numpy.org/neps/nep-0013-ufunc-overrides.html)'s
`__array_ufunc__` hook, which every NumPy ufunc call (`+`, `*`, `np.add.reduce`
for `.sum()`, `np.matmul` for `@`, etc.) is routed through when any operand is
a subclass that defines it. Each guarded ufunc first runs NumPy's own
(unguarded, wrapping) computation, then checks whether the exact mathematical
result fits the output dtype:

- **add/subtract**: an exact sign-comparison test (`sign(a) == sign(b) !=
  sign(result)` for addition; the mirror for subtraction), the same technique
  compilers use for build-time overflow checks. No promotion needed.
- **multiply**: for anything narrower than 64 bits, promote to a native 64-bit
  type, multiply exactly, and compare against the original dtype's range.
  `int64`/`uint64` have no wider native NumPy dtype, so a conservative
  magnitude bound (computed in `float64`, used only to pick candidates, never
  to decide overflow) selects the handful of elements that could plausibly
  overflow, and only those are re-computed exactly with Python's
  arbitrary-precision `int`.
- **negative/absolute**: only the dtype minimum is unrepresentable for signed
  types (`-(-128)` for `int8` is `128`, which does not fit) -- a single
  comparison.
- **power/left_shift**: exact Python-`int` recomputation, gated by the same
  candidate-selection trick (a `log2`-based bit-count estimate for power, "was
  the shift amount zero" for left_shift) to skip the exact path when it is
  provably unnecessary.
- **matmul**: promotes when a wider native dtype exists; for `int64`/`uint64`,
  a magnitude-times-inner-dimension bound decides whether an exact
  object-dtype matmul is needed at all.
- **reductions** (`sum`, `prod`, `cumsum`, `cumprod`, via `ufunc.reduce`/
  `.accumulate`): re-run the same reduction in a wider dtype (or, for
  64-bit types, in Python's `object` dtype) and compare.
- **astype**: narrowing casts between two integer dtypes are compared directly
  against the destination dtype's range.

`.astype()` and `.dot()` do not go through `__array_ufunc__` in NumPy itself
(the latter calls BLAS directly), so both are overridden explicitly;
`.dot()` is redirected through `np.matmul`, which does dispatch through
`__array_ufunc__`.

## Limitations

- **Overhead is real and, for the `int64`/`uint64` object-dtype fallback
  path, large.** Measured on this machine (Python 3.14.7, NumPy 2.5.3, best of
  7 runs, `bench/measure_overhead.py`):

  ```
  add (int64):
        size   plain (ms)   checked (ms)   overhead
         100       0.0002         0.0045      27.0x
       10000       0.0012         0.0088       7.3x
     1000000       0.1365         0.4755       3.5x

  multiply (int64, small magnitude -- fast path) (int64):
        size   plain (ms)   checked (ms)   overhead
         100       0.0002         0.0062      29.7x
       10000       0.0027         0.0200       7.3x
     1000000       0.2475         1.3780       5.6x

  multiply (int8, promoted to int64 and back) (int8):
        size   plain (ms)   checked (ms)   overhead
         100       0.0002         0.0045      21.7x
       10000       0.0004         0.0115      30.8x
     1000000       0.0165         0.6988      42.3x

  sum (dtype=int8, narrow accumulator) (int8):
        size   plain (ms)   checked (ms)   overhead
         100       0.0003         0.0037      12.6x
       10000       0.0004         0.0053      14.2x
     1000000       0.0093         0.1923      20.8x

  sum (dtype=int64, object fallback) (int64):
        size   plain (ms)   checked (ms)   overhead
         100       0.0003         0.0062      21.3x
       10000       0.0009         0.1520     173.7x
     1000000       0.0740        17.6606     238.7x
  ```

  Elementwise `add`/`multiply` on `int64` cost roughly 3.5-30x plain NumPy,
  falling as arrays grow (fixed per-call Python overhead amortizes). `sum`
  with an explicit narrow accumulator is the worst case: for `int64`/`uint64`
  there is no wider native dtype, so the check falls back to Python's
  arbitrary-precision `object` dtype for the *entire* reduction, which is
  roughly 175-240x slower at 10K-1M elements. There is no `sample=` or
  per-op opt-out for this in the current version -- a sampling check would
  make the "raises iff the exact result doesn't fit" guarantee probabilistic,
  which contradicts the point of the package, so it was left out rather than
  built half-heartedly. If you need a narrow-dtype accumulator over large
  `int64`/`uint64` arrays, this library is currently the wrong tool for that
  specific call; smaller dtypes' reductions are promoted to a native 64-bit
  accumulator and stay cheap.
- **Signed overflow wraparound is assumed, not guaranteed by the C standard.**
  The sign-comparison and bit-count checks rely on NumPy's compiled ufuncs
  actually wrapping (two's-complement, modular) on overflow. This holds on
  every platform NumPy ships wheels for, but signed overflow is undefined
  behavior in C and NumPy's own docs do not commit to it across compilers
  (numpy/numpy#17982).
- **`power`'s candidate selection uses a `log2`-based float estimate** to
  decide which elements need exact re-checking (with an 8-bit safety margin).
  The overflow decision itself is always exact; the estimate only affects
  which elements get the expensive exact path, and errs towards checking more,
  not fewer.
- **No `float16`/`float32`/`float64` guarding.** This library only instruments
  integer dtypes; floating-point arithmetic already has an overflow signal
  (`inf`) and different failure modes (precision loss) that this package does
  not address.
- **`matmul`/`.dot()` supports the shapes NumPy's `@` supports** (1-D dot
  products, 2-D matrix multiply, and their broadcasting rules); it was tested
  against 1-D and 2-D cases.
- **`out=` is passed through but only lightly tested.** The common
  no-`out` path is what the test suite and the property tests exercise.

## Compatibility

Tested on this machine: Python 3.14.7, NumPy 2.5.3. CI (`.github/workflows/test.yml`)
runs the suite on Python 3.11, 3.12, 3.13, and 3.14, but those runs were not
observed locally -- 3.14.7 is the only version actually exercised here.

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m mypy src tests --strict
.venv/bin/python bench/measure_overhead.py
```

## License

MIT. See [LICENSE](LICENSE).
