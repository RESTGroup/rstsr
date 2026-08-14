# NumPy Differences Report - Resolved (rstsr-core)

Fixed / resolved divergences, archived from `numpy_differences.md`. **Open**
divergences live in `numpy_differences.md`; see that file (and ADR-0003) for the
tags/format convention.

**Pinned NumPy:** v2.5.2 · **Checkout:** tag v2.5.2

Each entry here has `status: fixed`. Kept for history / regression context - the
parity test that surfaced it now asserts the correct NumPy behavior.

## `Tensor::i(int...)` 0-d scalar reads the correct element (FIXED)

- **numpy:** `_core/tests/test_indexing.py::TestIndexing::test_single_int_index`
  (L201) — `np.arange(10)[-1] == 9`.
- **rstsr:** entry_row_cpu::core_func::indexing::test_indexing::numpy_indexing::test_single_int_index
- **tag:** bug
- **status:** fixed

Integer indexing via `Tensor::i` that fully reduces the result to a **0-d**
(scalar) view returned the wrong element — it read offset 0, so e.g.
`arange(10).i(9).to_scalar() == 0` (not 9). The indexing itself (`dim_select`)
computed the layout offset correctly; the bug was in `TensorAny::to_scalar_f`,
which read `vec[0]` from the raw buffer returned by `to_cpu_vec` instead of
`vec[layout.offset()]`. Fixed: `to_scalar_f` now reads at the layout offset, so a
size-1 view reports its actual element regardless of where it slices into the
buffer. (This also makes the `.i(...).to_scalar()` checks in
`test_transpose::numpy_swapaxes` meaningful - they previously compared offset 0
against offset 0.) The parity test restores the 0-d scalar form
(`a.i(-1).to_scalar() == 9`, `m.i((1, 2)).to_scalar() == 6`).

## `sign(0.0)` returns 0 instead of NaN (FIXED)

- **numpy:** `np.sign(0.0) == 0` (and `np.sign` works on integers).
- **rstsr:** entry_row_cpu::core_func::math::test_unary_math::custom_math_basic
- **tag:** bug
- **status:** fixed

rstsr `rt::sign` was implemented as `x / x.abs()`, so `sign(0.0)` computed
`0 / 0 = NaN` (nonzero values were correct). Fixed in both the serial and rayon
`OpSignAPI` impls: a zero magnitude now maps to 0 (preserving the sign of `-0.0`
and complex zero, matching NumPy); otherwise `x / |x|` as before. The parity test
restores a `0.0` case asserting `sign([-2, 0, 3]) == [-1, 0, 1]`. Note rstsr
`sign` still requires a `Float` input (NumPy `sign` also accepts integers) - that
type-system difference remains intentional and is not covered here.

## `argmax_axes`/`argmin_axes` no longer panic for tensors of rank ≥ 3 (FIXED)

- **numpy:** `_core/tests/test_regression.py::TestRegression::test_argmax` (L268)
  expects high-dimensional argmax along each axis to succeed.
- **rstsr:** entry_row_cpu::core_func::reduction::test_argmax::numpy_argmax::test_regression
- **tag:** bug
- **status:** fixed

`argmax_all`/`argmin_all` and `argmax_axes`/`argmin_axes` on rank-1/2 tensors worked,
but for **rank ≥ 3** `argmax_axes`/`argmin_axes` panicked with `index out of bounds:
the len is 1 but the index is 1` at `rstsr-common/src/layout/layoutbase.rs:543`
(`index_uncheck`). Root cause: `reduce_axes_arg_cpu_serial`/`_cpu_rayon` raveled each
unraveled index through a `pseudo_layout` built from the **output** shape
(`layout_out`), but those indices live in the **reduced-axes** space (`layout_axes`).
For single-axis reduction the index is rank 1 while the output is rank `ndim - 1`,
so `index_uncheck` read past the 1-element index. Fixed: the `reduce_axes_unraveled_arg_*`
functions now return the axes layout (`layout_axes`) alongside the indices, and the arg
functions build `pseudo_layout` from `layout_axes.shape()` - the shape the indices
actually reference. This also keeps the rayon path correct, where `layout_axes` is
greedy-reordered before iteration. The parity test now asserts success (shape + values)
instead of `catch_unwind`; a parallel `custom_argmin::test_argmin_axes_high_rank` covers
the same shared code path.

## `reshape_f` on an overflowing/incompatible shape returns `Err` (FIXED)

- **numpy:** _core/tests/test_regression.py::TestRegression::test_reshape_size_overflow (L2275)
- **rstsr:** entry_row_cpu::core_func::manipulation::test_reshape::numpy_reshape::regression
- **tag:** bug
- **status:** fixed

NumPy raises `ValueError` when the shape product overflows (gh-7455). rstsr's fallible
`reshape_f` previously **panicked** instead of returning a clean `Err`, and the parity
test masked the panic with `catch_unwind`. Two unchecked sites combined:

1. The size product `shape_out.iter().product()` (`rstsr-common/src/layout/reshape.rs`,
   `quick_check`) was a plain `usize` multiply with no `checked_mul`. The gh-7455 factors
   multiply to `2**64 + 10`, so **in a release build the product wrapped to 10 ==
   `size_in`**, fooling the `size_in == size_out` mismatch check (and in a debug build
   this very multiply panicked on arithmetic overflow).
2. With the size check fooled in release, execution reached `attempt_nocopy_reshape`,
   which indexed `olddims[oj]` / `newdims[nj]` without bounds-checking against
   `oldnd` / `newnd`; `oj` ran past `oldnd` -> index-out-of-bounds panic.

Fixed: (a) `quick_check` computes the product with `try_fold`/`checked_mul` and returns
`Err(InvalidValue)` on overflow; (b) `attempt_nocopy_reshape` bounds-checks `oj`/`nj`
and returns `None` (fall through to copy) instead of panicking. The parity test now
asserts `reshape_f(new_shape).is_err()`.

## `stack` / `hstack` now accept 0-D inputs (FIXED)

- **numpy:** `_core/tests/test_shape_base.py::test_stack` (L463, 0d input);
  `TestHstack::test_0D_array` (L154)
- **rstsr:** entry_row_cpu::core_func::creation_from_tensor::test_stack::numpy_stack::test_0d_input;
  ::test_hstack::numpy_hstack::test_0d_array
- **tag:** bug
- **status:** fixed

rstsr `stack` previously required `ndim > 0` and `hstack` (via `concat`) errored on
0-D input, where NumPy accepts them (stack -> 1-D, hstack -> 1-D). Fixed: `stack` now
allows 0-D (its `expand_dims` path handles it), and `hstack` promotes inputs with
[`atleast_1d`] before concatenating. The parity tests now assert NumPy behavior.

## `vstack` now `atleast_2d`-promotes <2-D inputs (FIXED)

- **numpy:** `_core/tests/test_shape_base.py::TestVstack::test_1D_array` (L209);
  `TestVstack::test_0D_array` (L202); `TestVstack::test_2D_array2` (L223)
- **rstsr:** entry_row_cpu::core_func::creation_from_tensor::test_vstack::numpy_vstack::{test_1d_array, test_0d_array, test_2d_array2}
- **tag:** bug
- **status:** fixed

rstsr `vstack` previously concatenated along axis 0 directly (no promotion), so 1-D
inputs yielded a 1-D result (not 2-D) and 0-D errored - diverging from NumPy's
`atleast_2d` promotion. Fixed: `vstack` now promotes each input with [`atleast_2d`]
(0-D -> `(1, 1)`, 1-D `(N,)` -> `(1, N)`) before `concat` axis 0, matching NumPy.
[`atleast_1d`] / [`atleast_2d`] / [`atleast_3d`] were added as public view-returning
functions (NumPy `atleast_*`), implemented as `expand_dims` with the appropriate axes.

## `diag` on a non-square matrix: negative sub-diagonal range bug (FIXED)

- **numpy:** `lib/tests/test_twodim_base.py::TestDiag::test_diag_bounds` (L164)
- **rstsr:** entry_row_cpu::core_func::creation_from_tensor::test_diag::numpy_diag::test_diag_bounds
- **tag:** bug
- **status:** fixed

`Layout::diagonal` (`rstsr-common/src/layout/layoutbase.rs:351`) used the wrong
validity range for negative offsets: `(-d2+1..0)` (cols-based) instead of
`(-d1+1..0)` (rows-based). On a non-square matrix with more rows than cols, a
sub-diagonal offset beyond `-(cols-1)` was reported as empty instead of the correct
values - e.g. `diag([[1, 2], [3, 4], [5, 6]], k=-2)` returned `[]` instead of `[5]`
(`A[2, 0]`). Square matrices were unaffected (`d1 == d2`), which is why the square
`test_matrix` / `test_vector` cases passed while `test_diag_bounds` (3x2) failed. The
`d_diag` formula `(d1 - |offset|).min(d2)` was already correct; only the range check
was wrong. Found by the `test_diag_bounds` parity test; fixed by changing the range
to `(-d1+1..0)`.
