# NumPy Differences Report - Resolved (rstsr-core)

Fixed / resolved divergences, archived from `numpy_differences.md`. **Open**
divergences live in `numpy_differences.md`; see that file (and ADR-0003) for the
tags/format convention.

**Pinned NumPy:** v2.5.2 · **Checkout:** tag v2.5.2

Each entry here has `status: fixed`. Kept for history / regression context - the
parity test that surfaced it now asserts the correct NumPy behavior.

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
