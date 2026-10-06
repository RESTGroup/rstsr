# NumPy Differences Report - Resolved (rstsr-core)

Fixed / resolved divergences, archived from `numpy_differences.md`. **Open**
divergences live in `numpy_differences.md`; see that file (and ADR-0003) for the
tags/format convention.

**Pinned NumPy:** v2.5.2 · **Checkout:** tag v2.5.2

Each entry here has `status: fixed`. Kept for history / regression context - the
parity test that surfaced it now asserts the correct NumPy behavior.

## `ne` / `not_equal` implemented (FIXED)

- **numpy:** `np.not_equal` (covered by `test_umath.py::TestComparisons`).
- **rstsr:** entry_row_cpu::core_func::operators::test_comparison::custom_comparison
- **tag:** bug
- **status:** fixed

The five elementwise comparisons `eq`/`lt`/`le`/`gt`/`ge` worked, but `rt::ne` /
`rt::not_equal` did not compile. The tensor-level `TensorNotEqualAPI` trait and
impls were already generated (they delegate to `B: OpNotEqualAPI<...>`), and the
`OpNotEqualAPI` device trait was defined, but the **device impl was missing** -
the comparison `#[duplicate_item]` macro in `op_ternary_common.rs` listed
`OpEqualAPI`/`OpGreaterAPI`/.../`OpLessEqualAPI` but omitted `OpNotEqualAPI`, so
the `B: OpNotEqualAPI` bound could never be satisfied. Fixed by adding
`[OpNotEqualAPI] [bool] [PartialEq] [a != b]` to the device macro for both
`DeviceCpuSerial` and `DeviceRayonAutoImpl` (the latter is aliased to
`DeviceFaer`, so faer is covered too). The parity test now uses `rt::ne(&a, &b)`
instead of the `not(eq)` workaround.

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

## `eye` under ColMajor returned the transposed shape (FIXED)

- **numpy:** `np.eye(N, M=None, k=0, order='C'/'F')` keeps shape `(N, M)`; only the storage order changes.
- **rstsr:** entry_row_cpu::doc_draft::creation::test_creation::doc_eye (col-major case)
- **tag:** col-major-transfer
- **status:** fixed

With a device whose default order is `ColMajor`, `rt::eye((n_rows, n_cols, k, &device))`
returned a tensor of shape `(n_cols, n_rows)` with F-contiguous layout: the shape
arguments were transposed, so the col-major result was a *different function* from the
row-major one (NumPy's `order='F'` only changes the memory order, never the shape).
Fixed in `EyeAPI::eye_f` (`rstsr-core/src/tensor/creation.rs`): the layout is now
`[n_rows, n_cols].f()` under ColMajor, so the logical content (shape `(n_rows, n_cols)`,
ones on the k-th diagonal) is identical under both orders and only the layout differs -
matching NumPy. The `eye` docstring and its `doc_eye` twin document/assert the
same-shape F-contiguous behavior. While verifying the neighborhood, `diag`/`diagonal`
were checked for the same class of issue and found correct: both route through
`Layout::diagonal`, which reads axis strides directly and is order-independent
(twins `doc_diag` / `doc_diagonal` now carry F-contiguous input/output cases).

## `meshgrid` `copy = false` now returns views (FIXED)

- **numpy:** `np.meshgrid(*xi, indexing=..., copy=False)` returns broadcast *views* sharing the inputs' memory.
- **rstsr:** entry_row_cpu::doc_draft::creation::test_creation::doc_meshgrid (copy = false case);
  core_func::creation_from_tensor::test_meshgrid::custom_meshgrid::test_copy_false_shares_memory
- **tag:** bug
- **status:** fixed

With `copy = false`, rstsr's `meshgrid` still returned owned tensors: each grid was
materialized by `into_shape_f` on a view (always an owned copy), then broadcast by
`broadcast_arrays_f` into owned stride-0 grids; the flag only skipped an extra
contiguity pass. Fixed by building each grid layout-only from its input: the input's
own stride is kept on its grid axis and all other axes get stride 0, so no data is
moved. Because a view cannot borrow from a consumed value, the reference-input
overloads (`Vec<&TensorAny>`, `[&TensorAny; N]`, ...) now return
`Vec<TensorCow<'a, T, B, IxD>>` - `copy = true` gives fresh owned contiguous grids
(as before), `copy = false` gives true NumPy-style views over the inputs' memory.
The by-value owned overloads (`Vec<Tensor>`, `[Tensor; N]`) keep returning
`Vec<Tensor<T, B, IxD>>`, whose `copy = false` grids are owned stride-0 tensors
aliasing the inputs' own storages (also removing the intermediate reshape copy the
old path made). `&Vec<TensorAny>` forms forward and convert into owned grids as
before. NumPy's `test_writeback` (L2851, `copy = True` grids are writable fresh
copies, inputs untouched) is now ported; the view-sharing case is covered by a
custom supplement. Note NumPy's `copy=False` grids are still read-only-shimmed in
rstsr (immutable views); writing through them requires `into_owned` first.

## `to_contig` no-copy check aligned with the NumPy-style flags (FIXED)

- **numpy:** `np.ascontiguousarray` uses the `C_CONTIGUOUS` flag, which ignores
  size-1 dimensions, so a padded-singleton contiguous array (e.g. shape `[3,1]`
  stride `[1,3]` sliced from an F-stored parent) is returned as a **view**.
- **rstsr:** entry_row_cpu::doc_draft::manipulation::test_to_contig::doc_to_contig::test_doc_padded_singleton
- **tag:** bug
- **status:** fixed

rstsr `to_contig` decided view-vs-copy by exact layout equality
(`to_layout.rs:20`), which was stricter than both NumPy's contiguity flag and
rstsr's own `c_contig()` (`layoutbase.rs:202`, which agrees with NumPy). A
padded-singleton contiguous tensor was therefore **copied** by rstsr but
**viewed** by NumPy; output values were identical, only ownership differed. Fixed
by maintainer decision: `change_contig_f` now decides the view path via
`c_contig()`/`f_contig()` (NumPy-style flags), and a viewed result has its
singleton-axis strides reset so the layout becomes the usual contiguous one over
the same elements. `to_prefer` already used the flags for its fast path and is
unchanged; the exact-equality check in `change_layout_f`/`to_layout` (explicit
target layout) is intentionally kept. The padded-singleton case is now covered by
a twin (`doc_to_contig::test_doc_padded_singleton`) and a docstring example.

## `broadcast_arrays` now returns views for reference inputs (FIXED)

- **numpy:** `np.broadcast_arrays` returns views sharing the inputs' memory.
- **rstsr:** entry_row_cpu::core_func::manipulation::test_broadcast::numpy_broadcast_arrays::test_broadcast_arrays_reference_inputs;
  doc_draft::manipulation::test_broadcast::doc_broadcast::doc_broadcast_arrays_views
- **tag:** intentional
- **status:** fixed

rstsr `broadcast_arrays` only accepted consumed tensors
(`Vec<TensorAny>`) and returned owned stride-0 tensors aliasing the inputs'
storages; obtaining views required hand-building a vector of views first (the
docstring even said so). By maintainer decision, reference-input overloads were
added: `Vec<&'a TensorAny>` (also `&Vec<...>` and `[&TensorAny; N]`) now return
`Vec<TensorView<'a, T, B, IxD>>` - broadcast views sharing the inputs' memory, as
in NumPy. `TensorView` was chosen over `TensorCow` because the function has no
copy flag: the reference-input result is always a view, so the `Cow` owned branch
would be unreachable (`Cow` remains right for `meshgrid`, whose `copy` flag
switches at runtime). The by-value form keeps its previous behavior (consumed
inputs, owned stride-0 outputs, zero copy).
## `round` was round-half-away-from-zero, not ties-to-even (FIXED)

- **numpy:** `np.round` / `np.rint` resolve halfway cases to the even integer
  (`test_umath.py::TestRoundingFunctions`); the array-api elementwise `round`
  requires the same ("the even integer closest to `x_i`").
- **rstsr:** entry_row_cpu::core_func::math::test_unary_math::custom_math_basic::test_round_ties_to_even
- **tag:** bug
- **status:** fixed

`OpRoundAPI`'s device kernels called `f64::round`, which is
round-half-away-from-zero. Discovered 2026-10-06 through the rstsr-faer-py
conformance suite (`test_special_cases.py::test_unary[round ...]` drew an exact
halfway example that earlier runs had not). Both kernel tables
(`device_cpu_serial/operators/op_binary_common.rs`,
`feature_rayon/auto_impl/op_binary_common.rs`) now use an IEEE
`roundToIntegralTiesToEven` helper (`round_ties_even_f`, exact for `f32` via an
`f64` round trip); NaN/inf and signed zeros propagate per IEEE.
## `triu` indexed past the row when the diagonal left the matrix (FIXED)

- **numpy:** `np.triu` zeroes `j < i + k` clipped to the row: `np.triu(ones((3, 3)), 2)` is
  `[[0, 0, 1], [0, 0, 0], [0, 0, 0]]`, and a `k` beyond the matrix zeroes (or keeps) everything
  (`lib/tests/test_twodim_base.py::test_tril_triu_ndim2` covers the in-range cases).
- **rstsr:** entry_row_cpu::core_func::creation::test_tril_triu::custom_tril_triu::test_k_outside_row_bounds
  (+ `::test_extreme_k`)
- **tag:** bug
- **status:** fixed

`triu_ix2_cpu_serial` computed `j_end = max(i + k, 0)` but never clamped it to `ncol`, so any
element of the k-th diagonal outside the matrix (e.g. `k >= ncol`, or `M > N` with a negative
`k`) indexed past the buffer and panicked. Discovered 2026-10-06 through the rstsr-faer-py
conformance suite (`test_triu` draws `k` over `[-max(n, m), max(n, m)]`). The same kernels now
clamp to the row and use saturating arithmetic, so a `k` at the `isize` bounds cannot overflow
either.
## `linspace` endpoint was not exact and the serial kernel accumulated (FIXED)

- **numpy:** `np.linspace` includes `stop` exactly when `endpoint=True` and computes
  `y[i] = start + i * step` (`_core/tests/test_function_base.py::TestLinspace`).
- **rstsr:** entry_row_cpu::core_func::creation::test_linspace::custom_linspace::test_endpoint_exact
- **tag:** bug
- **status:** fixed

Both kernels left the last value at `start + (n - 1) * step` — off by one ulp from `stop` on
some inputs (`linspace(0, 6.4913965932284536e16, 25)`) — and the serial kernel accumulated
`v += step`, drifting up to a few ulp through the interior (`linspace(2, 10, 100)[-1]` was
`9.999999999999996`, and `linspace(0, 1, 11)[8]` was `0.7999999999999999`). Discovered
2026-10-06 through the rstsr-faer-py conformance suite (`test_linspace` asserts
`out[-1] == stop` exactly). Both kernels now compute `start + i * step` and assign the
endpoint directly when `endpoint=True`; `endpoint=False` keeps the half-open interval.
The result matches NumPy bit-for-bit for float64; for float32/complex the arithmetic runs
in the output dtype, so values may differ from NumPy by a few ulp
(`linspace(0, 1, 11, dtype=float32)[9]` is one ulp above NumPy's `0.9`).

## `Layout::diagonal` gated super-diagonals on the row count (FIXED)

- **numpy:** `np.eye(2, 4, k=2)` puts ones at `(0, 2)` and `(1, 3)`; a wide matrix's
  diagonal with `offset >= n_rows` still has elements.
- **rstsr:** rstsr-common `layout::test::test_diagonal` +
  entry_row_cpu::core_func::creation::test_eye::custom_eye::test_eye_offset_past_rows
- **tag:** bug
- **status:** fixed

`Layout::diagonal`'s super-diagonal branch tested `(0..d1)` (rows) where a super-diagonal
`k` is valid while `k < d2` (cols). `xp.eye(2, 4, k=2)` therefore produced an all-zero
matrix, and `xp.eye(3, 1, k=2)` a bogus "Layout is too large" error. Found 2026-10-06 by
the rstsr-faer-py review; the gate is now `(0..d2)`, which also keeps the `(d2 - k)`
length non-negative for every accepted offset. `rt::diagonal` consumers (diag, indexing)
are fixed by the same change.

## `squeeze` validated only the first axis of a descending sort (FIXED)

- **numpy:** `np.squeeze(zeros((1, 2, 1)), axis=(-4, 0))` raises `AxisError` (-4 out of
  bounds for a 3-d array).
- **rstsr:** entry_row_cpu::core_func::manipulation::test_squeeze::custom_squeeze_mixed_axes::test_mixed_invalid_negative_axis
- **tag:** bug
- **status:** fixed

`into_squeeze_f` mapped negative axes, sorted descending, then checked only `axes.first()`
— so a mixed list like `(-4, 0)` kept the invalid `-1` (which addressed a real axis after
the first elimination) and succeeded. Found 2026-10-06 through the rstsr-faer-py review;
the check now scans every mapped axis.

## `take` rejected empty indices on an empty axis (FIXED)

- **numpy:** `np.take(zeros((0,)), array([], dtype=int64))` returns shape `(0,)`.
- **rstsr:** entry_row_cpu::core_func::indexing::test_indexing::custom_indexing_take::test_take_empty_indices
- **tag:** bug
- **status:** fixed

Both `index_select` kernels bounds-checked `indices.iter().max().unwrap_or(&0)`, testing the
sentinel `0` against an empty axis (`0..0`). Found 2026-10-06 through the rstsr-faer-py
review; the check is skipped when `indices` is empty.

## `argmax`/`argmin` rejected an empty *output* (FIXED)

- **numpy:** `np.argmax(zeros((2, 0)), axis=0)` returns an empty array of shape `(0,)`; an
  empty *reduced* axis still raises.
- **rstsr:** entry_row_cpu::core_func::reduction::test_argmax::custom_arg_empty::test_empty_output
- **tag:** bug
- **status:** fixed

The axes-reduction kernels asserted `la.size() > 0` before splitting, conflating "nothing to
reduce" with "nothing to produce". Found 2026-10-06 through the rstsr-faer-py review; the
guard now tests the split reduced layout, so an empty output is legal while an empty reduced
axis still raises "empty sequence is not allowed for reduce_arg".

## `tril`/`triu` rank-1 input surfaced a bare AxisError (FIXED)

- **numpy:** n/a (array-API defines `x` as `(..., M, N)`; the reference implementation raises
  for rank < 2).
- **rstsr:** entry_row_cpu::core_func::creation::test_tril_triu::custom_tril_triu::test_ndim1_error
- **tag:** bug
- **status:** fixed

`tril_cpu_serial`/`triu_cpu_serial` reached `dim_split_at(-2)` first, surfacing
`AxisError { axis: -2, ndim: 1 }` (mapped to `IndexError` by the array-API wrapper). Found
2026-10-06 through the rstsr-faer-py review; the kernels now assert `ndim >= 2` up front.
