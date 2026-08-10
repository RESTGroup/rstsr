# NumPy Differences Report (rstsr-core)

Where rstsr diverges from NumPy. Distilled from the parity tests in `core_func/` and
the coverage checklist `numpy_coverage.csv`. See ADR-0003.

**Pinned NumPy:** v2.5.2 · **Checkout:** tag v2.5.2

This file holds **open** divergences. Fixed/resolved divergences are archived in
`numpy_differences_resolved.md`.

## Tags

- `intentional` - rstsr deliberately differs (ownership semantics, no `out=`, trait
  dispatch, col-major convention). Not a bug.
- `bug` - rstsr differs unintentionally. File an issue; link it here.
- `col-major-transfer` - difficulty mapping row-major NumPy behavior to rstsr's
  column-major convention. (Col-major tests live in `tests/col_major/`, deferred.)

## Format

One section per divergence. Cite the NumPy identifier and the rstsr test:

```
## <short title>

- **numpy:** _core/tests/test_multiarray.py::TestMethods::test_<x> (L<n>)
- **rstsr:** entry_row_cpu::core_func::<category>::test_<func>::numpy_<func>::<case>
- **tag:** intentional | bug | col-major-transfer
- **status:** open | fixed (issue #<n>) | wontfix

<what differs and why>
```

<!-- Entries below. Append new divergences here as parity tests are authored.
     When a divergence is fixed, move it to numpy_differences_resolved.md. -->

## Reshape on an overflowing/incompatible shape panics instead of returning `Err`

- **numpy:** _core/tests/test_regression.py::TestRegression::test_reshape_size_overflow (L2275)
- **rstsr:** entry_row_cpu::core_func::manipulation::test_reshape::numpy_reshape::regression
- **tag:** bug
- **status:** open

NumPy raises `ValueError` when the shape product overflows (gh-7455). rstsr's fallible
`reshape_f` does **not** return a clean `Err` for this case - it **panics**, and the
parity test masks the panic with `catch_unwind` (its own comment admits "panic occurs
on rust-side, not from RSTSR (i.e., not coverable)"). Two unchecked sites combine:

1. The size product `shape_out.iter().product()`
   (`rstsr-common/src/layout/reshape.rs:157`) is a plain `usize` multiply with no
   `checked_mul`. The gh-7455 factors multiply to `2**64 + 10`, so **in a release build
   the product wraps to 10 == `size_in`** - the `size_in == size_out` mismatch check is
   fooled and the overflow goes undetected. (In a debug build this very multiply panics
   on arithmetic overflow, which is the panic the test currently "catches".)
2. With the size check fooled in release, execution reaches `attempt_nocopy_reshape`,
   which indexes `olddims[oj]` / `newdims[nj]` without bounds-checking against
   `oldnd` / `newnd`; for this incompatible shape `oj` runs past `oldnd` and the index
   is out of bounds -> **panic**.

Net: `reshape_f` panics (debug: overflow panic; release: index-OOB panic) where NumPy
raises `ValueError`. Verified empirically - the test passes in **both** profiles, but
for different panic reasons, never via a clean `Err`. Fix: (a) compute the product with
`checked_mul` and return `Err(InvalidValue)` on overflow; (b) bounds-check `oj`/`nj`
in `attempt_nocopy_reshape` and return `None` (fall through to copy) instead of
panicking. Then the test should assert `reshape_f(new_shape).is_err()`, not
`catch_unwind`.

## Default order is `device.default_order()`, not C

- **numpy:** `np.reshape` / `np.ravel` default `order='C'` (C-order).
- **rstsr:** core_func::manipulation::test_reshape (all `numpy_reshape` cases)
- **tag:** intentional
- **status:** open

rstsr reshape/ravel default to `device.default_order()`, so on a `ColMajor`-default
device a plain `reshape(shape)` diverges from NumPy. All parity tests pin
`device.set_default_order(RowMajor)` first to match NumPy. Documented in the
reshape docstring's Row/Column Major Notice.

## `order='A'` / `order='K'` unsupported

- **numpy:** `_core/tests/test_multiarray.py::TestMethods::test_ravel` (L4088) exercises
  all four orders C/F/A/K.
- **rstsr:** reshape/ravel accept only `RowMajor` / `ColMajor`.
- **tag:** intentional
- **status:** open

rstsr has no `'A'` (any) or `'K'` (keep) order concept. The A/K cases of
`test_ravel` are therefore not-applicable; the C/F value cases are covered by
`numpy_reshape::test_ravel` (see `numpy_coverage.csv`, status `transferred`).

## `flatten()` always-copies vs `reshape(-1)` view-when-possible

- **numpy:** `_core/tests/test_multiarray.py::TestMethods::test_flatten` (L3717);
  `np.flatten()` returns a copy.
- **rstsr:** `flatten` is folded into `reshape(-1)` (`docs/numpy-cheatsheet.mdx`), which
  returns a view (Cow/Ref) when layout-compatible.
- **tag:** intentional
- **status:** open

rstsr's `reshape(-1)` matches NumPy's `ravel` (view when possible), not `flatten`
(always copy). No `flatten` API exists; the equivalence is documented in the
cheatsheet and exercised by `numpy_reshape::test_flatten` (C/F value cases via
`reshape(-1)`). Value results match for all orders rstsr supports.

## Error taxonomy: unified `InvalidValue` vs NumPy's `AxisError`/`ValueError` split

- **numpy:** transpose (`test_transpose` L2260, wrong axis count -> `ValueError`),
  swapaxes (`test_swapaxes` L4205, OOB -> `AxisError`), moveaxis (`test_errors` L3937:
  `AxisError` for OOB source/destination; `ValueError` for duplicates / length mismatch),
  squeeze/expand_dims/flip (similar split).
- **rstsr:** core_func::manipulation::{test_transpose, test_moveaxis, test_squeeze,
  test_expand_dims, test_flip}
- **tag:** intentional
- **status:** open

rstsr raises a single error kind per operation family (`InvalidValue` for moveaxis/
squeeze/expand_dims; `InvalidLayout` for transpose axis-count; `ValueOutOfRange` for
swapaxes OOB) rather than NumPy's `AxisError`-vs-`ValueError` distinction. All parity
tests assert only `.is_err()`, so coverage is unaffected, but error-kind parity is lost.
Error messages also differ (e.g. `"Duplicate axes are not allowed."` vs NumPy
`'repeated axis in source'`). Acceptable for a Rust `Result`-based API.

## Strides are element-unit, not byte-unit

- **numpy:** strides reported in bytes (e.g. int32 0-d->(1,1) reshape yields `(4,4)`).
- **rstsr:** `rstsr-common` layouts use element strides (the same case yields `[1,1]`).
- **tag:** intentional
- **status:** open

Behaviorally equivalent when scaled by dtype size. rstsr's `attempt_nocopy_reshape`
comment ("Assuming element size of 1") is correct *because* rstsr uses element strides.
Parity tests assert element-unit strides.

## Negative shapes / strides unsupported

- **numpy:** `test_broadcast_to_raises` (L268) includes negative-shape ->
  negative-stride readonly-view cases.
- **rstsr:** core_func::manipulation::test_broadcast::numpy_broadcast_to::test_broadcast_to_raises
- **tag:** intentional
- **status:** open

rstsr dimensions are `usize`; there are no negative shapes or strides. The 3
negative-shape error cases are skipped in the parity test (with an explicit comment).

## ColMajor broadcast applies from the left (rstsr extension)

- **numpy:** broadcast is strictly row-major (rules applied from the right).
- **rstsr:** `broadcast_shapes` / `broadcast_to` take an explicit `order`; in `ColMajor`
  the broadcast rules apply from the left.
- **tag:** intentional
- **status:** open

NumPy has no `order` parameter on broadcast. rstsr's ColMajor broadcast is an rstsr
extension exercised by the rstsr-only `test_broadcast_shapes_col_major` case. Row-major
behavior matches NumPy exactly.

## `broadcast_arrays` returns owned stride-0 tensors, not writeable views

- **numpy:** `broadcast_arrays` returns writeable views.
- **rstsr:** core_func::manipulation::test_broadcast::numpy_broadcast_arrays
- **tag:** intentional
- **status:** open

rstsr `broadcast_arrays` takes ownership and returns owned `TensorAny` tensors with
stride-0 axes (writeable but dangerous, as the docs warn), vs NumPy's writeable views.
Semantically aligned (both "writeable but dangerous"); the API shape differs.

## `broadcast_shapes` signature takes `&[IxD], order`, not varargs

- **numpy:** `np.broadcast_shapes(*shapes)` varargs.
- **rstsr:** `broadcast_shapes(&[IxD], order)` with an explicit order argument.
- **tag:** intentional
- **status:** open

API-shape difference; results are identical for the row-major cases.

## `to_contig` no-copy check is exact-layout-equality (stricter than NumPy flags)

- **numpy:** `np.ascontiguousarray` uses the C_CONTIGUOUS flag, which ignores
  size-1 dimensions, so a padded-singleton C-contiguous array (e.g. shape `[3,1]`
  stride `[1,5]`) is returned as a **view**.
- **rstsr:** core_func::manipulation::test_to_contig (custom)
- **tag:** intentional
- **status:** open

rstsr `to_contig` decides view-vs-copy by exact layout equality
(`to_layout.rs:20`), which is stricter than both NumPy's contiguity flag and rstsr's
own `c_contig()` (`layoutbase.rs:202`, which agrees with NumPy). A
padded-singleton C-contiguous tensor is therefore **copied** by rstsr but **viewed**
by NumPy. Output values are identical; only ownership differs. No existing test
constructs a padded-singleton case, so this is currently untested. Worth either
documenting or aligning `to_contig` with `c_contig()`.

## `np.flip(a)` default `axis=None` vs rstsr explicit-`None` argument

- **numpy:** `lib/tests/test_function_base.py::TestFlip::test_default_axis` (L234);
  `np.flip(a)` has an implicit `axis=None` default.
- **rstsr:** core_func::manipulation::test_flip::numpy_flip::test_default_axis
- **tag:** intentional
- **status:** open

rstsr `flip(tensor, axes)` requires an explicit `None` to flip all axes; there is no
default. Behavior is identical when `None` is passed (the parity test does so).

## `concat` has no `axis=None` (flatten-concat) mode

- **numpy:** `_core/tests/test_shape_base.py::TestConcatenate::test_concatenate_axis_None` (L311)
- **rstsr:** entry_row_cpu::core_func::creation_from_tensor::test_concat::numpy_concatenate
- **tag:** intentional
- **status:** open

NumPy `concatenate(..., axis=None)` flattens all inputs and concatenates into 1-D.
rstsr `concat` takes an explicit integer axis only - there is no `axis=None` mode.
The `axis=None` cases are not-applicable; to flatten-and-concat, chain
`reshape(-1)` / `concat` manually.

## `meshgrid` has no `sparse=` parameter

- **numpy:** `lib/tests/test_function_base.py::TestMeshgrid::test_sparse` (L2809)
- **rstsr:** (no rstsr equivalent)
- **tag:** intentional
- **status:** open

NumPy `meshgrid(..., sparse=True)` returns stride-0 broadcasted views. rstsr `meshgrid`
has no `sparse` parameter; it always returns dense broadcasts.

## `meshgrid` is homogeneous-dtype (no per-input dtype preservation)

- **numpy:** `lib/tests/test_function_base.py::TestMeshgrid::test_return_type` (L2827)
- **rstsr:** (no rstsr equivalent)
- **tag:** intentional
- **status:** open

NumPy `meshgrid` preserves each input's dtype (x=f32 -> X=f32, y=f64 -> Y=f64). rstsr
`meshgrid` is generic over a single `T`; all inputs must share one dtype. The
mixed-dtype `test_return_type` case is therefore not-applicable.

## `unstack` returns `Vec`, not a tuple

- **numpy:** `_core/tests/test_shape_base.py::test_unstack` (L531)
- **rstsr:** entry_row_cpu::core_func::creation_from_tensor::test_unstack::numpy_unstack::test_unstack
- **tag:** intentional
- **status:** open

API-shape difference; values match. rstsr `unstack` returns `Vec<TensorView>`; NumPy
returns a tuple.
