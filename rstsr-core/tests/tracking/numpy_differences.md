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

## Reductions have no `keepdims` parameter

- **numpy:** `_core/tests/test_numeric.py::TestNonarrayArgs::test_sum` (L320) uses
  `np.sum(m, axis=1, keepdims=True)`; reductions across NumPy support `keepdims=`.
- **rstsr:** entry_row_cpu::core_func::reduction::test_sum::numpy_sum::test_numeric
- **tag:** intentional
- **status:** open

rstsr `sum_axes`/`mean_axes`/etc. always **drop** the reduced axes (output rank =
input rank − #axes); there is no `keepdims` argument. The `None`-axis form
(`xxx_axes(None)`) reduces all axes to a 0-d tensor. Parity tests assert the
axis-dropped result; the NumPy `keepdims` shape is reached by a follow-up
`expand_dims`/`reshape` if needed. Reduction **values** match NumPy exactly.

## Statistical reductions require a `Float` input (no int→float promotion)

- **numpy:** `test_mean`/`test_std`/`test_var` (TestNonarrayArgs L142/303/360) call
  `np.mean/std/var` on integer lists; NumPy promotes int → float internally.
- **rstsr:** entry_row_cpu::core_func::reduction::{test_mean,test_std,test_var}
- **tag:** intentional
- **status:** open

rstsr `mean`/`std`/`var` require the element type to satisfy `num::Float +
FloatConst`; an integer tensor does not compile. Parity tests therefore build the
input as `f64` (e.g. `[[1.0, 2.0, 3.0], ...]`) rather than `i32`. Output values
match NumPy (population statistics, ddof = 0). `sum`/`prod`/`min`/`max`/`argmin`/
`argmax` do accept integers.

## `all`/`any` require a `bool` tensor (NumPy accepts truthy int)

- **numpy:** `lib/tests/test_function_base.py::TestAll::test_basic` (L283) /
  `TestAny::test_basic` (L266) pass Python int lists (`[0, 1, 1, 0]`), treating
  nonzero as True.
- **rstsr:** entry_row_cpu::core_func::reduction::{test_all,test_any}
- **tag:** intentional
- **status:** open

rstsr `all`/`any` operate on `Tensor<bool>`; there is no implicit truthiness for
integer tensors. Parity tests use bool tensors (`[false, true, true, false]`).
Also, `bool` result tensors cannot be compared with `assert_equal` (`bool:
ExtNum` unsatisfied), so the axes results are compared via `to_vec()`.

## `linspace` requires an explicit `num` (NumPy defaults to 50)

- **numpy:** `_core/tests/test_function_base.py::TestLinspace::test_basic` (L322)
  calls `linspace(0, 10)` with no `num`.
- **rstsr:** entry_row_cpu::core_func::creation::test_linspace::numpy_linspace::test_basic
- **tag:** intentional
- **status:** open

rstsr `linspace` has no `num` default — the call forms are
`(start, stop, num, &device)` and `(start, stop, num, endpoint, &device)`. Parity
tests pass `num` explicitly. Consequently `linspace(0, 10, num=-1)` (NumPy raises
`ValueError`) is not expressible — `num` is `usize`, so a negative count is a
compile-time type error rather than a runtime error. Output values match NumPy.

## `ne` / `not_equal` is unimplemented

- **numpy:** `np.not_equal` (covered by `test_umath.py::TestComparisons`).
- **rstsr:** entry_row_cpu::core_func::operators::test_comparison::custom_comparison
- **tag:** bug
- **status:** open

The other five elementwise comparisons (`eq`, `lt`, `le`, `gt`, `ge` — free
functions `rt::eq/lt/le/gt/ge`) compile and work on `&Tensor`. `rt::ne` /
`rt::not_equal` does **not compile**: the `TensorNotEqualAPI` / `OpNotEqualAPI`
auto-impl is missing (no impl for `&Tensor`, owned `Tensor`, or
`DeviceCpuSerial`). The parity test derives `ne = not(eq)` for the value check.
When implemented, replace that with `rt::ne(&a, &b)`.

## The `%` (`Rem`) operator returns garbage; `rt::rem` is correct

- **numpy:** `np.remainder` / Python `%`.
- **rstsr:** entry_row_cpu::core_func::operators::test_arithmetic::custom_rem
- **tag:** bug
- **status:** open

The free function `rt::rem(&a, &b)` returns the correct remainder for both 1-D
and 2-D integer tensors (`[1, 1, 3, 2]` / `[[1, 1], [3, 2]]`). The overloaded
`%` operator (`&a % &b`) returns **garbage** — e.g. `[135, 187, 319, 440]`
instead of `[1, 1, 3, 2]` for the same inputs — so the `Rem` tensor impl is
wired incorrectly (the `+ - * /` operators are all correct). The parity test
asserts `rt::rem` only and does not assert the `%` operator.

## `sign(0.0)` returns NaN instead of 0

- **numpy:** `np.sign(0.0) == 0` (and `np.sign` works on integers).
- **rstsr:** entry_row_cpu::core_func::math::test_unary_math::custom_math_basic
- **tag:** bug
- **status:** open

rstsr `rt::sign(0.0)` returns **NaN** (likely an `x / x.abs()`-style impl that
yields `0 / 0`); `sign` of nonzero values is correct (`sign([-2, 3]) == [-1, 1]`).
rstsr `sign` also requires a `Float` input (NumPy `sign` accepts integers). The
parity test avoids `0.0`; when the bug is fixed, restore a `0.0` case asserting
`sign(0.0) == 0.0`.

## `Tensor::i(int...)` reducing to a 0-d scalar returns the wrong element

- **numpy:** `_core/tests/test_indexing.py::TestIndexing::test_single_int_index`
  (L201) — `np.arange(10)[-1] == 9`.
- **rstsr:** entry_row_cpu::core_func::indexing::test_indexing::numpy_indexing::test_single_int_index
- **tag:** bug
- **status:** open

Integer indexing via `Tensor::i` that fully reduces the result to a **0-d**
(scalar) view returns the wrong element — it reads at offset 0, so e.g.
`arange(10).i(9).to_scalar() == 0` (not 9) and `m.i((1, 2)).to_scalar() == 0`
(not 6). Sub-tensor indexing (integer that drops one axis but leaves rank ≥ 1,
slices, `..`, `Ellipsis`, `None`/newaxis) reads correctly, as does 1-D
`index_select` gather. The parity test verifies element values via 1-element
slices (e.g. `a.i(slice!(9, None)) == [9]`) instead of the 0-d scalar form.
