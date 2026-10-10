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

## `broadcast_shapes` signature takes `&[IxD], order`, not varargs

- **numpy:** `np.broadcast_shapes(*shapes)` varargs.
- **rstsr:** `broadcast_shapes(&[IxD], order)` with an explicit order argument.
- **tag:** intentional
- **status:** open

API-shape difference; results are identical for the row-major cases.

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

## The `%` operator is matrix multiplication, not remainder

- **numpy:** `np.remainder` / Python `%` (elementwise remainder).
- **rstsr:** entry_row_cpu::core_func::operators::test_arithmetic::custom_rem
- **tag:** intentional
- **status:** open

In RSTSR the `%` (`Rem`) operator is bound to **matrix multiplication**
(`a % b == a.matmul(b)`), not elementwise remainder. The elementwise
`rem`→`Rem` binding is deliberately commented out in the core-ops module
(`op_binary_arithmetic.rs`), so the matmul `Rem` impl (`linalg/matmul.rs`)
applies instead. For example, with `a = [[10, 21], [33, 44]]` and
`b = [[3, 4], [5, 7]]`, `a % b` returns the matrix product `[[135, 187], [319, 440]]`
(not the elementwise remainder `[[1, 1], [3, 2]]`). The free function
`rt::rem(&a, &b)` provides the NumPy-compatible elementwise remainder; the
parity test asserts both `rt::rem` (remainder) and `a % b` (matmul) accordingly.

## `cumulative_sum`/`cumulative_prod` require `axis` for n-D input (no flatten-on-`None`)

- **numpy:** `np.cumsum(a)` / `np.cumprod(a)` with `axis=None` (the default) flatten the input
  first and return a 1-D cumulative result for any `ndim` (`test_function_base.py` `TestCumsum`
  relies on this legacy spelling).
- **rstsr:** entry_row_cpu::core_func::reduction::test_cumulative::custom_cumulative::test_axis_none_contract
- **tag:** intentional
- **status:** open

RSTSR follows the array-api-aligned `numpy.cumulative_sum` / `numpy.cumulative_prod`
(NumPy >= 2.1): `axis = None` is only valid for 1-D input, and an n-D input without an
explicit axis raises `InvalidValue` instead of flattening. Pass `axis` explicitly
(`rt::cumulative_sum(&x, 0)`) for n-D inputs; the legacy flatten-then-scan is a
caller-side `rt::cumulative_sum(&x.reshape([-1]), 0)`.

## `cumulative_sum`/`cumulative_prod` keep the input dtype (no platform-integer widening)

- **numpy:** `np.cumsum`/`np.cumprod` (and array-api `cumulative_sum`) accumulate narrow integer
  inputs (`int8`/`uint8`/`int16`/...) in the platform default integer (`int64`/`uint64`) when
  `dtype` is not given; `test_function_base.py::TestCumsum::test_basic` passes with small dtypes
  precisely because the accumulator is widened.
- **rstsr:** entry_row_cpu::core_func::reduction::test_cumulative::custom_cumulative::test_with_dtype
- **tag:** intentional
- **status:** open

RSTSR is strongly typed: with no explicit dtype the scan accumulates in `T` itself, so a
`u8` input that overflows panics in debug builds instead of silently widening. The
anti-overflow use case is served by the `*_with_dtype` variants (array-api `dtype=`):
`x.cumulative_sum_with_dtype::<i64>(args)` casts each element into the accumulator inside
the scan (no materialized cast copy).
## `where` condition must be a boolean tensor (no truthiness)

- **numpy:** `_core/tests/test_multiarray.py::TestWhere::test_dtype_mix` (L10016)
  accepts non-bool conditions (`c.astype(int)`, values like 34242324) via
  truthiness; the iterator casts any operand dtype to `NPY_BOOL`.
- **rstsr:** `rt::where` requires `TensorAny<R, bool, B, D>` for the condition;
  translate NumPy int masks with an explicit `ne(mask, 0)` (see
  `core_func::operators::test_where::numpy_where::test_dtype_mix`).
- **tag:** intentional
- **status:** open

Matches the array API standard ("condition should have a boolean data type")
and rstsr's typed-dtype design; no numeric-to-bool coercion exists elsewhere in
rstsr either.

## Scalar arguments promote like tensors (no NEP 50 weak scalars)

- **numpy:** `_core/tests/test_multiarray.py::TestWhere::test_exotic` (L9959)
  pins NEP 50 weak Python scalars: `np.where(True, float32_arr, float('nan'))`
  stays float32, `1e150` overflows with a warning but stays float32,
  `test_scalar_overflow` raises OverflowError for out-of-range Python ints.
- **rstsr:** `rt::where` scalar x/y follow the house rule of other elementwise
  functions (e.g. `maximum`): scalars are strong and promote (`f32` tensor +
  `0.5` -> `f64`); there is no value-based dtype minimization and no
  OverflowError analog. The NaN/inf minimality and overflow cases of
  `test_exotic` are not translated.
- **tag:** intentional
- **status:** open

Library-wide divergence (applies to all elementwise scalars), recorded here
because `where` is the first select-family function with scalar overloads.

## `where` has no one-argument form (indices/nonzero)

- **numpy:** `np.where(condition)` with a single argument returns the indices
  of truthy elements (equivalent to `np.nonzero`) - see
  `_core/tests/test_multiarray.py::TestWhere::test_empty_result` (L10086) /
  `test_largedim` (L10094).
- **rstsr:** `rt::where` provides only the three-argument select form
  `where(cond, x, y)`; the one-argument indices form is out of scope.
- **tag:** intentional
- **status:** open

API-surface scope decision: index retrieval should be served by dedicated
nonzero/argwhere-style APIs rather than an overloaded `where` (none wired yet).

## Basic indexing surface: no inclusive ranges, `Some(n)` panics

- **numpy:** bracket syntax accepts inclusive bounds (`x[1..=4]` -> `slice(1, 5)`),
  any tuple arity, and any expression in indexer position.
- **rstsr:** basic slicing (`tensor.i(index)`) accepts exclusive Rust ranges only
  (`1..5`, no `..=`), whole-index tuples of arity 1-10 plus a bare single indexer,
  and `None` as the only meaningful `Option` indexer: `Some(n)` compiles but
  always panics. (One-element tuples were unsupported until 2026-10-06; adding
  `TryFrom<(F1,)>` in `rstsr-common/src/axis_index.rs` resolved that part.)
  Verified NumPy-parity of the *semantics* (negative
  bounds, clamping, integer axis removal, `None`/`Ellipsis` placement) in
  `doc_draft::indexing::doc_slice*`.
- **tag:** intentional
- **status:** open

Surface-level consequence of Rust argument syntax instead of `[]`-bracket
indexing; documented inline on the `slice` anchor (src/docs/basic_indexing.md).

## Sort/argsort decline complex dtypes

- **numpy:** `np.sort` / `np.argsort` accept complex arrays; NumPy orders them
  lexicographically (real part first, then imaginary), NaN components last
  (`_core/tests/test_multiarray.py::TestMethods::test_sort_complex`).
- **rstsr:** `rt::sort` / `rt::argsort` raise `UnImplemented` for `Complex<f32>` /
  `Complex<f64>` (tensor-layer gate `decline_complex_sort`); the comparator
  variants `sort_custom`/`argsort_custom` with `ExtSortCmp` (lexicographic,
  NaN-part-last) remain available.
- **tag:** intentional
- **status:** open

The Python array-API standard restricts sort/argsort to real-valued data types
("Should have a real-valued data type"), so the decline is spec-aligned; the
comparator is implemented in `rstsr-dtype-traits/src/ext_sort_cmp.rs` should a
complex surface ever be wanted.

## `np.sort` `kind=` stability knob absent

- **numpy:** `np.sort(x, kind='stable'|'quicksort'|'heapsort'|'mergesort')` selects
  the algorithm; `stable=` maps onto it.
- **rstsr:** `SortArgs.stable` is honored as a requirement flag, but the kernels are
  always the stable merge-sort lines; `stable = false` does not select an
  unstable algorithm (documented on `SortArgs`).
- **tag:** intentional
- **status:** open

Single-algorithm implementation; a selection knob is a registered follow-up.

## `diff` `prepend`/`append` must match shape (no scalar expansion)

- **numpy:** a scalar `prepend`/`append` is expanded to length 1 along `axis`
  and to `x`'s shape on the other axes
  (`np.diff(np.arange(4).reshape(2, 2), axis=1, prepend=0)` → `[[0, 1], [2, 1]]`);
  otherwise the shape must match `x` except along `axis`.
- **rstsr:** `rt::diff` takes `Option<&TensorAny>`, so both sides must be
  tensors on the same device with the same `ndim` and the same shape as `x`
  outside `axis`. Callers must materialize a length-1 tensor explicitly
  (`x.full(1)`-style) to reach the scalar case.
- **tag:** intentional
- **status:** open

Typed `&TensorAny` signature keeps the device/dtype consistent; scalar
expansion is not performed (the array-api standard has no `prepend`/`append`
parameters at all). `bool` input is also N/A: `rt::diff` requires
`T: Sub<Output = T>`, and Rust's `bool` has no `Sub` (NumPy diffs bool arrays
as XOR).

## `unique_*` value order and NaN handling vs NumPy

- **numpy:** `np.unique` returns values in ascending (sorted) order and collapses
  NaNs into a single trailing entry. The array-api aliases (`np.unique_values`,
  `np.unique_counts`, `np.unique_inverse`, `np.unique_all`) pass
  `equal_nan=False` — so NaNs stay distinct — and, since NumPy 2.3, do not
  guarantee any order.
- **rstsr:** `rt::unique_values`/`unique_counts`/`unique_inverse`/`unique_all`
  return ascending order for orderable scalar dtypes (bool, integers, real
  floats — the `ExtSortCmp` fast path), but **first-occurrence order** over the
  device-default-order visit sequence for other dtypes (complex via the naive
  path). NaNs are kept as **distinct entries**.
- **tag:** intentional
- **status:** open

rstsr targets the array-api aliases: distinct NaNs is parity with them
(`equal_nan=False`; `np.unique_all([nan, 1, nan])` → values `[1., nan, nan]`,
counts `[1, 1, 1]`), and diffs from `np.unique`'s collapsing. Ascending order
matches `np.unique` (and the pre-2.3 aliases); the naive (general-bound
`Clone + PartialEq`) algorithm cannot order complex values without the
`ExtSortCmp` total order, so first-occurrence order stands for them
(substituting the sorted path for complex is a registered follow-up). Signed
zeros merge in both paths, keeping the first-seen encoding (also NumPy's
behavior: `np.unique([-0., 1., 0.])` → `[-0., 1.]`).

## Array indexing: no grouped index tuples

- **numpy:** `x[1:3, ([0, 1, 2], [0, 2, 1])]` groups index arrays into one advanced indexer.
- **rstsr:** entry_row_cpu::core_func::indexing::test_array_index (numpy_array_index + custom_array_index)
- **tag:** intentional
- **status:** open

`rt::array_index` implements NumPy's *vectorized indexing* for ungrouped keys
(`x[1:3, [0, 1, 2], [0, 2, 1]]`) including the placement rule for the broadcast dimensions
and boolean masks: a mask is lowered to its `nonzero` coordinates, so it consumes one axis
per mask axis (they must match the indexed axes exactly — a zero-size mask axis is allowed,
NumPy-style) and contributes one `(count,)` dimension. Grouped ("parenthesized") index
tuples are still not supported and raise `UnImplemented`. In the Python layer a lone boolean
array still routes to whole-tensor mask indexing (`rt::mask_select`, i.e. `x[mask]`); a
boolean mask mixed into a Python key rides the same `rt::array_index` path (the faer-py shim
exposes it since 2026-10-09). Advanced key assignment (`x[idx] = value`) is implemented by
[`rt::array_index_assign`]; the duplicate-index and column-major divergences of its write order
are registered separately (see "Array-indexing assignment: write order"). The index
argument is a dedicated argument type ([`ArrayIndexArgs`]) rather than
`AxesIndex<ArrayIndexer<B>>`, because the latter's conversions cannot be implemented outside
`rstsr-common` (the orphan rule); the `AxesIndex` form is still accepted through `TryFrom`.

## Array-indexing assignment: write order

- **numpy:** when an index selects the same element more than once
  (`a[[3, 3, 3]] = [1, 2, 3]`), the value written last wins, in C (row-major) order.
- **rstsr:** entry_row_cpu::core_func::indexing::test_array_index_assign
- **tag:** col-major-transfer
- **status:** open

`rt::array_index_assign` overwrites the selected elements in place. When an element is selected by
more than one index entry, the value written last wins, and the entries are visited in the device
default order — exactly NumPy's C-order last-write-wins under `RowMajor`, but a **different winner**
under `ColMajor` (the visit order is the device's, registered under `col-major-transfer`). The
scatter is deliberately serial: a rayon scatter would race on duplicate destinations (undefined
behavior for concurrent non-atomic stores), so the device trait contract is serial and the rayon
device delegates to the serial kernel. The derived setters (`put_along_axis`, `index_put`,
array-valued `mask_assign`) follow the same rule: their gathers (`take_along_axis`, `index_select`,
`mask_select`) are parallel, but the scatters are serial with the same last-write-wins order.

## Array indexing: no zero-dimensional boolean indexers

- **numpy:** a zero-dimensional boolean adds a `{0, 1}`-sized block without consuming an axis.
- **rstsr:** entry_row_cpu::core_func::indexing::test_array_index::custom_array_index
- **tag:** intentional
- **status:** wontfix

A zero-dimensional boolean is declined in `rt::array_index` (`UnImplemented`): it would add a
`{0, 1}`-sized block without consuming an axis, which the index-array lowering cannot express.
This is a permanent decision rather than a follow-up; a lone boolean array in the Python layer
is unaffected (it routes to `rt::mask_select`).

## Gathering results follow the device order when flattened

- **numpy:** a gathered result is C-ordered, so its flattened sequence (`tolist()`, `ravel()`,
  iteration) is the row-major one — for `x[mask]` (`np.nonzero` order) and for
  `x[int_array, ..]` alike.
- **rstsr:** `entry_row_cpu::core_func::indexing::test_array_index::device_order` —
  `test_array_index_order_arrangement` (the displaced-run placement flip, the
  arrangement of a multi-dimensional result — identical for 1-D results — with
  the row-major/column-major flattenings checked against NumPy's `ravel()` /
  `ravel(order='F')`), `test_array_index_order_invariance` (the shapes and
  values of together runs) and `test_boolean_mask_orders` (the same for boolean
  masks, plus the count-axis `nonzero` sequence of a rank-2 mask), plus
  [`mask_select`]'s Row/Column Major Notice for the mask side
- **tag:** col-major-transfer
- **status:** open

`rt::array_index` follows NumPy exactly under the row-major default order (the placement rule and
the trailing-aligned broadcast of the index arrays included). Under a `ColMajor` device the
placement rule is measured in the device's access order: a run of advanced indexers that other
indexers *displace* is placed at the back of the result instead of the front (the broadcast block
keeps its contiguity role — the most-strided axis), so the *shape* differs from NumPy's there,
while a run that stays together keeps its shape and only the arrangement differs (the output is
allocated with `new_contig(order)` like every other op, so its flattened visit order is
column-major where NumPy's is row-major). A boolean mask in a key is lowered to its `nonzero`
coordinates, so a mask of rank two or more also carries the device's `nonzero` sequence on its
count dimension: under `ColMajor` it gathers the same positions in a different order, a value
difference and not merely an arrangement one. `rt::mask_select` goes one step further: its selection
*sequence* is the mask visit order, which follows the device default order by design (its
Row/Column Major Notice), so there the element order itself — not merely the memory arrangement —
is device-dependent, and a 1-D result carries the device sequence. All of these follow NumPy
exactly under the row-major default order, which is the order the library is held to (the
column-major device-order convention of
[`order_semantics`](https://github.com/RESTGroup/rstsr-core/blob/main/rstsr-core/src/docs/order_semantics.md)
is the registered transfer this entry tracks).


## `clip` takes an `(lo, hi)` group with `None` for an absent bound

- **numpy:** `numpy.clip(a, a_min, a_max)`; an absent bound is `None` in
  either position (`np.clip(a, 3)`, `np.clip(a, max=4)`).
- **rstsr:** `rt::clip(&a, (lo, hi))` / `a.clip((lo, hi))` - a single argument
  group whose `lo`/`hi` are each a scalar, a tensor (broadcast), or the `None`
  wildcard. With both bounds absent, the result is a copy of `a` (array-API);
  NumPy's bound-less `clip(a)` raises `ValueError` instead.
- **tag:** intentional
- **status:** open

The `None` wildcard borrows the tensor's own element type, so
`rt::clip(&x_f32, (0.0_f32, None))` stays `float32`, while a differently-typed
scalar bound still promotes (see *Scalar arguments promote like tensors*). NumPy's
in-place `out=` / `casting=` variants are not provided (rstsr clip is
out-of-place only).


## `tensordot`'s bare axes collection means "same axes", not NumPy's pair

- **numpy:** `_core/tests/test_numeric.py::TestTensordot` - `np.tensordot(a, b, [0, 1])`
  treats a 2-element `axes` as the *pair* `(0, 1)` (contract `a` axis 0 with `b` axis 1),
  identical to `axes=(0, 1)`.
- **rstsr:** `rt::tensordot(&a, &b, [0, 1])` / `vec![0, 1]` is the `AxesPairIndex`
  same-axes shorthand - contract `a` axes {0, 1} with `b` axes {0, 1}. For NumPy's pair
  meaning use the explicit tuple `(0, 1)`. The array-API-valid forms (`int`, `(seq, seq)`)
  are unaffected; this only touches the bare-collection spelling.
- **tag:** intentional
- **status:** open

`axes = None` / `()` is accepted as the array-API default `2` (Rust has no default
arguments); NumPy raises on `None`. This is a Rust-callers' convenience, not a behavioral
divergence for the Python surface.
