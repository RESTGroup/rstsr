# Changelog

## v0.10.4 -- 2026-11-17

Dtype conversion, triangular pack/unpack free functions, and a device-API
refactor for host copy and formatting.

API breaking changes (user should not feel that)

- Host copy and formatting move into the `DeviceToCpuAPI` and `DeviceFormatAPI`
  capability traits; `Raw` and `Data` no longer have to be `Clone` on the device
  API. (RESTGroup/rstsr#164)

New features

- `astype` / `astype_f` convert a tensor to another dtype. (RESTGroup/rstsr#165)
- `pack_tri` / `pack_tril` / `pack_triu` and `unpack_tri` / `unpack_tril` /
  `unpack_triu` are free functions, with `with_output` variants.
  (RESTGroup/rstsr#166)

Enhancement

- `Debug` formats host-backed tensors in place. (RESTGroup/rstsr#164)

Dev infrastructure

- Test entries follow `entry_<order>_<device>`, and CI runs in tiers.
  (RESTGroup/rstsr#163)

## v0.10.3 -- 2026-10-19

Zero-size layouts get a stride and leading-dimension contract, and the BLAS-3 and
LAPACK wrappers accept empty operands.

API breaking changes (user should not feel that)

- `ld_row` / `ld_col` return `None` for overlapping layouts and `max(1, other)`
  for axes of length 0 or 1. (RESTGroup/rstsr#162)

New features

- `TensorBase::testify_ld` reports an illegal leading dimension as an
  `InvalidLayout` error. (RESTGroup/rstsr#162)
- `gemm`, `syhemm` and `trsm` accept empty operands. (RESTGroup/rstsr#162)
- LAPACK wrappers accept empty operands; `gesv`/`sysv` still factor `A` when the
  right-hand side has no columns. (RESTGroup/rstsr#162)

Behavior change

- Contiguous constructors report all-zero strides for zero-size shapes.
  (RESTGroup/rstsr#162)

Bug Fix

- Empty operands no longer reach BLAS/LAPACK drivers. (RESTGroup/rstsr#162)
- `syhemm` used `a.ncol()` instead of `b.ncol()` for the output columns.
  (RESTGroup/rstsr#162)

Dev infrastructure

- Add an OpenBLAS-only zero-size test suite. (RESTGroup/rstsr#162)

## v0.10.2 -- 2026-10-15

Two small post-release fixes surfaced by downstream and review follow-ups. No
user-facing API is removed; new bounds on `BlasDriverBaseAPI` affect trait
implementors only.

Bug Fix

- The `ManuallyDrop`-backed `asarray` path (`(&mut [T], ...)` inputs) fabricated
  its `Vec` through a shared-provenance pointer; the two `&mut [T]` sites now
  use `as_mut_ptr`, fixing a miri-confirmed Stacked-Borrows violation.
  (RESTGroup/rstsr#159)

Enhancement

- `iter` / `indexed_iter`, `iter_mut` / `indexed_iter_mut`, `axes_iter` /
  `indexed_axes_iter` and their `_mut` twins are available directly on `Tensor`,
  `TensorArc` and `TensorCow` (`&self` for the shared forms; `TensorCow` is
  immutable-only), as borrow-tied delegating wrappers over the view API.
  Iterating an owned temporary remains a compile error, and view forms keep
  their inner-lifetime semantics (`a.t().iter()`). (RESTGroup/rstsr#159)
- `BlasDriverBaseAPI` carries the bounds generic LAPACK-driver code needs to
  call the core matmul entry points: `DeviceMatMulAPI` (fixed and dynamic rank)
  and explicit `DeviceRawAPI<MaybeUninit<...>, Raw = Vec<...>>` projections, so
  downstream code bounded by `LapackDriverAPI<T>` no longer restates raw-type
  bounds at every call site. The five vendor device impls gain `+ 'static`.
  Implementors of `BlasDriverBaseAPI` are affected. (RESTGroup/rstsr#160)

## v0.10.1 -- 2026-10-14

A documentation release. The `rstsr` facade gains a NumPy/SciPy cheat sheet, its
docs.rs pages render feature-gated items, and the Array API fulfillment pages are
brought up to date. No API changes.

Documentation

- **NumPy/SciPy cheat sheet** for the `rstsr` crate: a core-operations page and a
  faer-backed `rt::linalg` page map common NumPy and SciPy calls to their RSTSR
  equivalents. The RSTSR column prefers associated methods (`a.sum()`,
  `a.zeros_like()`) over the free functions, except where only a free function
  exists or the method means something else (`rt::rem`, `rt::add_with_output`).
  (RESTGroup/rstsr#158)
- The `rstsr` facade renders its docs.rs pages with `doc_cfg`, so feature-gated
  items carry "available on crate feature ..." badges. (RESTGroup/rstsr#158)
- The `order_semantics` page is rewritten in shorter paragraphs, with an explicit
  note that row/column-major (iteration order) is not c/f-contiguity (storage
  order) and a new section on the `iter()` flattening order. (RESTGroup/rstsr#158)
- The Array API fulfillment pages are corrected: `cumulative_sum` /
  `cumulative_prod` and the linear-algebra extension functions are marked
  implemented, and `from_dlpack` is available behind the `dlpack` feature of the
  facade crate rather than dropped. (RESTGroup/rstsr#158)

Dev infrastructure

- A CI workflow runs the cheat-sheet integration tests and the NumPy/SciPy parity
  script. (RESTGroup/rstsr#158)

## v0.10.0 -- 2026-10-14

The first stable release of the 0.10 series. This cycle consolidates rather than
extends: a documentation campaign over the overloaded APIs, a LAPACKE-shaped
retrofit of the LAPACK drivers, and the first distribution of the
`rstsr-faer-py` validation wheel.

API breaking changes

- `rstsr-cpu-dlpack` is rebased on `dlpark`: the raw DLPack types are reached as
  `rstsr_cpu_dlpack::dlpark::ffi::*` instead of `::dlpack_ffi::*` (the C layout is
  unchanged), and the export entry points now require `Send` on the storage they
  hand out. (RESTGroup/rstsr#153)

New features

- **`rstsr-faer-py` is distributed as a wheel.** A workflow builds the pyo3
  extension for manylinux and uploads it through Trusted Publishing (OIDC, no
  stored secret), and the package now carries the metadata, long description and
  "not for real use" notice a PyPI page needs. (RESTGroup/rstsr#155)
- `pinv` accepts an array `rtol` broadcast against the matrix batch — the same
  argument shape as `matrix_rank` — in both the Rust API and the shim's
  `linalg.pinv`. (RESTGroup/rstsr#156)

Enhancement

- **LAPACK drivers take the LAPACKE two-tier shape**: one `unsafe *_work` thunk
  per routine holds the column/row-major branch, including the leading-dimension
  checks row-major calls were missing, and the driver impl runs the workspace
  query, allocation and compute through it. Query results are validated before
  use, allocation failures return the LAPACKE sentinels instead of panicking, and
  `info` is decoded in one place for every builder. (RESTGroup/rstsr#152)
- **API-docstring campaign**: every overloaded `rstsr-core` entry documents its
  overload forms with tested snippets, the API specification page covers the
  identifiers added since it was last revised, and the `ext_*` entries state that
  they exist for array-API promotion and point at the ordinary form for Rust use.
  (RESTGroup/rstsr#149, #151)

Bug Fix

- `unpack_tri` / `unpack_triu` with `FlagSymm::N` zero-fill the triangle the
  packed data does not describe, instead of exposing uninitialized memory.
  (RESTGroup/rstsr#150)
- The complex `sysv` / `syev` / `syevd` / `sygv` / `sygvd` drivers no longer
  write a complex workspace query through an `f64` stack slot (undefined
  behavior). (RESTGroup/rstsr#152)

Python array-API conformance (validation instrument)

- Against `array-api-tests` (pinned `6c0b59f`, spec 2025.12) the last measured
  standing is 1332 passed / 17 failed / 33 skipped of 1382 — the 14 out-of-scope
  `fft` name nodes plus three `linalg` value nodes held back by one faer SVD
  non-convergence. The wheel published to PyPI is this instrument, not a usable
  Python tensor library.

Dev infrastructure

- The `rstsr-faer-py` extension is split into 13 crates under
  `crates-interop/rstsr-faer-py/`, so no single `rustc` monomorphizes the whole
  dtype-dispatch surface (25 GB → 4 GB peak per process). (RESTGroup/rstsr#155)
- The distributed wheel is stripped (the symbol tables alone were ~410 MB;
  51 MB → 39 MB) and built with four parallel jobs; local grading wheels stay
  unstripped so backtraces survive.
- Tests only: empty-collection assertions are pinned to `is_empty()`, so clippy's
  all-features run stops failing on `E0282` inference. (RESTGroup/rstsr#154)

## v0.10.0-alpha.2 -- 2026-10-12

This cycle completes the Python array API standard's `linalg` extension and
fills in the advanced-indexing, promoted-dtype and remaining element-wise
surface. As in alpha.1, the surface is still moving: further API-breaking
changes are expected within the alpha: we haven't touched BLAS vendor devices.

API breaking changes

- The faer linear-algebra entries take n-dimensional (stacked) input and return
  dynamic-dimension outputs: `det_f` returns the determinant as a `Tensor<_,
  IxD>` — a 0-d array for a 2-D input, read back with `.to_scalar()` — instead
  of a bare scalar, and the allocating `solve_general` / `solve_triangular` and
  the generalized `eigh` return `IxD` tensors instead of their previous fixed
  ranks. (RESTGroup/rstsr#140)
- `SLogDetResult` carries two independent type parameters, and `slogdet` on a
  plain 2-D matrix returns a pair of 0-d tensors instead of `(T, T::Real)`
  scalars — destructure the result as `(sign, logabsdet)`. (RESTGroup/rstsr#133)
- `ExtNum` gains a required `ext_rem` method, and several element-trait bounds
  are re-pointed: `OpSignBitAPI` moves onto `ExtReal`, `ceil` / `floor` /
  `trunc` return the input type, and `pow` follows the promoted binary-op
  shape. The `rstsr_native_impl::scalar_math` module is removed (ties-to-even
  now lives in `ExtNum::ext_round`). Trait implementors are affected; call
  sites are not. (RESTGroup/rstsr#123, #124)

New features

- **Advanced (fancy) indexing and assignment.** `rt::array_index` gathers by
  integer index arrays and boolean masks with NumPy's vectorized semantics: the
  index arrays broadcast, their result dimensions follow NumPy's placement rule
  (in place for a contiguous run of indexers, at the front otherwise), basic
  indexers mix freely in the same key, and a mask is lowered to its `nonzero`
  coordinates. The setter counterparts write the same keys —
  `rt::array_index_assign` plus the derived `put_along_axis` / `index_put` /
  `mask_assign` — and `rt::mask_select` / `rt::mask_fill` cover the whole-tensor
  single-mask case (`x[mask]`, `x[mask] = value`). The gather kernels run on the
  rayon pool (about 5–6× at kernel level); the scatters are serial by design,
  since duplicate index targets are last-write-wins in the device order. New
  device traits back each family. (RESTGroup/rstsr#126, #127, #128, #129)
- **The array-API `linalg` extension is complete.** Every member the standard
  defines is now present. The faer device gains stacked (n-dimensional) forms of
  `cholesky`, `det`, `eigh` / `eigvalsh`, `inv`, `pinv`, `svd` / `svdvals`,
  `slogdet` and `solve` — with vector or matrix right-hand sides and batch
  broadcasting — and the remaining members: `matrix_power`, `matrix_norm` /
  `vector_norm`, `matrix_rank`, the general (non-Hermitian) `eig` / `eigvals`,
  and `qr`. The device-independent tier adds `tensordot`, `outer` /
  `outer_from`, `trace` (with its `dtype` form) and `cross`.
  (RESTGroup/rstsr#133, #135, #138, #140, #141, #142, #143, #145, #146, #147)
- **Mixed-dtype (promoted) entries.** A family of `ext_*` forms serves the
  array-API promotion contract — `ext_add` / `ext_sub` / `ext_mul` / `ext_div`
  and the bitwise and shift operators, `ext_matmul` / `ext_vecdot` /
  `ext_tensordot`, `ext_outer`, `ext_cross` and `ext_neg` — promoting their
  operands to the common data type in the kernel and accumulating there.
  Promotion is resolved on the device impl, so the tensor tier carries no
  element-type bound; the same-dtype functions and the Rust operators remain the
  idiomatic surface. (RESTGroup/rstsr#125, #137, #138, #143)
- Floored element-wise remainder, `signbit` extended to unsigned and half
  dtypes, `pow` for integer bases with integer exponents and complex bases, and
  `clip` with optional `(lo, hi)` bounds. (RESTGroup/rstsr#123, #124, #132)
- `rstsr-faer-py`: `result_type`, `can_cast` and `isdtype` for runtime dtype
  queries, over a `dtype_promote` binding derived from the very
  `DTypePromoteAPI` impls the operations use. (RESTGroup/rstsr#131)

Behavior change

- `maximum` / `minimum` and the `max` / `min` reductions now propagate NaN
  (IEEE 754-2019 `maximum` / `minimum`, the standard's contract); the previous
  behavior dropped it. (RESTGroup/rstsr#124)
- `signbit` returns the sign bit proper (true for negative values, `-0.0` and
  negatively-signed NaN) instead of the inverse of `is_positive()`.
  (RESTGroup/rstsr#124)
- `ceil` / `floor` / `trunc` / `round` preserve the input dtype instead of
  widening integers to `f64`; `round` is componentwise and ties-to-even, and
  `round` / `ext_neg` accept complex. (RESTGroup/rstsr#124)
- `rem` on floating-point dtypes follows the standard's floored `remainder`
  (the result takes the sign of the divisor), including the signed-zero and
  infinite-divisor cases, instead of Rust's `fmod`; `pow` with a negative
  integer exponent now fails with `InvalidValue`. (RESTGroup/rstsr#123, #124)
- `solve_general` / `solve_triangular` reject a rectangular `a` with
  `InvalidLayout` rather than solving it in the least-squares sense, following
  the standard. (RESTGroup/rstsr#140)

Python array-API conformance (validation instrument, not published)

- The `rstsr_faer.api` namespace now covers the `linalg` extension end to end
  (including `Array.__matmul__`), `x[key]` for integer-array and boolean-mask
  keys with the matching `__setitem__`, the promoted `ext_*` entries, and the
  runtime dtype queries. (RESTGroup/rstsr#131, #133, #137, #138, #142, #143,
  #144, #145, #146, #147)
- Against `array-api-tests` (pinned `6c0b59f`, spec 2025.12) the standing is
  1332 passed / 17 failed / 33 skipped of 1382, up from 1144 / 156 / 82 at
  alpha.1. The residual 17 are the 14 out-of-scope `fft` name nodes and three
  `linalg` value nodes (`matrix_rank` / `pinv` / `svd`) held back by one faer
  SVD non-convergence on large-magnitude inputs. (RESTGroup/rstsr#130, #131,
  #132, #144)

Enhancement

- Refresh faer to 0.24 (`faer-ext` to 0.8); backend behavior is unchanged.
  (RESTGroup/rstsr#139)
- Document the shim-side dtype pre-casts ("dtype-cheating": `solve`, `isin`,
  `concat` / `stack`) in a new `DTYPE-CASTS.md`, distinguishing them from the
  casts the standard itself mandates. (RESTGroup/rstsr#144)

Bug Fix

- `Layout::diagonal` rejects `axis1 == axis2` (previously an unsound view past
  the end of the buffer); `Layout::is_broadcasted` no longer misreads a
  zero-size layout as broadcast, so contiguous empty tensors are writable
  again; and `Layout::dim_narrow` follows Python's slicing rules for negative
  steps. (RESTGroup/rstsr#126, #127, #138)
- `pinv` returns the transposed zero matrix with rank 0 for a zero-size matrix
  axis instead of panicking; `eig` / `eigvals` treat a zero-size batch axis as
  empty output. (RESTGroup/rstsr#146)
- `rstsr-faer-py`: a `uint64` scalar carrier so large unsigned values no longer
  read back as negative, `conj` preserving integer dtypes, and a set of surface
  corrections (`finfo` on complex dtypes, `devices()` / `dtypes()` shapes,
  positional-or-keyword `shape`, `astype(device=…)`). (RESTGroup/rstsr#130,
  #137)

Dev infrastructure

- Deduplicate the five BLAS device crates' `linalg_func` / `driver_impl` test
  sources into `rstsr-blas-traits/src/blas_tests_impl/`, symlinked into each
  device, with only genuine device divergences kept as real files.
  (RESTGroup/rstsr#134)

## v0.10.0-alpha.1 -- 2026-10-08

Its theme is conformance to the Python array API standard, so the surface is still
moving: further API-breaking changes are expected within the alpha series while
rstsr-core is fitted to the standard.

This release also introduces `rstsr-cpu-dlpack`, which will be beneficial for interop with Python and other languages.

New features

- New crate `rstsr-cpu-dlpack` (`crates-interop/`): CPU DLPack interchange for rstsr
  tensors, pure Rust and without Python bindings. Foreign capsules import zero-copy
  and read-only (the tensor carries the producer's deleter and owns its lifetime),
  and tensors export as movable, copied, or shared capsules, including a zero-copy
  basic-indexed view of a shared base. Exposed through the `rstsr` prelude behind the
  `dlpack` cargo feature as `rt::dlpack::*`. (RESTGroup/rstsr#107)
- `rt::where` / `rt::where_f` (method `cond.r#where(x, y)`): element-wise select
  following NumPy's three-argument `np.where` and the array API `where(condition,
  x1, x2)`. The condition is a strict boolean tensor; `x` / `y` accept tensors
  (reference or by-value view) or scalars, broadcasting the three inputs together
  under the device default order with house-strong promotion. Built on a new
  quaternary (output + three inputs) elementwise kernel family.
  (RESTGroup/rstsr#114, #116)
- `rt::from_scalar(value, &device)`: the 0-d tensor owning a value — rstsr's spelling
  of NumPy's 0-d `numpy.array(value)`. The 0-d result broadcasts with any tensor,
  giving the sanctioned way to pass constants, bool included, to tensor-level APIs
  (`&rt::from_scalar(false, &device)` for a bool scalar). (RESTGroup/rstsr#115)
- `rt::positive` as a first-class unary op (`TensorPositiveAPI` / `OpPositiveAPI`),
  with no element-type bound on the kernel. (RESTGroup/rstsr#108)
- Reduction argument structs `ReduceArgs { axes, keepdims }` and
  `VarArgs { axes, keepdims, correction }`, with `<fam>_with_args` for every
  reduction family (keepdims without moving to the `_axes` form) and
  `*_with_dtype::<TOut>` for sum / prod / mean / var / std (cast per element inside
  the fold, no input-sized intermediate). `var` / `std` gain NumPy's `ddof` as
  `correction`. (RESTGroup/rstsr#110)
- Expert custom reductions `rt::reduce_all` / `rt::reduce_axes` / `rt::reduce_with_args`
  over a user-provided init/fold/combine/finalize closure with a generic accumulator
  and output (e.g. an Lp-norm or a multi-accumulator mean in one pass), and the
  array API cumulative scans `rt::cumulative_sum` / `rt::cumulative_prod` with
  `CumulativeArgs { axis, include_initial }`. (RESTGroup/rstsr#111)
- Manipulation, sorting, searching and set-function wave, all device-independent:
  `rt::repeat` / `rt::roll` / `rt::tile`; `rt::sort` / `rt::argsort` with
  `SortArgs { axis, descending, stable }` plus the `*_custom` comparator forms
  (NaN ordered last in both directions); `rt::searchsorted` with a NumPy-style
  `sorter`; `rt::nonzero`; `rt::take_along_axis`; `rt::diff`; and the set family
  `rt::unique_values` / `unique_counts` / `unique_inverse` / `unique_all` (named
  structs) and `rt::isin`. New support types `ExtSortCmp` / `ExtZero` /
  `AxisIndex` and matching `Op*API` device traits are implemented across the serial,
  rayon/faer and BLAS backends. (RESTGroup/rstsr#118)
- `AxesPairIndex<isize>` accepts a single axes collection as the pair `(axes, axes)`,
  so `rt::vecdot(a, b, axes)` and `tensordot` take the same axes for both operands
  (write the pair explicitly as `(0, 1)` when the axes differ). (RESTGroup/rstsr#120)
- `rt::log1p` (implemented on every device, and prelude-exported at last), and
  complex support for `expm1`. Both avoid the cancellation of `ln(1 + z)` / `exp(z)
  - 1` for complex input. Backed by a new `ExtComplexFloat` trait so one device-table
  row serves real and complex element types. (RESTGroup/rstsr#121)
- C99-conformant complex elementary functions (`sqrt`, `cosh`, `sinh`, `tanh`, `tan`,
  `acos`, `asin`, `acosh`, `asinh`, `atanh`) via the new `c99_complex` module:
  NumPy-compatible branch cuts and special values in place of the NaN components
  `num-complex` returns, and better accuracy for large `|z|`. (RESTGroup/rstsr#122)
- `AxesIndex` conversions accept one-element tuples, so `a.i((1..4,))` indexes like
  `a.i(1..4)`. (RESTGroup/rstsr#117)
- `Clone` for `DataArc` and `TensorArc`: a zero-copy handle clone with copy-on-write
  through `Arc::make_mut`. (RESTGroup/rstsr#107)

Python array-API conformance (validation instrument, not published)

- New `rstsr-faer-py` crate (`crates-interop/`): a pyo3 binding exposing the array-API
  namespace `rstsr_faer.api` over `DeviceFaer`, used to grade rstsr with the official
  data-apis `array-api-tests` suite. Wrapper-only by design — marshalling in Python,
  dtype dispatch and `rt::` calls in Rust, no algorithms in either layer — and never
  published (`publish = false`, no wheels). Divergences are recorded as gap/register
  entries rather than worked around in the shim. (RESTGroup/rstsr#108)
- Surface bound over the cycle: elementwise/operator dunders, creation,
  manipulation, searching/indexing, statistical reductions and cumulative scans,
  `where`, `log1p` and the other transcendental entries, with the `"data-dependent
  shapes"` capability reported. (RESTGroup/rstsr#108, #112, #113, #116, #118, #121)
- Against `array-api-tests` (pinned `6c0b59f`, spec 2025.12) the current standing is
  1144 passed / 156 failed / 82 skipped of 1382. (RESTGroup/rstsr#108, #112, #113,
  #116, #118, #121, #122)

Enhancement

- Adopt the tensor-tier API conventions across the new surface: `func_f` carries
  exactly the signature of its panicking twin; every tensor-first function gains an
  associated `TensorAny` method (views borrowed, not consumed); argument groups
  travel as one tuple parameter supplying overloads via `From` / `TryFrom`, with
  `None` accepted wherever `()` is; read-only parameters take `impl TensorViewAPI`;
  and tuple-style constructors accept `TensorView` values. (RESTGroup/rstsr#118)
- Move element-type bounds off the tensor layer onto the device impls, and make the
  flattened visit order follow the device default order for `repeat` / `roll`
  (axis = None), `nonzero`, and the `unique_*` first-occurrence sequence.
  (RESTGroup/rstsr#118)
- FFI dependency refresh: `rstsr-openblas-ffi` 0.5 → 0.6 (Linux-only affinity API plus
  generator hardening), `rstsr-mkl-ffi` 0.2 → 0.3 (Intel oneAPI MKL 2026.1), and
  `rstsr-aocl-ffi` 0.2 → 0.3 (AOCL 5.3.0). Released from the sibling `rstsr-ffi` repo.
- Refresh the other dependencies toward their latest releases: `itertools` 0.13 → 0.15,
  `num-bigint` 0.4 → 0.5, `npyz` 0.8 → 0.9 and `criterion` 0.5 → 0.8 drop in with no
  source changes; `rand` 0.8 → 0.10 and `ndarray` 0.15 → 0.17 needed a test-code
  migration (`RngExt::random`, `into_raw_vec_and_offset`).
- Rewrite the slice basic-indexing documentation around the `tensor.i()` + tuple call
  forms, as an `include_str!`-attached docs page with `doc_draft`-verified examples
  (NumPy-parity of newaxis/ellipsis placement against the pinned NumPy 2.5.2), plus
  `compile_fail` fences carrying real diagnostics. (RESTGroup/rstsr#117)
- Move the anchor docstrings of `allclose`, `var_with_args` and `std_with_args` onto
  the panicking forms, restoring the "fallible twin defers with See also"
  convention. (RESTGroup/rstsr#119)

Bug Fix

- Fix `Layout::diagonal` gating the super-diagonal range on rows instead of columns
  (`eye(2, 4, k=2)` returned an all-zero matrix and `eye(3, 1, k=2)` a bogus
  layout-overflow error); all `diagonal` consumers share the fix. (RESTGroup/rstsr#113)
- Fix `triu` indexing past the row when the diagonal leaves the matrix (a panic for
  `k >= ncols` or `M > N`), and `tril` / `triu` surfacing a bare `AxisError` for
  rank-1 input. (RESTGroup/rstsr#113)
- Fix `linspace` not including the endpoint exactly and the serial kernel's
  accumulated drift; both kernels now compute `start + i * step`. (RESTGroup/rstsr#113)
- Fix `squeeze` accepting a mixed axis list containing an invalid negative axis, and
  `take` / `index_select` raising on empty indices against a zero-length axis;
  `argmax` / `argmin` no longer reject an empty (but non-reduced) output.
  (RESTGroup/rstsr#113)
- Fix axes reductions over broadcast (stride-0) inputs returning silently wrong
  values, an out-of-bounds panic when reducing all axes via an axis list, and
  zero-size inputs panicking through an all-axes list (they now return the identity
  value per output cell, with empty mean as NaN). (RESTGroup/rstsr#110)
- Fix the promotion matrix missing `DTypePromoteAPI<i16> for i8` (the impl line had
  been swallowed by a section comment since 2025-09-29), so every promotion-bound op
  — maximum/minimum, floor_divide, atan2, copysign, hypot, nextafter, logaddexp,
  comparisons — now accepts int8 x int16 pairs. (RESTGroup/rstsr#116)
- Fix the contiguous-assignment fast path panicking on zero-size assignments in both
  the serial and rayon kernels. (RESTGroup/rstsr#118)
- Fix `DataArc::into_owned` panicking when the buffer is shared; it now falls back to
  a data clone, as documented. (RESTGroup/rstsr#107)
- `rt::round` resolves halfway cases to the even integer (IEEE ties-to-even, matching
  NumPy) instead of Rust's ties-away `f64::round`. (RESTGroup/rstsr#112)
- `arange` with a step pointing away from the stop is now direction-aware:
  sign-mismatched ranges and `start == end` return empty instead of looping
  unboundedly (float) or coming back empty (narrow integers). (RESTGroup/rstsr#108)
- Complex `sqrt`/`cosh`/`sinh`/`tanh`/`tan`/`acos`/`asin`/`acosh`/`asinh`/`atanh`
  return the standard's special values instead of NaN; `expm1(±0 ± 0i)` returns
  `+0 + 0i`; `tanh(±inf + iy)` takes its imaginary zero from the standard's fixed
  `+0`. (RESTGroup/rstsr#122)
- `rstsr-faer-py`: `meshgrid()` with zero vectors returns an empty tuple; `all` /
  `any` on a 0-d input keep the input shape; `searchsorted` with a float scalar
  against an integer array raises `TypeError`, and negative `repeat` / `tile` /
  `sorter` arguments raise `ValueError`. (RESTGroup/rstsr#113, #118)

Dev infrastructure

- New `rstsr-cpu-dlpack-test.yml` workflow covering the crate's tests (default and
  col-major) and the `rt::dlpack` prelude test. (RESTGroup/rstsr#107)
- Move `rstsr-dtype-traits` tests out of `src/` into `tests/` (one file per module),
  so the library is always tested in its `no_std` form. (RESTGroup/rstsr#122)
- Grow the NumPy-parity test surface and tracking for every new wave
  (`numpy_coverage.csv` 204 → 250 rows, `sync_numpy.py` oracle reporting zero
  MISSING, `numpy_differences.md` divergence entries), with `doc_draft` twins for the
  new docstrings. (RESTGroup/rstsr#110, #113, #114, #118)
- Add the `rstsr-faer-py-tests` harness skill (in `rstsr-agents`) for grading against
  `array-api-tests`. (RESTGroup/rstsr#108)

## v0.9.0 -- 2026-10-02

v0.9 will mark as the final version of full human-driven. We will be more aggressive in
leveraging AI-assisted development in future versions.

API breaking changes

- Element/axes iteration is now view-only: `iter` / `indexed_iter` / `axes_iter` /
  `indexed_axes_iter` are methods of `TensorView` (iterators tied to the view's inner
  data lifetime), and `iter_mut` / `indexed_iter_mut` / `axes_iter_mut` /
  `indexed_axes_iter_mut` are consuming methods of `TensorMut`. Owned/arc/cow tensors
  must call `.view()` / `.view_mut()` first; iterating an owned temporary is a compile
  error. This removes an unsound lifetime transmute in `Tensor::iter` that could
  reference freed data (use-after-free UB). (RESTGroup/rstsr#103)
- `meshgrid` return types now split by input ownership: reference inputs
  (`Vec<&TensorAny>`, arrays of references) return `Vec<TensorCow>`, where `copy = false`
  yields true NumPy-style broadcast views over the inputs' memory (previously owned
  copies were returned on all paths); by-value overloads now take concrete `Tensor`
  inputs (the previous generic by-value form also accepted views inside the `Vec`; use
  the reference forms instead). (RESTGroup/rstsr#99)

New features

- `rt::nanargmin` / `rt::nanargmax` (with `_f`, `_all`, `_axes` variants and `Tensor`
  methods) following NumPy `nanarg*` semantics: NaN elements are skipped wherever they
  appear, and an all-NaN slice raises `InvalidValue`; available on `DeviceCpuSerial`
  and rayon devices, with new device traits `OpNanArgMinAPI` / `OpNanArgMaxAPI`.
  (RESTGroup/rstsr#100)
- `broadcast_arrays` with reference-input overloads returning stride-0 broadcast views
  (no copy), as in NumPy; prelude-exported via `BroadcastArraysAPI`. (RESTGroup/rstsr#99)
- New storage accessors with documented validity contracts:
  `DataRef::try_as_true_ref` / `as_slice_ref`, `DataMut::try_into_true_mut` /
  `into_slice_mut`. (RESTGroup/rstsr#103)

Enhancement

- Performance: 8-lane contiguous fast path for `argmin`/`argmax` (serial and rayon),
  about 50x on large contiguous f64 inputs; the rayon variant combines chunk results
  deterministically, fixing a pairing-order-dependent wrong index. (RESTGroup/rstsr#100)
- Performance: blocked (64x64 tiled) 2-D iteration fast path for strided elementwise
  kernels (`op_mutc_refa_refb_func`, `op_muta_refb_func`, serial and rayon), with
  negative strides and non-zero offsets supported; strided `a + b^T` at 2048^2 is
  about 3x faster. (RESTGroup/rstsr#101)
- Performance: 2-D transpose-pattern assignments (equal-shape `a.t()`-pattern pairs,
  fast strides exactly +1) are routed through cache-friendly blocked orderchange
  kernels in `assign`/`assign_arbitary` (all promote/uninit/plain families, serial and
  rayon). (RESTGroup/rstsr#102)
- `Layout::check_strides` no longer allocates in the common case and raises
  `InvalidLayout` on span overflow instead of wrapping in release mode.
  (RESTGroup/rstsr#105)
- The `uninitialized_vec` allocation family now has a written contract
  (`alloc_vec_contract.md`, rustdoc-included on all three allocation functions).
  (RESTGroup/rstsr#105)
- Wholesale API documentation retrofit of rstsr-core: every docstring example is
  verified as rustdoc doctests (now CI-gated) with `doc_draft` twin tests asserting
  captured printed output. (RESTGroup/rstsr#99)

Behavior change

- Writes through broadcast (stride-0) layouts are now rejected on every write path
  (in-place assign family, `mapi` family, matmul `*_with_output` drivers,
  `op_with_func` drivers, `vecdot_from_f`, and the `iter_mut`/`axes_iter_mut`
  constructors). Previously a broadcast output under rayon handed overlapping offsets
  to different threads (data race), and the serial path silently aliased writes.
  Constructing and reading a broadcast `TensorMut` stays legal; unary in-place ops on
  owned broadcasted tensors fall back to producing a fresh output.
  (RESTGroup/rstsr#105)
- `to_contig` decides view-vs-copy by NumPy-style contiguity flags (`c_contig()` /
  `f_contig()`) instead of exact layout equality; padded-singleton contiguous tensors
  (e.g. shape `[3, 1]`, stride `[1, 3]`) are no longer copied. (RESTGroup/rstsr#99)

API breaking changes (user should not feel that)

- `Send`/`Sync` bounds of the storage wrappers `DataRef`/`DataCow`/`DataArc`/
  `DataReference` tightened: a wrapper that can expose a shared reference is
  `Send`/`Sync` only when `C: Send + Sync` (previously `DataRef<Cell<i32>>` could be
  sent across threads). `IntoParallelIterator` for the axes parallel iterators
  additionally requires `B::Raw: Sync`. (RESTGroup/rstsr#104)
- `DeviceMatMulAPI` gains the required method `matmul_uninit` (write-only
  `c = alpha * a @ b` into `MaybeUninit` storage), implemented by all in-repo devices.
  (RESTGroup/rstsr#105)
- `DimShapeAPI::unravel_index_f`/`unravel_index_c` and `Layout::index_uncheck` are no
  longer `unsafe`; `ChangeableDefault::change_default` is now a safe fn (defaults
  backed by `AtomicU8`); faer owned-`Mat` `into_rstsr` gained a `T: Clone` bound (it
  now copies; see Bug Fix). `IntoRSTSR` is exported via the prelude `rstsr_traits`.
  (RESTGroup/rstsr#105)

Bug Fix

- Fix heap out-of-bounds in `rt::full((layout, fill_value))` when the user-supplied
  `Layout` has non-zero offset or negative strides; allocation now uses
  `bounds_index()`, exactly like `empty`/`zeros`/`ones`/`uninit`. (RESTGroup/rstsr#104)
- Fix allocation-size overflow in `aligned_uninitialized_vec` (`aligned_alloc` feature,
  rstsr-common): the byte count is now `checked_mul`'d and raises `RuntimeError` on
  overflow. (RESTGroup/rstsr#104)
- Fix pointer-provenance UB in parallel kernels: the write-through
  `c.as_ptr() as *mut` pattern is replaced by hoisted `AtomicPtr` at all rstsr-
  native-impl rayon kernel sites and the rstsr-sci-traits cdist rayon kernels; the
  batched-broadcast gemm parallel-outer branch and syrk write-back now derive
  per-task pointers with unique provenance. (RESTGroup/rstsr#105)
- Fix an unsound allocation re-home in faer owned-`Mat` `into_rstsr`
  (`mem::forget` + `Vec::from_raw_parts` mismatched the dealloc layout on numeric
  conversion); it now copies column-wise, and the zero-copy route is
  `mat.as_ref().into_rstsr()` (a `TensorView`). (RESTGroup/rstsr#105)
- Fix a data race on the process-wide `TensorIterOrder` default (was a `static mut`
  read by safe code on effectively every tensor operation). (RESTGroup/rstsr#105)
- Fix `axes_iter`/`axes_iter_mut` (and indexed variants) panicking on an empty axes
  list; `axes_iter(())` now yields a single whole-tensor view, matching the NumPy
  `ndindex()` convention. (RESTGroup/rstsr#105)
- Fix naive matmul kernels violating the `beta = 0` non-read convention: a non-finite
  value in `c` could propagate through `0 * c`. (RESTGroup/rstsr#105)
- Fix `diag`/`concatenate` dropping uninitialized memory as `T` for allocatable
  element types; both now use `uninit_impl` + `assign_uninit` + `assume_init_impl`.
  (RESTGroup/rstsr#105)
- Fix GEMM/SYHEMM/TRSM wrappers in rstsr-blas-traits passing allocation-base pointers
  (ignoring `layout.offset()`) and hard-coded `ldc`/`ldb = m`: for f-prefer views with
  non-zero offset or padded leading dimension, operands were read from a wrong window
  and outputs written at the wrong stride (silently wrong results plus writes into
  parent elements outside the view). (RESTGroup/rstsr#106)
- Fix `driver_gesvd` order detection in rstsr-blas-traits: an inverted condition took
  the col-major driver path for row-major input and vice versa. (RESTGroup/rstsr#106)
- Fix GESVD/GESDD on empty matrices (`min(m, n) == 0`): both now return LAPACK-shaped
  zero-filled factors directly, and `superb` is sized via `saturating_sub` (was
  `usize` underflow). (RESTGroup/rstsr#106)
- Fix `getrf` allocating `ipiv` with `n` entries (LAPACK contract is `min(m, n)`);
  `getri` now rejects a short `ipiv` instead of reading out of bounds.
  (RESTGroup/rstsr#106)
- Fix ColMajor `eye` returning shape `(n_cols, n_rows)`; the layout is now
  `[n_rows, n_cols].f()`, identical logical content under both orders.
  (RESTGroup/rstsr#99)

Dev infrastructure

- Update agent-workflow references to the shared rstsr-agents workflow; NumPy tracking
  headers are machine-independent (checkout location recorded per-developer in
  `AGENTS.local.md`). (RESTGroup/rstsr#98)
- CI: new doctest job for rstsr-core (`cargo test -p rstsr-core --doc --release`).
  (RESTGroup/rstsr#99)

## v0.8.0 -- 2026-09-01

Behavior change

- The `openmp` feature of rstsr-openblas is now a default feature, and `rstsr`
  default features propagate it whenever the `openblas` feature is on. (RESTGroup/rstsr#94)
  Default-feature builds must now link an OpenMP runtime (`gomp`/`omp`);
  pthread-built OpenBLAS libraries remain compatible. Opt out with
  `default-features = false, features = ["linalg"]`.
  This default-feature change is the main motivation for the minor version bump.

Enhancement

- Add `ExtNum::ext_sign` following NumPy `np.sign` semantics: `-1`/`0`/`1` for
  signed integers (comparison form, no overflow at the type minimum), `0`/`1` for
  unsigned integers, NaN/±inf and signed-zero handling for floats, `z / |z|` for
  complex, and the same rules for `f16`/`bf16`. (RESTGroup/rstsr#93)
  `rt::sign` now accepts integer dtypes, and the previous `x / |x|` formula
  divergences are fixed: `sign(±inf)` no longer returns NaN, and `sign(-0.0)`
  returns `+0.0`.

API breaking changes (user should not feel that)

- `ExtNum` gained the new required method `ext_sign`; crates implementing `ExtNum`
  themselves must add it. The closed `ExtNum` type set also replaces the open
  `ComplexFloat` bound as the fallback for `rt::sign`, so not all `ComplexFloat`
  types remain accepted. (RESTGroup/rstsr#93)

Bug Fix

- Fix `OpenBLASConfig::get_parallel` panicking when both `openmp` and
  `dynamic_loading` were disabled, even on pthread builds; it now panics only
  when the library itself reports an OpenMP build. (RESTGroup/rstsr#94)
- Fix `no_std` builds: use rstsr-cblas-base 0.1.2 (which fixes its own `no_std`
  support), and replace `std` float operations and imports with `core`/`alloc`
  equivalents via the `num` crate. (RESTGroup/rstsr#96)

## v0.7.10 -- 2026-08-14

API breaking changes (user should not feel that)

- Rename module `manuplication` to `manipulation` (rstsr-core). (RESTGroup/rstsr#89)
  Only direct `rstsr_core::tensor::manuplication` import paths move; the `rstsr`
  facade prelude is unaffected.

Enhancement

- Introduce `AxisError` and `IndexError` error types in rstsr-common, replacing
  generic errors raised on axis/index misuse. (RESTGroup/rstsr#91)

Bug Fix

- Fix NumPy-parity bugs in reshape, argmax/argmin, sign, indexing and
  not-equal comparisons. (RESTGroup/rstsr#92)
- Fix dynamic-linking setup of rstsr-openblas for development on macOS. (RESTGroup/rstsr#88)

## v0.7.9 -- 2026-07-22

Bug Fix

- Fix index_select/bool_select with zero-size mask (RESTGroup/rstsr#86)

## v0.7.8 -- 2026-06-25

Bug Fix

- Fix thread oversubscription of broadcast matmul (BLAS devices) (RESTGroup/rstsr#85)

## v0.7.7 -- 2026-06-22

Bug Fix

- Fix matmul with broadcast in multiple-dimensions

  Now `[1, M, K] * [B, K, N] -> [B, M, N]` (the 7th rule of [matmul](https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html) with better broadcast support) now works.

## v0.7.6 -- 2026-06-07

Enhancement (also behavior change)

- Supports broadcasted matmul in column major. (RESTGroup/rstsr#81)

Behavior change

- Will not use aligned allocation in any situation when operating system is not Linux/MacOS. (RESTGroup/rstsr#82)

## v0.7.5 -- 2026-05-24

All changes at RESTGroup/rstsr#80.

Enhancement

- Allow LowerExp/UpperExp (scientific float print).

API breaking changes (user should not feel that)

- Changes the behavior of Display print, now the tensor will only be referenced when printing at least for CPU devices.

We still make Debug print to copy the whole tensor. In this way, the debug print uses minimal trait bounds, easier for debugging in other devices. This behavior may also change if necessary.

## v0.7.4 -- 2026-05-21

All changes at RESTGroup/rstsr#79

Bug Fix

- Fix function `rt::vecdot_from`, which previously falsely returned result instead of panic.

Enhancement

- add associated function of `vecdot` series
- add trait function (use as associated function) `einsum_from`

## v0.7.3 -- 2026-04-13

Bug Fix

- Fix previous behavior that layout equality check does not involve offset. (issue RESTGroup/rstsr#77, pr RESTGroup/rstsr#78)

## v0.7.2 -- 2026-04-10

All changes at RESTGroup/rstsr#76.

API breaking changes

- Refactored the `flip` methods for `FlagSide` and `FlagUpLo` enums to return the flipped value directly instead of a `Result`, since the operation cannot fail. Updated all call sites to remove unnecessary `?` error propagation.

Enhancements

- Re-exported the BLAS traits prelude under the new feature in `rstsr/src/prelude.rs` for easier access.
- Made all BLAS flag enums (`FlagOrder`, `TensorIterOrder`, `FlagTrans`, `FlagSide`, `FlagUpLo`, `FlagDiag`, `FlagSymm`) derive `Serialize` and `Deserialize`, and added Serde `rename` attributes for better control over their string representations.

Crate structure

- Added `serde` as a dependency in the relevant `Cargo.toml` files and imported the necessary traits in `flags.rs`.
- Symlinked `CHANGELOG.md` to rstsr crate directory.

## v0.7.1 -- 2026-03-30

Documentation update:

- rstsr-tblis: Updated documentation for einsum functions.

## v0.7.0 -- 2026-03-30

API breaking changes:

- rstsr-core: Removed unnecessary trait bounds on `DeviceCreationAPI` and related traits. Deprecated `arange_int_impl` as device function. Removed `DimStrideAPI` trait. (RESTGroup/rstsr#74)
- rstsr-core: Changed `TensorDotAxes` to `AxesPairIndex<T>` for general usage. Changed vecdot device trait function definition. (RESTGroup/rstsr#74)
- rstsr-core: Refactored AxesIndex trait bounds for improved error handling. (RESTGroup/rstsr#71)

New features:

- rstsr-core: Added `matrix_transpose` function for array-api compliance. (RESTGroup/rstsr#73)
- rstsr-core: Added `vecdot` traits and implementation (parallel for rayon devices). (RESTGroup/rstsr#73)
- rstsr-core: Added `reshape_with_args`, `into_compatible_shape` functions for flexible reshape operations. (RESTGroup/rstsr#70)
- rstsr-tblis: Implemented `tensordot` using einsum. (RESTGroup/rstsr#73)

Enhancements:

- rstsr-core: Efficiency improvement for `sum_axes` and related reduction with contiguous memory optimization. (RESTGroup/rstsr#74)
- rstsr-core: Parallel `arange` and `linspace` for devices supporting rayon. (RESTGroup/rstsr#74)
- rstsr-core: Changed output layout rule for binary functions; contiguous axes parts are preserved. (RESTGroup/rstsr#74)
- rstsr-core: Different-type reduction shares same implementation with same-type reduction. (RESTGroup/rstsr#74)
- rstsr-core: Restructured linalg directory; updated API documentation style. (RESTGroup/rstsr#73)
- rstsr-core: Enhanced stride checking in layout and reshape functions. Added NumPy-style tests for reshape and expand_dims. (RESTGroup/rstsr#70, RESTGroup/rstsr#71)
- rstsr-core: Testing framework updated for DeviceCpuSerial. Added reshape tests. (RESTGroup/rstsr#69)

Fixes:

- rstsr-core: Fix `change_layout` (contig may still require data copy).
- rstsr-tblis: Fixed threading and prelude imports. (RESTGroup/rstsr#74)
- rstsr-core: Various manipulation function bug fixes (expand_dims, flip). (RESTGroup/rstsr#71)

Dev infrastructure:

- Introduced Claude Code configuration for AI-assisted development. (RESTGroup/rstsr#72)

## v0.6.2 -- 2025-11-15

MSRV specified to 1.82.0, written to Cargo.toml.

Functionality changes:

- rstsr-core: `asarray` will not panic when layout is not compact if pass `&[T]` or `&mut [T]` to give view/mut-view tensor. However, pass `Vec<T>` to give owned tensor will still panic if layout is not compact.

## v0.6.0 -- 2025-11-03

Refactor:

- rstsr-core: Split `tensor/manuplication.rs` into a module with folder. (RESTGroup/rstsr#67)

API breaking changes:

- rstsr-core: Supporting type-promotion for assign. (RESTGroup/rstsr#65)
- rstsr-dtype-traits: Split previous dtype trait `PromotionAPI` to two parts `DTypePromoteAPI` and `DTypeCastAPI`. (RESTGroup/rstsr#66)
- Refactor of some dtype-related traits and impl. (RESTGroup/rstsr#66)
- rstsr-core: Add lifetime annotation to `TensorRefAPI` / `TensorRefMutAPI` (these two traits are currently not applied to any implementations). (RESTGroup/rstsr#67)
- rstsr-common: Added traceback in error-handling, which changed the fundamental data structure of `Error` of RSTSR. (RESTGroup/rstsr#67)
- rstsr-common: `DimBaseAPI::const_ndims` is method (with `&self` as argument) now. (RESTGroup/rstsr#67)

Enhancements:

- rstsr-dtype-traits: add function `isclose`. (RESTGroup/rstsr#66)
- rstsr-core: add function `allclose`. (RESTGroup/rstsr#66)
- rstsr-core: simplifies some trait bounds for obtaining view of tensor. (RESTGroup/rstsr#66)
- rstsr-core: Add macro `tensor_from_nested!` (similar to `ndarray::array!`). (RESTGroup/rstsr#67)
- rstsr-common: Added `rstsr_unwrap` to print unwrap with traceback. Added cargo feature `traceback` in many crates for printing traceback info of place of panic. (RESTGroup/rstsr#67)
- rstsr-common: Added `normalize_axes_index` for development (similar to `numpy.normalize_axis_tuple`). (RESTGroup/rstsr#67)
- rstsr-common: Added `Option<i/usize>` to `AxesIndex` trait implementation. (RESTGroup/rstsr#67)
- Some API documents updated. (RESTGroup/rstsr#67)

Fixes:

- Fixes some possible memory-safety problems of `uninitialized_vec` in reduce implementations in rstsr-native-impl. (RESTGroup/rstsr#66)
- Fix `no_std`. (RESTGroup/rstsr#67)
- Fixed `rt::expand_dims` and `rt::flip` in multiple axes cases (accordance to NumPy). (RESTGroup/rstsr#67)

## v0.6.0-alpha.1 -- 2025-09-30

This is not a completed version. May have other API breaking changes before v0.6.0 release.

API breaking changes:

- Using `DeviceRawAPI<MaybeUninit<T>>` instead of `DeviceRawAPI<T>` for output types in device operator traits, changing both parameter types and trait bounds (RESTGroup/rstsr#60).
- Using `ExtNum`, `ExtFloat`, `ExtReal` for trait extensions to crate `num`. Removed previous traits (per-functionality) `AbsAPI`, `ReImAPI`, etc. This affects trait bounds (RESTGroup/rstsr/#63).
- Using data type promotion rules in several common functions in CPU device, changing trait bounds (RESTGroup/rstsr#64).

Enhancements:

- Added TBLIS plugin (RESTGroup/rstsr#60). Now Einstein summation is available from `rt::tblis::einsum` (with cargo feature `rstsr/tblis` enabled, and linkage of libtblis.so).
- Added `uninit`, `assume_init` (RESTGroup/rstsr#61).
- Added `take`, `all`, `any`, `count_nonzero`, `nextafter`, `reciprocal` (RESTGroup/rstsr#63).
- Using data type promotion rules for several common functions in CPU device implementations (sin, greater, etc., making comparasion of different types, or sin to integer list be evaluatable) (RESTGroup/rstsr#64).

Refactor:

- Changed most internal device implementation that works with `empty_impl` to `uninit_impl` (RESTGroup/rstsr#61).
- Changed directory structure for device implementations (currently BLAS devices are categorized to directory `crates-device`) (RESTGroup/rstsr#62).
- Removed previous rstsr-book in this repository (RESTGroup/rstsr#63).

Parts of API document also updated. 

## v0.5.1 -- 2025-09-01

MSRV specified:

- 1.84.1: with crate faer built;
- 1.82.0: other cases (due to crate `half` and rust language usage of `unsafe extern "C"`).

Code refactor:

- Revert MSRV from 1.87 to 1.84/1.82 (RESTGroup/rstsr#59).

## v0.5.0 -- 2025-08-26

API breaking changes:

- Revert to non-default for dynamic loading (RESTGroup/rstsr#57)

Fixes (with API breaking behavior change):

- Change Behavior of `DeviceCpuRayon::generate_pool` (RESTGroup/rstsr#56, RESTGroup/rstsr-ffi#9)
- Fix KML threading lock in LAPACK functions (RESTGroup/rstsr#58)

## v0.4.1 -- 2025-08-05

Enhancements:

- Added CPU devices (backends) MKL ([#48](https://github.com/RESTGroup/rstsr/pull/48)), BLIS ([#49](https://github.com/RESTGroup/rstsr/pull/49)), AOCL ([#51](https://github.com/RESTGroup/rstsr/pull/51)), KML ([#53](https://github.com/RESTGroup/rstsr/pull/53))

Possible API breaking change:

- For conversion between CBLAS flags and RSTSR flags (defined in crate rstsr-common), previously CBLAS flags are in crate rstsr-lapack-ffi. Now those flags are defined in rstsr-cblas-base, and been applied in all FFI crates (at [RESTGroup/rstsr-ffi](https://github.com/RESTGroup/rstsr-ffi)).

Actions:

- Added ARM support ([#52](https://github.com/RESTGroup/rstsr/pull/52))

## v0.4.0 -- 2025-07-25

API breaking change: Supporting dynamic loading for OpenBLAS ([#47](https://github.com/RESTGroup/rstsr/pull/47))

- Update `rstsr-lapack-ffi` and `rstsr-openblas-ffi` version to v0.4.
- Default to `dynamic_loading` for using OpenBLAS.
- Changes internal ways to call BLAS and LAPACK functions.

If compile time and disk usage becomes very large for `rstsr-openblas-ffi`, you may wish to set those options in Cargo.toml:

```toml
[profile.dev.package.rstsr-lapack-ffi]
opt-level = 0
debug = false

[profile.dev.package.rstsr-openblas-ffi]
opt-level = 0
debug = false
```

## v0.3.10 -- 2025-07-22

Fix:
- Fix unpack_tri signature
- Fix gemm/syrk bug when k=0 ([#46](https://github.com/RESTGroup/rstsr/pull/46))

## v0.3.9 -- 2025-07-07

Enhancements:
- Feature with optional dependencies. Now in main crate `rstsr`, using feature `faer` and `openblas` along with `linalg` and `sci` should be ok, without explicitly declaring `rstsr-linalg-traits` and `rstsr-sci-traits` as dependencies. 

## v0.3.8 -- 2025-07-04

Bug Fix:
- Tested complex linalgs for Faer and OpenBLAS devices.

Enhancements:
- DeviceFaer: generalized eigen, triangular solve ([#44](https://github.com/RESTGroup/rstsr/pull/44))

## v0.3.7 -- 2025-06-25

Bug Fix:
- Fix panic when layout iterator size is zero ([#42](https://github.com/RESTGroup/rstsr/pull/42))

Enhancements:
- Add common function numa_refb and refb_numa implementation ([#42](https://github.com/RESTGroup/rstsr/pull/42))
- Implement clone for Tensor and TensorCow ([#42](https://github.com/RESTGroup/rstsr/pull/42))
- Added into_pack_array, into_unpack_array as associated function of TensorAny ([#42](https://github.com/RESTGroup/rstsr/pull/42))
- linalg: Solve-related functions supports vector (Ix1) RHS ([#43](https://github.com/RESTGroup/rstsr/pull/43))

## v0.3.6 -- 2025-06-05

Bug Fix:
- OpenBLAS device OpenMP `get_num_thread` function ([#40](https://github.com/RESTGroup/rstsr/pull/40))
    - Note that threading control is not stablized. There may be an incoming API breaking change on this feature for v0.4+.

Enhancements:
- Feature addition (meshgrid, concat, stack, bool_select) ([#38](https://github.com/RESTGroup/rstsr/pull/38))
- Feature addition (cdist, lebedev_rule) ([#39](https://github.com/RESTGroup/rstsr/pull/39))

Refactor:
- Eliminate `Error: From<I::Error>` trait bound ([#38](https://github.com/RESTGroup/rstsr/pull/38))

## v0.3.5 -- 2025-05-22

Bug Fix:
- Fix too strict stride check ([#36](https://github.com/RESTGroup/rstsr/pull/36))

API Breaking Change:
- Remove `into_slice_mut` ([#35](https://github.com/RESTGroup/rstsr/pull/35))

Enhancements:
- Diagonal arguments now allows i32 as input

## v0.3.4 -- 2025-05-20

Bug Fix:
- Fix rayon parallel in `op_muta_refb_func_cpu_rayon` ([#32](https://github.com/RESTGroup/rstsr/pull/32))
- Fix for conversion to self-device ([#33](https://github.com/RESTGroup/rstsr/pull/33))

## v0.3.3 -- 2025-05-19

Bug Fix:
- Fix Faer linalg functions when tensor offset != 0 ([#30](https://github.com/RESTGroup/rstsr/pull/30)).

## v0.3.2 -- 2025-05-19

Summary
- Added linalg functions for `DeviceFaer` ([#28](https://github.com/RESTGroup/rstsr/pull/28)).

API Breaking Change (user should not feel that):
- updates Faer version to v0.22, seems that v0.20/v0.21 changes handling logic for complex values
- Conversion from/to Faer made simple (but API breaking)
- Matmul made simple (but API breaking), now requires `faer::traits::ComplexField` type (trait impl based), instead of manually dispatch types (macro_rules based)

Enhancements:
- Functions added: cholesky, det, eigh (does not include generalized eigh), eigvalsh (same to eigh), inv, pinv, solve_general, svdvals

## v0.3.1 -- 2025-05-16

API Breaking Change:
- Remove `ge`, `gt`, `ne`, ... in traits `TensorGreaterAPI`, `TensorNotEqualAPI`, ... ([#25](https://github.com/RESTGroup/rstsr/pull/25))

Enhancements:
- linalg: functions added: slogdet, det, svd, eigvalsh, svdvals, pinv ([#23](https://github.com/RESTGroup/rstsr/pull/23))
- Summation to boolean tensor ([#25](https://github.com/RESTGroup/rstsr/pull/25))
- Basic advanced indexing function `index_select` ([#26](https://github.com/RESTGroup/rstsr/pull/26))
- Added TensorCow support for binary arithmetic operations ([#22](https://github.com/RESTGroup/rstsr/pull/22))

Something for fun:
- Changed logo to be ABBA-like style ([#24](https://github.com/RESTGroup/rstsr/pull/24))

## v0.3.0 -- 2025-05-09

API Breaking Change:
- Now `rt::linalg::eigh` returns `EighResult`, instead of simple 2-element tuple (eigenvalues, eigenvectors) ([#18](https://github.com/RESTGroup/rstsr/pull/18)).
- Now Lapack bindings will use Lapack (Fortran FFI) instead of LAPACKE (C FFI) by default ([#18](https://github.com/RESTGroup/rstsr/pull/18)).

Enhancements:
- Now `TensorCow` can perfrom binary arithmetic operations, such like `2.0 * a.reshape((2, 3, 4))`. Note that in some cases, rust compiler/rust-analyzer may not be able to deduce type of this result [#22](https://github.com/RESTGroup/rstsr/pull/18).
- Performed various refactor to linalg functions.
- Lapack (Fortran FFI) is supported and used by default. It is implemented like LAPACKE (but in rust), and many codes are generated by AI ([#18](https://github.com/RESTGroup/rstsr/pull/18)).

Various code refactors:
- Move more macro_rule implementatios to duplicate.

## v0.2.7 -- 2025-04-15

Internal refactor:
- Move out crate `rstsr-openblas-ffi` to rstsr-ffi repository, changes FFI bindings ([#19](https://github.com/RESTGroup/rstsr/pull/19)).
- Now `row_major` and `col_major` features are mutually exclusive in complie time.

## v0.2.6 -- 2025-04-02

Feature addition:
- Column major is now supported ([#16](https://github.com/RESTGroup/rstsr/pull/16)).

## v0.2.5 -- 2025-03-31

Bug fix:
- fix `pack_tril` (correctness fix for col-major case).

## v0.2.4 -- 2025-03-25

Bug fix:
- fix `pack_tril` (correctness fix, trait bound fix).

## v0.2.2 -- 2025-03-25

API breaking changes:
- Rayon thread pool getter function `get_pool` changed, added `get_current_pool`, removed `get_serial_pool` ([#14](https://github.com/RESTGroup/rstsr/pull/14)).

Code style changes ([#15](https://github.com/RESTGroup/rstsr/pull/15))

## v0.2.1 -- 2025-03-24

Bug fix:
- fix rayon pool memory blow up ([#13](https://github.com/RESTGroup/rstsr/pull/13))

## v0.2.0 -- 2025-03-11

This release features on BLAS and linalg implementations. Currently, functions such as `cholesky`, `eigh`, `solve_general` in `rstsr-linalg-traits` have been implemented.

Also many enhancements in `rstsr-core`.

## v0.1.0

Initial release. Most features in Python Array API has been implemented.
