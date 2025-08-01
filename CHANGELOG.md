# Changelog

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
