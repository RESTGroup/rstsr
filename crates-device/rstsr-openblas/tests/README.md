# rstsr-openblas tests

Entry binaries (ADR-0012, entry-binary test matrix):

| Target | Body | Notes |
|---|---|---|
| `entry_row_openblas` | `core_func/` + `test_utils/` + `test_issues/` (symlinked from rstsr-core) | shared device-agnostic body, row-major `TESTCFG` |
| `tests_driver_impl` | `driver_impl/` (symlinked from `rstsr-blas-traits/src/blas_tests_impl/`) | BLAS driver conformance, `DeviceBLAS` = `DeviceOpenBLAS` |
| `tests_linalg_func` | `linalg_func/` (same source, feature `linalg`) | LAPACK-backed linalg conformance |
| `tests_issues` | `issues/` (real files) | backend-specific regression tests (issues found on OpenBLAS) |
| `tests_zero_size` | `zero_size/` (real files) | zero-size BLAS/LAPACK operand coverage (issue #161, PR #162) |
| `test_workable` | real file | smoke test: threaded matmul via OpenBLAS |
| `trait_bound` | real file | trait-bundle sufficiency: generic code bounded only by `LapackDriverAPI<T>` can call the core matmul entry points and the linalg-tier members of `BlasDriverBaseAPI` (complex counterpart, vector assignment, index dtypes, in-place mutation, n-dim norm ops) |

## Divergences from the shared bodies

None at entry level. Per-file dedicated exceptions (ADR-0012 §5): none yet.
