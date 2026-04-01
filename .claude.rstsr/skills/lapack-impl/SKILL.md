---
name: lapack-impl
description: Implementation guide for LAPACK driver functions in `rstsr-blas-traits/driver_impl/lapack/`.
---

# LAPACK Driver Implementation Guide

This skill guides implementation of LAPACK driver functions in `rstsr-blas-traits/driver_impl/lapack/`.

## Overview

The rstsr project implements LAPACK functionality through a layered architecture:
- **Trait layer** (`src/lapack_*/`): Defines `XXXDriverAPI` traits and `XXX_` structs (e.g., `SYEVDriverAPI`, `SYEV_`)
- **Driver implementation** (`driver_impl/lapack/`): Implements traits using raw LAPACK Fortran interface (column-major, with manual row-major handling)
- **LAPACKE implementation** (`driver_impl/lapacke/`): Implements traits using LAPACKE C interface (handles row-major internally)

## Directory Structure

```
rstsr-blas-traits/
├── src/
│   ├── lapack_eig/     # General eigenvalue trait definitions (geev)
│   ├── lapack_eigh/    # Eigenvalue trait definitions (syev, syevd, sygv, sygvd)
│   ├── lapack_qr/      # QR trait definitions (geqrf, orgqr, geqp3, ormqr)
│   ├── lapack_solve/   # Linear solve trait definitions (gesv, getrf, getri, potrf, sysv)
│   ├── lapack_svd/     # SVD trait definitions (gesvd, gesdd)
│   └── trait_def.rs    # Aggregates all traits into LapackDriverAPI
├── driver_impl/
│   ├── lapack/         # Raw LAPACK implementations (requires row-major handling)
│   │   ├── eig/        # geev.rs
│   │   ├── eigh/       # syev.rs, syevd.rs, sygv.rs, sygvd.rs
│   │   ├── qr/         # geqrf.rs, orgqr.rs, geqp3.rs, ormqr.rs
│   │   ├── solve/      # gesv.rs, getrf.rs, getri.rs, potrf.rs, sysv.rs
│   │   └── svd/        # gesvd.rs, gesdd.rs
│   └── lapacke/        # LAPACKE implementations (simpler, handles row-major)
│       ├── eig/
│       ├── eigh/
│       ├── qr/
│       ├── solve/
│       └── svd/
```

## Implementation Steps

### 1. Study LAPACKE Reference

Reference LAPACKE source at `../other-repos/lapack/LAPACKE/src/lapacke_<func>.c`:

```bash
# Example: study syev implementation
cat ../other-repos/lapack/LAPACKE/src/lapacke_dsyev.c
cat ../other-repos/lapack/LAPACKE/src/lapacke_dsyev_work.c
```

Key observations:
- High-level `LAPACKE_dsyev` handles work array allocation and calls `_work` variant
- `_work` variant handles row/col-major conversion via `LAPACKE_dsy_transpose` macros
- Header signatures in `../other-repos/lapack/LAPACKE/include/lapacke.h`

### 2. Define Trait API

Create trait in `src/lapack_<category>/<func>.rs`:

```rust
pub trait XXXDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_xxx(
        order: FlagOrder,
        // ... LAPACKE-style parameters (matrix_layout first)
        // Use usize for sizes, FlagUpLo/FlagSide/FlagTrans for enums
    ) -> blas_int;
}
```

**Important**: Use existing flag types from `rstsr-common/src/flags.rs`:
- `FlagOrder` - RowMajor/ColMajor (C/F)
- `FlagTrans` - N/T/C/CN (NoTrans/Trans/ConjTrans)
- `FlagSide` - L/R (Left/Right)
- `FlagUpLo` - U/L (Upper/Lower)
- `FlagDiag` - N/U (NonUnit/Unit)

Do NOT define new flag types - reuse the existing ones which already have `From` implementations for `char` and `c_char`.

### 3. Define Struct with Builder Pattern

Use `derive_builder` for ergonomic API:

```rust
#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct XXX_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    // ... other parameters with defaults
}

impl<'a, B, T> XXX_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + XXXDriverAPI<T>,
{
    pub fn run(self) -> Result<(OutputTypes)> {
        // Implementation logic
    }
}
```

### 4. Implement LAPACKE Driver (Simple Path)

In `driver_impl/lapacke/<category>/<func>.rs`:

```rust
use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_sxxx];
   [f64] [LAPACKE_dxxx];
)]
impl XXXDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_xxx(
        order: FlagOrder,
        // ... parameters with flag types (FlagSide, FlagTrans, etc.)
    ) -> blas_int {
        use std::os::raw::c_char;
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            side.into() as c_char,  // Use .into() for flag conversions
            trans.into() as c_char,
            // ... cast parameters (m as _, lda as _)
        )
    }
}

// For complex types (if applicable):
#[duplicate_item(
    T              lapacke_func  ;
   [Complex::<f32>] [LAPACKE_cxxx];  // NOTE: Use turbofish syntax!
   [Complex::<f64>] [LAPACKE_zxxx];
)]
impl XXXDriverAPI<T> for DeviceBLAS {
    // ... similar implementation with pointer casts
}
```

**Critical**: Use turbofish syntax `[Complex::<f32>]` in the duplicate macro, NOT `[Complex<f32>]` - Rust will interpret `<` and `>` as comparison operators otherwise.

### 5. Implement Raw LAPACK Driver (Complex Path)

In `driver_impl/lapack/<category>/<func>.rs`:

Pattern for real types:

```rust
#[duplicate_item(
    T     func_   ;
   [f32] [sxxx_];
   [f64] [dxxx_];
)]
impl XXXDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_xxx(
        order: FlagOrder,
        // ... parameters
    ) -> blas_int {
        use lapack_ffi::lapack::func_;
        use num::Zero;

        if order == ColMajor {
            // 1. Query work array size
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(/* query call with lwork = -1 */);
            let lwork = work_query.to_usize().unwrap();

            // 2. Allocate work arrays
            let mut work: Vec<T> = uninitialized_vec(lwork).unwrap();

            // 3. Direct call
            func_(/* direct call */);
            info
        } else {
            // Row-major: transpose first, then query, then call
            // ... see Row-Major Handling section below
        }
    }
}
```

**Critical for Row-Major**: The work array query MUST be done AFTER transposing input matrices, using the column-major leading dimensions. Querying with row-major lda values will give incorrect results.

**CRITICAL BUG FIX (2026-03-31)**: For non-square matrices, the work query was incorrectly using the input `lda` (row-major stride) instead of `lda_t` (column-major leading dimension for transposed matrix). This caused LAPACK error -4 (illegal parameter value).

**Correct pattern for row-major work query**:
```rust
if order == ColMajor {
    // Query with input lda
    func_(/* ... */, &(lda as _), /* ... */);
} else {
    let lda_t = m.max(1);  // Column-major LDA for transposed matrix

    // Query with lda_t, NOT lda! Use null pointer for matrix data
    func_(/* ... */, std::ptr::null_mut(), &(lda_t as _), /* ... */);
}
```

### 6. Row-Major Handling Pattern

For functions with multiple matrices (e.g., ORMQR with A and C):

```rust
if order == ColMajor {
    // Query and call directly with lda, ldc
} else {
    // 1. Calculate column-major leading dimensions
    let lda_t = rows_of_a.max(1);
    let ldc_t = rows_of_c.max(1);

    // 2. Allocate temporary buffers
    let size_a = r * k;
    let size_c = m * n;
    let mut a_t: Vec<T> = uninitialized_vec(size_a)?;
    let mut c_t: Vec<T> = uninitialized_vec(size_c)?;

    // 3. Transpose inputs from row-major to column-major
    let a_slice = from_raw_parts_mut(a as *mut T, size_a);
    let la = Layout::new_unchecked([r, k], [lda as isize, 1], 0);
    let la_t = Layout::new_unchecked([r, k], [1, lda_t as isize], 0);
    orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la)?;

    // ... same for C matrix

    // 4. Query work array size WITH column-major dimensions
    let mut work_query: T = T::zero();
    func_(
        &side.into(),
        &trans.into(),
        /* ... */,
        a_t.as_ptr(),
        &(lda_t as _),  // Column-major LDA
        /* ... */,
        c_t.as_mut_ptr(),
        &(ldc_t as _),  // Column-major LDC
        &mut work_query,
        &lwork,
        &mut info,
    );
    let lwork = work_query.to_usize().unwrap();

    // 5. Allocate work array
    let mut work: Vec<T> = uninitialized_vec(lwork)?;

    // 6. Call LAPACK with column-major buffers
    func_(/* ... with a_t, c_t, lda_t, ldc_t */);

    // 7. Transpose outputs back to row-major
    orderchange_out_c2r_ix2_cpu_serial(c_slice, &lc, &c_t, &lc_t)?;

    info
}
```

Key functions:
- `orderchange_out_r2c_ix2_cpu_serial(&mut dst, &layout_dst, src, &layout_src)` - row→col
- `orderchange_out_c2r_ix2_cpu_serial(dst, &layout_dst, &src, &layout_src)` - col→row

### 7. Update Module Exports

Add to `src/lapack_<category>/mod.rs`:
```rust
pub mod xxx;
pub use xxx::*;
```

Add to `src/trait_def.rs` in `LapackDriverAPI`:
```rust
+ XXXDriverAPI<T>
```

Add to `driver_impl/lapack/<category>/mod.rs`:
```rust
pub mod xxx;
```

### 8. Testing with Python Validation

**Step 1: Create Python script for fingerprint computation**

Create a script in `tmp/` directory to compute expected fingerprint values:

```python
#!/usr/bin/env python3
import numpy as np
from scipy import linalg
import os

# Get the path to the test data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'rstsr-test-manifest', 'resources')

def fingerprint(arr):
    """Compute fingerprint matching Rust's rstsr_core::prelude_dev::fingerprint"""
    arr = np.asarray(arr).flatten()
    indices = np.arange(len(arr))
    return np.sum(np.cos(indices) * arr)

def sorted_eigenvalue_fingerprint(w):
    """For general eigenvalues (unsorted by LAPACK)."""
    if np.iscomplexobj(w):
        w_sorted = sorted(w, key=lambda x: (x.real, x.imag))
    else:
        w_sorted = sorted(w)
    return fingerprint([v.real for v in w_sorted])

# Load test data
a = np.load(os.path.join(DATA_DIR, 'a-f64.npy'))
b = np.load(os.path.join(DATA_DIR, 'b-f64.npy'))

# Example: GGEV test
a_32 = a[:32*32].reshape(32, 32)
b_32 = b[:32*32].reshape(32, 32)
alphar, alphai, beta, vl, vr, work, info = linalg.lapack.dggev(a_32, b_32)

# Compute fingerprint for validation
eigenvalues = [complex(alphar[i], alphai[i]) / beta[i] for i in range(len(alphar)) if abs(beta[i]) > 1e-15]
fp = sorted_eigenvalue_fingerprint(eigenvalues)
print(f"GGEV eigenvalue fingerprint: {fp}")
```

**Step 2: Write Rust test with fingerprint assertions**

```rust
use rstsr_blas_traits::lapack_eig::*;
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
use rstsr_test_manifest::get_vec;

/// Use .raw() to access underlying slice (zero-copy for views)
fn sorted_ggev_eigenvalue_fingerprint(
    alphar: &Tensor<f64, DeviceBLAS, Ix1>,
    alphai: &Tensor<f64, DeviceBLAS, Ix1>,
    beta: &Tensor<f64, DeviceBLAS, Ix1>,
) -> f64 {
    use num::Complex;
    let alphar_v = alphar.raw();  // Zero-copy access to slice
    let alphai_v = alphai.raw();
    let beta_v = beta.raw();
    // ... compute fingerprint
}

#[test]
fn test_dggev() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [32, 32].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((get_vec::<f64>('b'), [32, 32].c(), &device)).into_dim::<Ix2>();

    let driver = GGEV::default().a(a.view()).b(b.view()).build().unwrap();
    let (alphar, alphai, beta, _, _, _, _) = driver.run().unwrap();

    let fp = sorted_ggev_eigenvalue_fingerprint(&alphar, &alphai, &beta);
    assert!((fp - expected_value).abs() < 1e-8);
}
```

**Key patterns for tensor data access:**
- `.raw()` → `&[T]` for views (zero-copy)
- `.raw_mut()` → `&mut [T]` for owned tensors
- `.to_vec()` → `Vec<T>` for owned conversion
- `.into_vec()` → `Vec<T>` takes ownership (zero-cost)

**Test command:**
```bash
RUST_BACKTRACE=1 RSTSR_DEV=1 cargo test -p rstsr-openblas --test tests_driver_impl \
    --features "rstsr/backtrace openmp linalg" \
    -- test_ggev --test-threads=1 --nocapture
```

## Key Patterns

### Duplicate Macro

Use `duplicate::duplicate_item` for type polymorphism:
- Real types: `[f32] [sxxx_]`, `[f64] [dxxx_]`
- Complex types: `[Complex::<f32>] [cxxx_]`, `[Complex::<f64>] [zxxx_]` (turbofish!)
- LAPACKE functions: `[LAPACKE_sxxx]`, `[LAPACKE_dxxx]`, etc.

### Flag Type Conversions

Flag types have `From` implementations for `char` and `c_char`:

```rust
use rstsr_common::flags::{Left, Trans};  // Convenient aliases

// In trait signature:
fn driver_xxx(side: FlagSide, trans: FlagTrans, ...) -> blas_int;

// In implementation:
&side.into()  // Returns c_char for LAPACK FFI
side.into() as c_char  // Explicit for LAPACKE
```

### Char Parameters

Some LAPACK functions use plain `char` parameters that don't correspond to existing flag types (e.g., `jobvl`, `jobvr` in GEEV). For these:

```rust
// In trait signature - use char directly
unsafe fn driver_geev(
    order: FlagOrder,
    jobvl: char,  // 'N' or 'V'
    jobvr: char,  // 'N' or 'V'
    ...
) -> blas_int;

// In LAPACKE implementation - cast with as _
lapack_ffi::lapacke::lapacke_func(
    order as _,
    jobvl as _,  // Rust char to c_char works for ASCII
    jobvr as _,
    ...
)

// In raw LAPACK implementation - pass as reference
func_(
    &(jobvl as _),  // Pass as reference for Fortran interface
    &(jobvr as _),
    ...
)
```

**Note**: Rust's `char` is 4 bytes but the `as _` cast to `c_char` (1 byte) works correctly for ASCII characters like 'N', 'V', 'U', 'L'.

### Work Array Query

Many LAPACK functions require work arrays:
1. Call with `lwork = -1` to query optimal size
2. `work_query` receives recommended size
3. Allocate `work` array with that size
4. Call again with proper `lwork`

**For row-major**: Query AFTER transposing, with column-major lda values.

**Critical: Work Query Type Declaration**

When declaring `work_query` for LAPACK work array size queries, ALWAYS use explicit type annotation or type-inferred zero:

```rust
// CORRECT: Type annotation with T::zero()
let mut work_query: T = T::zero();

// CORRECT: Type-inferred zero (T is inferred from context)
let mut work_query = T::zero();

// WRONG: Defaults to f64, incorrect for f32 and Complex types!
let mut work_query = 0.0;  // NEVER use this!
```

**Why this matters**:
- `0.0` defaults to `f64` in Rust, which is wrong for `f32` types
- For complex types, work arrays are typically `Complex<T>`, not `T`
- Using the wrong type causes pointer casts (`as *mut _ as *mut _`) to pass incorrect data to LAPACK

**For rwork in complex functions**: Use `<T as ComplexFloat>::Real::zero()`:
```rust
// For complex functions with rwork (real-valued work array)
let mut rwork_query = <T as ComplexFloat>::Real::zero();
```

**Required imports**:
```rust
use num::{ToPrimitive, Zero};
// or for complex functions:
use num::{Complex, ToPrimitive, Zero};
use num::complex::ComplexFloat;
```

### Error Codes

Standard LAPACK info codes:
- `info = 0`: success
- `info < 0`: parameter error (e.g., `-1010` for memory allocation failure)
- `info > 0`: algorithmic error (e.g., convergence failure)

Use `rstsr_errcode!(info, "Lapack XXX")?` for error propagation.

### Real vs Complex Type Handling

Some LAPACK functions have different signatures for real and complex types. For example, GEEV:

**Real types (f32, f64)**:
```rust
// Trait signature
unsafe fn driver_geev(
    ...
    wr: *mut T::Real,  // Real part of eigenvalues
    wi: *mut T::Real,  // Imaginary part of eigenvalues
    ...
) -> blas_int;
```

**Complex types (Complex<f32>, Complex<f64>)**:
```rust
// LAPACKE implementation needs temporary array
let mut w: Vec<T> = uninitialized_vec(n)?;
// Call LAPACKE with single complex array
LAPACKE_zgeev(..., w.as_mut_ptr(), ...);
// Split into real/imaginary parts
for i in 0..n {
    *wr.add(i) = w[i].re;
    *wi.add(i) = w[i].im;
}
```

This unified interface allows the higher-level `GEEV_` struct to work consistently across all types.

### Layout Helpers

```rust
let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);  // row-major input
let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);  // col-major temp
```

### Zero-Cost Tensor to Vec Conversion

For contiguous tensors of primitive types:
```rust
let jpvt: Tensor<i32, DeviceBLAS, Ix1> = /* ... */;
let vec: Vec<i32> = jpvt.into_vec();  // Zero-cost, just takes ownership of data
```

## Reference Locations

- **LAPACKE source**: `../other-repos/lapack/LAPACKE/src/lapacke_<func>.c`
- **LAPACKE headers**: `../other-repos/lapack/LAPACKE/include/lapacke.h`
- **Existing implementations**:
  - `rstsr-blas-traits/driver_impl/lapack/eigh/syev.rs`
  - `rstsr-blas-traits/driver_impl/lapack/qr/ormqr.rs` (complex row-major example)
- **Tests**: `crates-device/rstsr-openblas/tests/driver_impl/lapack_*_f64.rs`
- **Trait definitions**: `rstsr-blas-traits/src/lapack_*/`
- **NumPy reference**: `../other-repos/numpy/numpy/linalg/_linalg.py`
- **SciPy reference**: `../other-repos/scipy/scipy/linalg/`

## Common LAPACK Functions by Category

### General Eigenvalue (eig)
- `geev`: General eigenvalues and eigenvectors
  - For real types: returns separate wr (real part) and wi (imaginary part) arrays
  - For complex types: returns single complex w array
  - Has left (`vl`) and right (`vr`) eigenvector options
- `ggev`: Generalized eigenvalues and eigenvectors (A*v = λ*B*v)
  - Eigenvalues returned as (alpha, beta) pairs where λ = alpha/beta
  - For real types: returns alphar, alphai, beta arrays
  - For complex types: returns complex alpha, beta arrays
  - Has left (`vl`) and right (`vr`) eigenvector options
- Future: `geevx`, `ggevd`

### Eigenvalue (eigh)
- `syev/syevd`: Symmetric eigenvalues
- `sygv/sygvd`: Symmetric generalized eigenvalues
- `heev/heevd`: Hermitian eigenvalues (complex)
- Future: `syevr`, `syevx`, `heevr`, `heevx`

### QR
- `geqrf`: Basic QR factorization (returns QR in packed form + tau)
- `orgqr/ungqr`: Generate Q matrix from QR (real/complex)
- `geqp3`: QR with column pivoting (rank-revealing QR)
- `ormqr/unmqr`: Multiply Q with matrix without forming Q explicitly (real/complex)
- Future: `gelqf`, `orglq`, `ormlq`, `geqlf`, `gerqf`

### SVD
- `gesvd`: Standard SVD
- `gesdd`: Divide-and-conquer SVD
- Future: `gesvdx`, `gesvdq`

### Linear Solve
- `gesv`: General solve
- `getrf/getri`: LU factorization + inverse
- `potrf`: Cholesky factorization
- `sysv`: Symmetric solve
- Future: `posv`, `gels`, `gelsy`, `gelss`, `gelsd`

## Checklist for New Function

1. [ ] Study LAPACKE reference (`lapacke_<func>.c` and `_work.c`)
2. [ ] Check `rstsr-common/src/flags.rs` for existing flag types
3. [ ] Define `XXXDriverAPI` trait in `src/lapack_<category>/<func>.rs`
4. [ ] Define `XXX_` struct with builder pattern
5. [ ] Implement trait in `driver_impl/lapacke/<category>/<func>.rs`
6. [ ] Implement trait in `driver_impl/lapack/<category>/<func>.rs`
   - [ ] Handle ColMajor: query + direct call
   - [ ] Handle RowMajor: transpose → query → call → transpose back
7. [ ] Add exports to `mod.rs` files
8. [ ] Add trait to `LapackDriverAPI` in `trait_def.rs`
9. [ ] Write Python validation to get fingerprint values
10. [ ] Write Rust tests with `fingerprint` verification
11. [ ] Run `cargo fmt` and `cargo clippy`
12. [ ] Test on at least one device crate

## Important Notes

### Eigenvalue Ordering (GEEV, etc.)

**CRITICAL**: LAPACK does NOT guarantee eigenvalue ordering. Different implementations return eigenvalues in different orders:

| Implementation | Used By | Ordering |
|----------------|---------|----------|
| MKL | scipy | Implementation-specific |
| OpenBLAS | rstsr | Implementation-specific |
| Reference LAPACK | - | Implementation-specific |

**Testing pattern for eigenvalue correctness**:
```rust
/// Compute fingerprint of sorted eigenvalues (implementation-agnostic)
fn sorted_eigenvalue_fingerprint(wr: &Tensor<f64, DeviceBLAS, Ix1>, wi: &Tensor<f64, DeviceBLAS, Ix1>) -> f64 {
    use num::Complex;
    let wr_vec: Vec<f64> = wr.iter().copied().collect();
    let wi_vec: Vec<f64> = wi.iter().copied().collect();
    let n = wr_vec.len();
    let mut eigenvalues: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(wr_vec[i], wi_vec[i])).collect();
    eigenvalues.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap().then_with(|| a.im.partial_cmp(&b.im).unwrap()));
    let sorted_wr: Vec<f64> = eigenvalues.iter().map(|c| c.re).collect();
    (0..n).map(|i| (i as f64).cos() * sorted_wr[i]).sum()
}
```

**Why this matters**:
- Eigenvalue fingerprint tests using raw order will fail across different LAPACK implementations
- Eigenvector fingerprints also depend on eigenvalue ordering
- For symmetric matrices (EIGH), eigenvalues ARE sorted by LAPACK, so direct comparison works
- For general matrices (EIG/GEEV), you MUST sort before comparing

**Eigenvector comparison**: More complex because eigenvectors for complex eigenvalue pairs are stored in two columns (real + imaginary parts). The ordering matches the eigenvalue ordering, making cross-implementation comparison difficult.