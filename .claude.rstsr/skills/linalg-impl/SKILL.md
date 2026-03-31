---
name: linalg-impl
description: Implementation guide for linalg functions in `rstsr-linalg-traits/`.
---

# Linalg Function Implementation Guide

This skill guides implementation of high-level linalg functions in `rstsr-linalg-traits/`.

## Overview

The rstsr project implements linalg functionality through a layered architecture:
- **Trait layer** (`traits_def.rs`): Defines `XXXAPI` traits, result structs, and optional args builders
- **Reference implementation** (`ref_impl_blas.rs`): Internal functions that call LAPACK drivers
- **Device implementation** (`blas_impl/`): Overloaded implementations for `DeviceBLAS`
- **Device exports** (`crates-device/*/linalg_auto_impl/`): Re-exports for device crates

## Directory Structure

```
rstsr-linalg-traits/
├── traits_def.rs        # Trait definitions, result structs, args builders
├── ref_impl_blas.rs     # Reference implementations using LAPACK drivers
├── lib.rs               # Module exports
├── prelude.rs           # Public exports (rstsr_traits, rstsr_funcs, rstsr_structs)
└── prelude_dev.rs       # Developer exports

crates-device/rstsr-openblas/
└── src/linalg_auto_impl/
    ├── mod.rs           # Module exports
    ├── cholesky.rs      # Simple overloading pattern
    ├── svd.rs           # Mode handling pattern
    ├── eigh.rs          # Args builder pattern
    └── qr.rs            # QR implementation
```

## Implementation Patterns

### Pattern 1: Simple Overloading (like Cholesky)

For functions with simple arguments (1-2 parameters), use direct overloading without args builder.

**When to use**: Function has optional parameters with sensible defaults (e.g., `uplo: Option<FlagUpLo>`)

```rust
// traits_def.rs
#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [CholeskyAPI       ] [cholesky        ] [cholesky_f        ];
)]
pub trait LinalgAPI<Inp> {
    type Out;
    fn func_f(self) -> Result<Self::Out>;
    fn func(self) -> Self::Out { ... }
}

// blas_impl/cholesky.rs
#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> CholeskyAPI<DeviceBLAS> for (Tr, Option<FlagUpLo>) {
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> { ... }
}
```

### Pattern 2: Mode Handling (like SVD)

For functions with multiple modes, handle mode selection in overloading.

**When to use**: Function has `mode` parameter affecting output shape/type

```rust
// traits_def.rs - Result struct
pub struct SVDResult<U, S, Vt> {
    pub u: U,
    pub s: S,
    pub vt: Vt,
}

// Optional args builder for advanced usage
#[derive(Builder)]
pub struct SVDArgs_<'a, B, T> {
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(default = "Some(true)")]
    pub full_matrices: Option<bool>,
}

// blas_impl/svd.rs - Mode-based overloading
impl<T, D, R> SVDAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, bool) {
    type Out = SVDResult<Tensor<T, DeviceBLAS, D>, ...>;
    fn svd_f(self) -> Result<Self::Out> {
        let (a, full_matrices) = self;
        let svd_args = SVDArgs::default().a(a).full_matrices(full_matrices).build()?;
        ref_impl_svd_simple_f(svd_args)
    }
}
```

### Pattern 3: Complex Args Builder (like Eigh)

For functions with many configuration options, use args builder pattern.

**When to use**: Function has 3+ optional parameters, subsets, driver selection, etc.

```rust
// traits_def.rs
#[derive(Builder)]
pub struct EighArgs_<'a, 'b, B, T> {
    pub a: TensorReference<'a, T, B, Ix2>,
    pub b: Option<TensorReference<'b, T, B, Ix2>>,
    pub uplo: Option<FlagUpLo>,
    pub eigvals_only: bool,
    pub eig_type: i32,
    pub subset_by_index: Option<(usize, usize)>,
    pub subset_by_value: Option<(T::Real, T::Real)>,
    pub driver: Option<&'static str>,
}
```

## Implementation Steps

### 1. Define Result Structs

Create result struct(s) in `traits_def.rs`:

For functions with multiple modes, use a unified struct with Option fields:

```rust
/// Unified QR decomposition result for all modes
///
/// Different modes populate different fields:
/// - "reduced": q=Some(Q), r=Some(R), p=Some/None
/// - "complete": q=Some(Q), r=Some(R), p=Some/None
/// - "r": q=None, r=Some(R), p=Some/None
/// - "raw": q=None, r=None, h=Some(H), tau=Some(tau), p=Some/None
pub struct QRResult<Q, R, H, Tau, P> {
    pub q: Option<Q>,     // Orthogonal matrix (reduced/complete)
    pub r: Option<R>,     // Upper triangular R (reduced/complete/r)
    pub h: Option<H>,     // Packed Householder matrix (raw)
    pub tau: Option<Tau>, // Tau vector (raw)
    pub p: Option<P>,     // Pivot indices (if pivoting)
}
```

Implement `From` for tuple conversions:

```rust
impl<Q, R, H, Tau, P> From<(Option<Q>, Option<R>, Option<H>, Option<Tau>, Option<P>)> for QRResult<Q, R, H, Tau, P> {
    fn from((q, r, h, tau, p): (Option<Q>, Option<R>, Option<H>, Option<Tau>, Option<P>)) -> Self {
        Self { q, r, h, tau, p }
    }
}
```

**Key pattern**: Instead of creating multiple result structs (QRResult, QRResultR, QRResultRaw), use a single unified struct with Option fields. This simplifies API and avoids code duplication.
```

### 2. Define Trait

Use `#[duplicate_item]` to generate trait and functions:

```rust
#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [QRAPI             ] [qr               ] [qr_f             ];
)]
pub trait LinalgAPI<Inp> {
    type Out;
    fn func_f(self) -> Result<Self::Out>;
    fn func(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::func_f(self).rstsr_unwrap()
    }
}

#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [QRAPI             ] [qr               ] [qr_f             ];
)]
pub fn func_f<Args, Inp>(args: Args) -> Result<<Args as LinalgAPI<Inp>>::Out>
where
    Args: LinalgAPI<Inp>,
{
    Args::func_f(args)
}
```

### 3. Implement Reference Function

In `ref_impl_blas.rs`, create internal implementation:

```rust
pub fn ref_impl_qr_f<'a, T, B>(
    a: TensorReference<'a, T, B, Ix2>,
    mode: &'static str,
    pivoting: bool,
) -> Result<(Option<Tensor<T, B, Ix2>>, Tensor<T, B, Ix2>, Option<Tensor<blas_int, B, Ix1>>)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());

    let task = || {
        // 1. Call GEQRF or GEQP3
        let (qr, tau, p) = if pivoting {
            let (qr, jpvt, tau) = GEQP3::default().a(a).build()?.run()?;
            (qr, tau, Some(jpvt))
        } else {
            let (qr, tau) = GEQRF::default().a(a).build()?.run()?;
            (qr, tau, None)
        };

        // 2. Extract R (upper triangular)
        // 3. Generate Q if needed using ORGQR
        // 4. Return result based on mode
    };

    device.with_blas_num_threads(nthreads, task)
}
```

### 4. Implement Overloading

In `blas_impl/<func>.rs`, implement trait for various input types:

```rust
use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

// Full arguments
impl<T, D, R> QRAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, &'static str, bool)
where
    R: DataAPI<Data = Vec<T>>,
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = QRResult<Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, Tensor<blas_int, DeviceBLAS, Ix1>>;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode, pivoting) = self;
        let a = a.view().into_dim::<Ix2>();
        let (q, r, p) = ref_impl_qr_f(a.into(), mode, pivoting)?;
        Ok(QRResult { q: q.unwrap(), r, p })
    }
}

// Default arguments
impl<T, D, R> QRAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    // same bounds
{
    type Out = /* ... */;
    fn qr_f(self) -> Result<Self::Out> {
        QRAPI::<DeviceBLAS>::qr_f((self, "reduced", false))
    }
}
```

### 5. Update Exports

In `prelude.rs`:

```rust
pub mod rstsr_traits {
    pub use crate::traits_def::{QRAPI, /* other traits */};
}

pub mod rstsr_funcs {
    pub use crate::traits_def::{qr, qr_f, /* other funcs */};
}

pub mod rstsr_structs {
    pub use crate::traits_def::{QRResult, /* other structs */};
}
```

### 6. Add Device Module

In `crates-device/rstsr-openblas/src/linalg_auto_impl/mod.rs`:

```rust
pub mod qr;
```

The implementation file `crates-device/rstsr-openblas/src/linalg_auto_impl/qr.rs` implements the trait for `DeviceBLAS`:

```rust
use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

// Implement trait for various input types
impl<T, D, R> QRAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, &'static str, bool)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = QRResult<Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, Tensor<blas_int, DeviceBLAS, Ix1>>;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode, pivoting) = self;
        // ... implementation using ref_impl_qr_f
    }
}
```

## Key Patterns

### duplicate_item Macro

Used for generating multiple trait/function definitions:

```rust
#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [CholeskyAPI       ] [cholesky        ] [cholesky_f        ];
   [QRAPI             ] [qr               ] [qr_f             ];
   [SVDAPI            ] [svd             ] [svd_f             ];
)]
```

### Overloading Variants

Handle different tensor ownership:

```rust
// View types (return owned tensor)
[&TensorAny<R, T, DeviceBLAS, D>]
[TensorView<'_, T, DeviceBLAS, D>]

// Mutable types (may reuse storage)
[TensorMut<'a, T, DeviceBLAS, D>]
[Tensor<T, DeviceBLAS, D>]
```

### Mode Handling

For functions with modes, use static string for compile-time dispatch:

```rust
impl QRAPI<DeviceBLAS> for (&TensorRef, &'static str, bool) {
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode, pivoting) = self;
        match mode {
            "reduced" => /* ... */,
            "complete" => /* ... */,
            "r" => /* ... */,
            "raw" => /* ... */,
            _ => rstsr_invalid!(mode)?,
        }
    }
}
```

## Reference Locations

- **Trait definitions**: `rstsr-linalg-traits/src/traits_def.rs`
- **Reference implementations**: `rstsr-linalg-traits/src/ref_impl_blas.rs`
- **Device implementations**: `rstsr-linalg-traits/src/blas_impl/`
- **LAPACK drivers**: `rstsr-blas-traits/src/lapack_*/`
- **NumPy reference**: `../other-repos/numpy/numpy/linalg/_linalg.py`
- **SciPy reference**: `../other-repos/scipy/scipy/linalg/`

## Key Utility Functions

### `overwritable_convert`

Located in `rstsr-blas-traits/src/util/util_tensor.rs`.

Converts a tensor reference to a mutable tensor with optimal layout for LAPACK:

```rust
/// Convert a tensor reference to a mutable tensor with optimal layout for LAPACK.
///
/// This function converts the input tensor to a contiguous layout suitable for
/// LAPACK operations:
/// - F-prefer (column-major prefer) tensors are converted to ColMajor (stride=[1, ld])
/// - C-prefer (row-major prefer) tensors are converted to RowMajor (stride=[ld, 1])
///
/// The conversion is "overwritable" meaning:
/// - If the input is a reference (read-only), a new owned tensor is created
/// - If the input is already mutable and contiguous, it's returned as-is
/// - If the input is mutable but not contiguous, it's converted with data cloning
pub fn overwritable_convert<T, B, D>(a: TensorReference<'_, T, B, D>) -> Result<TensorMutable<'_, T, B, D>>
```

**Correct usage pattern** (from SYEVD):

```rust
let mut a = overwritable_convert(a)?;
let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };
let lda = a.view().ld(order).unwrap();
// Now call LAPACK with the correct order and lda
```

**CRITICAL BUG FIX (2026-03-31)**: Previously, some implementations incorrectly used `overwritable_convert_with_order(a, order)` with a pre-determined order. This caused issues for non-square matrices. The correct pattern is:

1. Use `overwritable_convert(a)?` WITHOUT specifying order
2. Determine order AFTER conversion based on the result's `f_prefer()`/`c_prefer()`
3. Get `lda` using `a.view().ld(order).unwrap()`

This allows the tensor to naturally convert to its preferred contiguous layout, avoiding unnecessary transposes and incorrect LDA calculations.

### When to use `overwritable_convert_with_order`

Use this function only when you specifically need a particular order (rare). Most LAPACK drivers should use `overwritable_convert` without specifying order.

## Performance Tips for Matrix Operations

### Manual Stride Indexing

For inner loops that extract/copy matrix data, use raw stride indexing instead of `[[i,j]]`:

```rust
// Good: Manual stride indexing (faster for strided access)
let r_stride = *r.stride();
let qr_stride = *qr_view.stride();
let r_offset = r.offset();
let qr_offset = qr_view.offset();
let vec_r = r.raw_mut();
let vec_qr = qr_view.raw();
for i in 0..(k as isize) {
    for j in i..(n as isize) {
        let idx_r = (i * r_stride[0] + j * r_stride[1] + r_offset as isize) as usize;
        let idx_qr = (i * qr_stride[0] + j * qr_stride[1] + qr_offset as isize) as usize;
        vec_r[idx_r] = vec_qr[idx_qr];
    }
}

// Avoid: Element-by-element indexing (slower)
for i in 0..k {
    for j in 0..n {
        r[[i, j]] = qr_view[[i, j]];
    }
}
```

### Upper Triangular Extraction

Use `zeros_f` to pre-initialize with zeros, then only copy upper triangular elements:

```rust
// Good: Pre-initialize zeros, copy only upper triangular (j >= i)
let mut r: Tensor<T, B, Ix2> = zeros_f(([k, n].c(), &device))?;
for i in 0..(k as isize) {
    for j in i..(n as isize) {  // j starts from i, not 0
        // copy only upper triangular
    }
}

// Avoid: Empty array + conditional setting
let mut r = unsafe { empty_f(([k, n].c(), &device))? };
for i in 0..k {
    for j in 0..n {
        if i <= j { r[[i, j]] = qr_view[[i, j]]; }
        else { r[[i, j]] = T::zero(); }  // unnecessary
    }
}
```

### Bulk Operations with Slicing

For larger matrix operations, use tensor slicing and assignment:

```rust
// Good: Use slicing and assign for bulk operations
q_full.i_mut((.., ..n)).assign(&qr_view);

// Avoid: Element-by-element copy
for i in 0..m {
    for j in 0..n {
        q_full[[i, j]] = qr_view[[i, j]];
    }
}
```

### Identity Matrix Simplification

When filling identity portion of pre-initialized zeros matrix:

```rust
// Good: Only set diagonal (zeros already initialized)
for j in n..m {
    q_full[[j, j]] = T::one();
}

// Avoid: Conditional in inner loop
for i in 0..m {
    for j in n..m {
        q_full[[i, j]] = if i == j { T::one() } else { T::zero() };
    }
}
```

## Checklist for New Linalg Function

1. [ ] Study NumPy/SciPy reference for API design
2. [ ] Define result struct(s) in `traits_def.rs`
3. [ ] Add trait definition using `duplicate_item`
4. [ ] Add function definitions using `duplicate_item`
5. [ ] Implement reference function in `ref_impl_blas.rs`
6. [ ] Implement overloading in `blas_impl/<func>.rs`
7. [ ] Update `prelude.rs` exports
8. [ ] Add device module in `crates-device/*/linalg_auto_impl/`
9. [ ] Run `cargo fmt` and `cargo clippy`
10. [ ] Test with device crate