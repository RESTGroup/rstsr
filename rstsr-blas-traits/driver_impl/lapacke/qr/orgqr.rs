use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

// ORGQR for real types
#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_sorgqr];
   [f64] [LAPACKE_dorgqr];
)]
impl ORGQRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_orgqr(
        order: FlagOrder,
        m: usize,
        n: usize,
        k: usize,
        a: *mut T,
        lda: usize,
        tau: *const T,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, m as _, n as _, k as _, a, lda as _, tau)
    }
}

// UNGQR for complex types
#[duplicate_item(
    T              lapacke_func  ;
   [Complex::<f32>] [LAPACKE_cungqr];
   [Complex::<f64>] [LAPACKE_zungqr];
)]
impl ORGQRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_orgqr(
        order: FlagOrder,
        m: usize,
        n: usize,
        k: usize,
        a: *mut T,
        lda: usize,
        tau: *const T,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, m as _, n as _, k as _, a as *mut _, lda as _, tau as *mut _)
    }
}
