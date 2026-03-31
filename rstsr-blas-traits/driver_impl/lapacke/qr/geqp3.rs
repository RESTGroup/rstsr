use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_sgeqp3];
   [f64] [LAPACKE_dgeqp3];
)]
impl GEQP3DriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geqp3(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        jpvt: *mut blas_int,
        tau: *mut T,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, m as _, n as _, a, lda as _, jpvt, tau)
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex::<f32>] [LAPACKE_cgeqp3];
   [Complex::<f64>] [LAPACKE_zgeqp3];
)]
impl GEQP3DriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geqp3(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        jpvt: *mut blas_int,
        tau: *mut T,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, m as _, n as _, a as *mut _, lda as _, jpvt, tau as *mut _)
    }
}
