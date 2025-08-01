use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func   ;
   [f32] [LAPACKE_sgetrf];
   [f64] [LAPACKE_dgetrf];
)]
impl GETRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, m as _, n as _, a, lda as _, ipiv)
    }
}

#[duplicate_item(
    T              lapacke_func   ;
   [Complex<f32>] [LAPACKE_cgetrf];
   [Complex<f64>] [LAPACKE_zgetrf];
)]
impl GETRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, m as _, n as _, a as *mut _, lda as _, ipiv)
    }
}
