use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::{blas_int, lapack_solve::getri::*};
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func   ;
   [f32] [LAPACKE_sgetri];
   [f64] [LAPACKE_dgetri];
)]
impl GETRIDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getri(
        order: FlagOrder,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(order as _, n as _, a, lda as _, ipiv)
    }
}

#[duplicate_item(
    T              lapacke_func   ;
   [Complex<f32>] [LAPACKE_cgetri];
   [Complex<f64>] [LAPACKE_zgetri];
)]
impl GETRIDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getri(
        order: FlagOrder,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(order as _, n as _, a as *mut _, lda as _, ipiv)
    }
}
