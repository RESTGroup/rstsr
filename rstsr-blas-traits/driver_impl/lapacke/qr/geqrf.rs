use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_sgeqrf];
   [f64] [LAPACKE_dgeqrf];
)]
impl GEQRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geqrf(order: FlagOrder, m: usize, n: usize, a: *mut T, lda: usize, tau: *mut T) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, m as _, n as _, a, lda as _, tau)
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex::<f32>] [LAPACKE_cgeqrf];
   [Complex::<f64>] [LAPACKE_zgeqrf];
)]
impl GEQRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geqrf(order: FlagOrder, m: usize, n: usize, a: *mut T, lda: usize, tau: *mut T) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, m as _, n as _, a as *mut _, lda as _, tau as *mut _)
    }
}
