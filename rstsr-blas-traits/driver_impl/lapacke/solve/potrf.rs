use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func   ;
   [f32] [LAPACKE_spotrf];
   [f64] [LAPACKE_dpotrf];
)]
impl POTRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_potrf(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(order as _, uplo.into(), n as _, a, lda as _)
    }
}

#[duplicate_item(
    T              lapacke_func   ;
   [Complex<f32>] [LAPACKE_cpotrf];
   [Complex<f64>] [LAPACKE_zpotrf];
)]
impl POTRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_potrf(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(
            order as _,
            uplo.into(),
            n as _,
            a as *mut _,
            lda as _,
        )
    }
}
