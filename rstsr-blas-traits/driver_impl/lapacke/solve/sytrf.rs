use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func   ;
   [f32] [LAPACKE_ssytrf];
   [f64] [LAPACKE_dsytrf];
)]
impl<const HERMI: bool> SYTRFDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_sytrf(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, uplo.into(), n as _, a, lda as _, ipiv)
    }
}

#[duplicate_item(
    T              lapacke_func    HERMI ;
   [Complex<f32>] [LAPACKE_csytrf] [false];
   [Complex<f32>] [LAPACKE_chetrf] [true ];
   [Complex<f64>] [LAPACKE_zsytrf] [false];
   [Complex<f64>] [LAPACKE_zhetrf] [true ];
)]
impl SYTRFDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_sytrf(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, uplo.into(), n as _, a as *mut _, lda as _, ipiv)
    }
}
