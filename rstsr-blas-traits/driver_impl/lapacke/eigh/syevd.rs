use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func   ;
   [f32] [LAPACKE_ssyevd];
   [f64] [LAPACKE_dsyevd];
)]
impl SYEVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut T,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, jobz as _, uplo.into(), n as _, a, lda as _, w)
    }
}

#[duplicate_item(
    T              lapacke_func   ;
   [Complex<f32>] [LAPACKE_cheevd];
   [Complex<f64>] [LAPACKE_zheevd];
)]
impl SYEVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut <T as ComplexFloat>::Real,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(order as _, jobz as _, uplo.into(), n as _, a as *mut _, lda as _, w)
    }
}
