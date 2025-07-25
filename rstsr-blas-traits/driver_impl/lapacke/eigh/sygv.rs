use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_ssygv];
   [f64] [LAPACKE_dsygv];
)]
impl SYGVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygv(
        order: FlagOrder,
        itype: i32,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        w: *mut T,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            itype as _,
            jobz as _,
            uplo.into(),
            n as _,
            a,
            lda as _,
            b,
            ldb as _,
            w,
        )
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex<f32>] [LAPACKE_chegv];
   [Complex<f64>] [LAPACKE_zhegv];
)]
impl SYGVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygv(
        order: FlagOrder,
        itype: i32,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        w: *mut <T as ComplexFloat>::Real,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            itype as _,
            jobz as _,
            uplo.into(),
            n as _,
            a as _,
            lda as _,
            b as _,
            ldb as _,
            w,
        )
    }
}
