use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::lapack_eigh::SYGVXDriverAPI;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_ssygvx];
   [f64] [LAPACKE_dsygvx];
)]
impl SYGVXDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygvx(
        order: FlagOrder,
        itype: blas_int,
        jobz: char,
        range: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        vl: T,
        vu: T,
        il: blas_int,
        iu: blas_int,
        abstol: T,
        m: *mut blas_int,
        w: *mut T,
        z: *mut T,
        ldz: usize,
        ifail: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            itype,
            jobz as _,
            range as _,
            uplo.into(),
            n as _,
            a,
            lda as _,
            b,
            ldb as _,
            vl,
            vu,
            il,
            iu,
            abstol,
            m,
            w,
            z,
            ldz as _,
            ifail,
        )
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex<f32>] [LAPACKE_chegvx];
   [Complex<f64>] [LAPACKE_zhegvx];
)]
impl SYGVXDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygvx(
        order: FlagOrder,
        itype: blas_int,
        jobz: char,
        range: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        vl: <T as ComplexFloat>::Real,
        vu: <T as ComplexFloat>::Real,
        il: blas_int,
        iu: blas_int,
        abstol: <T as ComplexFloat>::Real,
        m: *mut blas_int,
        w: *mut <T as ComplexFloat>::Real,
        z: *mut T,
        ldz: usize,
        ifail: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            itype,
            jobz as _,
            range as _,
            uplo.into(),
            n as _,
            a as *mut _,
            lda as _,
            b as *mut _,
            ldb as _,
            vl,
            vu,
            il,
            iu,
            abstol,
            m,
            w,
            z as *mut _,
            ldz as _,
            ifail,
        )
    }
}
