use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::lapack_eigh::SYEVRDriverAPI;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_ssyevr];
   [f64] [LAPACKE_dsyevr];
)]
impl SYEVRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevr(
        order: FlagOrder,
        jobz: char,
        range: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        vl: T,
        vu: T,
        il: blas_int,
        iu: blas_int,
        abstol: T,
        m: *mut blas_int,
        w: *mut T,
        z: *mut T,
        ldz: usize,
        isuppz: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobz as _,
            range as _,
            uplo.into(),
            n as _,
            a,
            lda as _,
            vl,
            vu,
            il,
            iu,
            abstol,
            m,
            w,
            z,
            ldz as _,
            isuppz,
        )
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex<f32>] [LAPACKE_cheevr];
   [Complex<f64>] [LAPACKE_zheevr];
)]
impl SYEVRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevr(
        order: FlagOrder,
        jobz: char,
        range: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        vl: <T as ComplexFloat>::Real,
        vu: <T as ComplexFloat>::Real,
        il: blas_int,
        iu: blas_int,
        abstol: <T as ComplexFloat>::Real,
        m: *mut blas_int,
        w: *mut <T as ComplexFloat>::Real,
        z: *mut T,
        ldz: usize,
        isuppz: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobz as _,
            range as _,
            uplo.into(),
            n as _,
            a as *mut _,
            lda as _,
            vl,
            vu,
            il,
            iu,
            abstol,
            m,
            w,
            z as *mut _,
            ldz as _,
            isuppz,
        )
    }
}
