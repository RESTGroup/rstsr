use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func   ;
   [f32] [LAPACKE_sgelsd];
   [f64] [LAPACKE_dgelsd];
)]
impl GELSDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gelsd(
        order: FlagOrder,
        m: usize,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        s: *mut <T as ComplexFloat>::Real,
        rcond: <T as ComplexFloat>::Real,
        rank: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _, m as _, n as _, nrhs as _, a, lda as _, b, ldb as _, s, rcond, rank,
        )
    }
}

#[duplicate_item(
    T              lapacke_func   ;
   [Complex<f32>] [LAPACKE_cgelsd];
   [Complex<f64>] [LAPACKE_zgelsd];
)]
impl GELSDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gelsd(
        order: FlagOrder,
        m: usize,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        s: *mut <T as ComplexFloat>::Real,
        rcond: <T as ComplexFloat>::Real,
        rank: *mut blas_int,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            m as _,
            n as _,
            nrhs as _,
            a as *mut _,
            lda as _,
            b as *mut _,
            ldb as _,
            s,
            rcond,
            rank,
        )
    }
}
