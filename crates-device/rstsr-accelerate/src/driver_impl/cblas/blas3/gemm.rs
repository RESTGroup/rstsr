use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::blas3::gemm::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     cblas_func  ;
   [f32] [cblas_sgemm];
   [f64] [cblas_dgemm];
)]
impl GEMMDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gemm(
        order: FlagOrder,
        transa: FlagTrans,
        transb: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *const T,
        ldb: usize,
        beta: T,
        c: *mut T,
        ldc: usize,
    ) {
        lapack_ffi::cblas::cblas_func(
            order.into(),
            transa.into(),
            transb.into(),
            m as _,
            n as _,
            k as _,
            alpha,
            a,
            lda as _,
            b,
            ldb as _,
            beta,
            c,
            ldc as _,
        );
    }
}

#[duplicate_item(
    T              cblas_func  ;
   [Complex<f32>] [cblas_cgemm];
   [Complex<f64>] [cblas_zgemm];
)]
impl GEMMDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gemm(
        order: FlagOrder,
        transa: FlagTrans,
        transb: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *const T,
        ldb: usize,
        beta: T,
        c: *mut T,
        ldc: usize,
    ) {
        lapack_ffi::cblas::cblas_func(
            order.into(),
            transa.into(),
            transb.into(),
            m as _,
            n as _,
            k as _,
            &alpha as *const _ as *const _,
            a as *const _,
            lda as _,
            b as *const _,
            ldb as _,
            &beta as *const _ as *const _,
            c as *mut _,
            ldc as _,
        );
    }
}
