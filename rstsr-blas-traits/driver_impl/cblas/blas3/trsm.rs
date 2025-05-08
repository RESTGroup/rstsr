use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::blas3::trsm::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     cblas_func  ;
   [f32] [cblas_strsm];
   [f64] [cblas_dtrsm];
)]
impl TRSMDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_trsm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        transa: FlagTrans,
        diag: FlagDiag,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *mut T,
        ldb: usize,
    ) {
        rstsr_lapack_ffi::cblas::cblas_func(
            order.into(),
            side.into(),
            uplo.into(),
            transa.into(),
            diag.into(),
            m as _,
            n as _,
            alpha,
            a,
            lda as _,
            b,
            ldb as _,
        );
    }
}

#[duplicate_item(
    T              cblas_func  ;
   [Complex<f32>] [cblas_ctrsm];
   [Complex<f64>] [cblas_ztrsm];
)]
impl TRSMDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_trsm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        transa: FlagTrans,
        diag: FlagDiag,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *mut T,
        ldb: usize,
    ) {
        rstsr_lapack_ffi::cblas::cblas_func(
            order.into(),
            side.into(),
            uplo.into(),
            transa.into(),
            diag.into(),
            m as _,
            n as _,
            &alpha as *const _ as *const _,
            a as *const _,
            lda as _,
            b as *mut _,
            ldb as _,
        );
    }
}
