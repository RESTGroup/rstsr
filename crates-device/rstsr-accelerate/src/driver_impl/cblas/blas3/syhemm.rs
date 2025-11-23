use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::blas3::syhemm::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     cblas_func  ;
   [f32] [cblas_ssymm];
   [f64] [cblas_dsymm];
)]
impl<const HERMI: bool> SYHEMMDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
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
            side.into(),
            uplo.into(),
            m as _,
            n as _,
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
    T              cblas_func    HERMI ;
   [Complex<f32>] [cblas_csymm] [false];
   [Complex<f32>] [cblas_chemm] [true ];
   [Complex<f64>] [cblas_zsymm] [false];
   [Complex<f64>] [cblas_zhemm] [true ];
)]
impl SYHEMMDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
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
            side.into(),
            uplo.into(),
            m as _,
            n as _,
            &alpha as *const _ as *const _,
            a as *const _ as *const _,
            lda as _,
            b as *const _ as *const _,
            ldb as _,
            &beta as *const _ as *const _,
            c as *mut _ as *mut _,
            ldc as _,
        );
    }
}
