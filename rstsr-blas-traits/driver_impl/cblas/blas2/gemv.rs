use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::blas2::gemv::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     cblas_func  ;
   [f32] [cblas_sgemv];
   [f64] [cblas_dgemv];
)]
impl GEMVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gemv(
        order: FlagOrder,
        trans: FlagTrans,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        x: *const T,
        beta: T,
        y: *mut T,
    ) {
        lapack_ffi::cblas::cblas_func(order.into(), trans.into(), m as _, n as _, alpha, a, lda as _, x, 1, beta, y, 1);
    }
}

#[duplicate_item(
    T              cblas_func  ;
   [Complex<f32>] [cblas_cgemv];
   [Complex<f64>] [cblas_zgemv];
)]
impl GEMVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gemv(
        order: FlagOrder,
        trans: FlagTrans,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        x: *const T,
        beta: T,
        y: *mut T,
    ) {
        lapack_ffi::cblas::cblas_func(
            order.into(),
            trans.into(),
            m as _,
            n as _,
            &alpha as *const _ as *const _,
            a as *const _,
            lda as _,
            x as *const _,
            1,
            &beta as *const _ as *const _,
            y as *mut _,
            1,
        );
    }
}
