use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

// ORMQR for real types
#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_sormqr];
   [f64] [LAPACKE_dormqr];
)]
impl ORMQRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_ormqr(
        order: FlagOrder,
        side: FlagSide,
        trans: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        a: *const T,
        lda: usize,
        tau: *const T,
        c: *mut T,
        ldc: usize,
    ) -> blas_int {
        use std::os::raw::c_char;
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            side.into() as c_char,
            trans.into() as c_char,
            m as _,
            n as _,
            k as _,
            a,
            lda as _,
            tau,
            c,
            ldc as _,
        )
    }
}

// UNMQR for complex types
#[duplicate_item(
    T              lapacke_func  ;
   [Complex::<f32>] [LAPACKE_cunmqr];
   [Complex::<f64>] [LAPACKE_zunmqr];
)]
impl ORMQRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_ormqr(
        order: FlagOrder,
        side: FlagSide,
        trans: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        a: *const T,
        lda: usize,
        tau: *const T,
        c: *mut T,
        ldc: usize,
    ) -> blas_int {
        use std::os::raw::c_char;
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            side.into() as c_char,
            trans.into() as c_char,
            m as _,
            n as _,
            k as _,
            a as *const _,
            lda as _,
            tau as *const _,
            c as *mut _,
            ldc as _,
        )
    }
}
