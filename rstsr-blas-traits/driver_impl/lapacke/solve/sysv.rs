use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_ssysv];
   [f64] [LAPACKE_dsysv];
)]
impl<const HERMI: bool> SYSVDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut T,
        ldb: usize,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(
            order as _,
            uplo.into(),
            n as _,
            nrhs as _,
            a,
            lda as _,
            ipiv,
            b,
            ldb as _,
        )
    }
}

#[duplicate_item(
    T              lapacke_func    HERMI ;
   [Complex<f32>] [LAPACKE_csysv] [false];
   [Complex<f32>] [LAPACKE_chesv] [true ];
   [Complex<f64>] [LAPACKE_zsysv] [false];
   [Complex<f64>] [LAPACKE_zhesv] [true ];
)]
impl SYSVDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut T,
        ldb: usize,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(
            order as _,
            uplo.into(),
            n as _,
            nrhs as _,
            a as *mut _,
            lda as _,
            ipiv,
            b as *mut _,
            ldb as _,
        )
    }
}
