use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_sgesv];
   [f64] [LAPACKE_dgesv];
)]
impl GESVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesv(
        order: FlagOrder,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut T,
        ldb: usize,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(
            order as _, n as _, nrhs as _, a, lda as _, ipiv, b, ldb as _,
        )
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex<f32>] [LAPACKE_cgesv];
   [Complex<f64>] [LAPACKE_zgesv];
)]
impl GESVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesv(
        order: FlagOrder,
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
