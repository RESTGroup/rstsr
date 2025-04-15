use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude::*;

impl GESVDriverAPI<f32> for DeviceBLAS {
    unsafe fn driver_gesv(
        order: FlagOrder,
        n: usize,
        nrhs: usize,
        a: *mut f32,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut f32,
        ldb: usize,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_sgesv(
            order as _, n as _, nrhs as _, a, lda as _, ipiv, b, ldb as _,
        )
    }
}

impl GESVDriverAPI<f64> for DeviceBLAS {
    unsafe fn driver_gesv(
        order: FlagOrder,
        n: usize,
        nrhs: usize,
        a: *mut f64,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut f64,
        ldb: usize,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_dgesv(
            order as _, n as _, nrhs as _, a, lda as _, ipiv, b, ldb as _,
        )
    }
}

impl GESVDriverAPI<Complex<f32>> for DeviceBLAS {
    unsafe fn driver_gesv(
        order: FlagOrder,
        n: usize,
        nrhs: usize,
        a: *mut Complex<f32>,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut Complex<f32>,
        ldb: usize,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_cgesv(
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

impl GESVDriverAPI<Complex<f64>> for DeviceBLAS {
    unsafe fn driver_gesv(
        order: FlagOrder,
        n: usize,
        nrhs: usize,
        a: *mut Complex<f64>,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut Complex<f64>,
        ldb: usize,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_zgesv(
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
