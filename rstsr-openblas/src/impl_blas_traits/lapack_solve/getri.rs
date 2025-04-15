use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::{blasint, lapack_solve::getri::*};
use rstsr_core::prelude::*;

impl GETRIDriverAPI<f32> for DeviceBLAS {
    unsafe fn driver_getri(
        order: FlagOrder,
        n: usize,
        a: *mut f32,
        lda: usize,
        ipiv: *mut blasint,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_sgetri(order as _, n as _, a, lda as _, ipiv)
    }
}

impl GETRIDriverAPI<f64> for DeviceBLAS {
    unsafe fn driver_getri(
        order: FlagOrder,
        n: usize,
        a: *mut f64,
        lda: usize,
        ipiv: *mut blasint,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_dgetri(order as _, n as _, a, lda as _, ipiv)
    }
}

impl GETRIDriverAPI<Complex<f32>> for DeviceBLAS {
    unsafe fn driver_getri(
        order: FlagOrder,
        n: usize,
        a: *mut Complex<f32>,
        lda: usize,
        ipiv: *mut blasint,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_cgetri(order as _, n as _, a as *mut _, lda as _, ipiv)
    }
}

impl GETRIDriverAPI<Complex<f64>> for DeviceBLAS {
    unsafe fn driver_getri(
        order: FlagOrder,
        n: usize,
        a: *mut Complex<f64>,
        lda: usize,
        ipiv: *mut blasint,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_zgetri(order as _, n as _, a as *mut _, lda as _, ipiv)
    }
}
