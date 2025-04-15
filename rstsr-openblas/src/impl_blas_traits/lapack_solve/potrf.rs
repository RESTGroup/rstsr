use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude::*;

impl POTRFDriverAPI<f32> for DeviceBLAS {
    unsafe fn driver_potrf(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        a: *mut f32,
        lda: usize,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_spotrf(order as _, uplo.into(), n as _, a, lda as _)
    }
}

impl POTRFDriverAPI<f64> for DeviceBLAS {
    unsafe fn driver_potrf(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        a: *mut f64,
        lda: usize,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_dpotrf(order as _, uplo.into(), n as _, a, lda as _)
    }
}

impl POTRFDriverAPI<Complex<f32>> for DeviceBLAS {
    unsafe fn driver_potrf(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        a: *mut Complex<f32>,
        lda: usize,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_cpotrf(
            order as _,
            uplo.into(),
            n as _,
            a as *mut _,
            lda as _,
        )
    }
}

impl POTRFDriverAPI<Complex<f64>> for DeviceBLAS {
    unsafe fn driver_potrf(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        a: *mut Complex<f64>,
        lda: usize,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_zpotrf(
            order as _,
            uplo.into(),
            n as _,
            a as *mut _,
            lda as _,
        )
    }
}
