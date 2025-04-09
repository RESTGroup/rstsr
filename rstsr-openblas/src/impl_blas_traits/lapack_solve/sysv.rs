use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude::*;

impl<const HERMI: bool> SYSVDriverAPI<f32, HERMI> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut f32,
        lda: usize,
        ipiv: *mut blasint,
        b: *mut f32,
        ldb: usize,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_ssysv(
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

impl<const HERMI: bool> SYSVDriverAPI<f64, HERMI> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut f64,
        lda: usize,
        ipiv: *mut blasint,
        b: *mut f64,
        ldb: usize,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_dsysv(
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

impl SYSVDriverAPI<Complex<f32>, false> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut Complex<f32>,
        lda: usize,
        ipiv: *mut blasint,
        b: *mut Complex<f32>,
        ldb: usize,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_csysv(
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

impl SYSVDriverAPI<Complex<f64>, false> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut Complex<f64>,
        lda: usize,
        ipiv: *mut blasint,
        b: *mut Complex<f64>,
        ldb: usize,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_zsysv(
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

impl SYSVDriverAPI<Complex<f32>, true> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut Complex<f32>,
        lda: usize,
        ipiv: *mut blasint,
        b: *mut Complex<f32>,
        ldb: usize,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_chesv(
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

impl SYSVDriverAPI<Complex<f64>, true> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut Complex<f64>,
        lda: usize,
        ipiv: *mut blasint,
        b: *mut Complex<f64>,
        ldb: usize,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_zhesv(
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
