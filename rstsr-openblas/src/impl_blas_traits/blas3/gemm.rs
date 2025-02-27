use std::ffi::c_void;

use crate::DeviceOpenBLAS;
use num::Complex;
use rstsr_blas_traits::blas3::gemm::*;
use rstsr_blas_traits::blasint;
use rstsr_blas_traits::cblas_flags::*;

impl GEMMFuncAPI<f32> for DeviceOpenBLAS {
    unsafe fn cblas_gemm(
        order: CBLAS_ORDER,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: blasint,
        n: blasint,
        k: blasint,
        alpha: f32,
        a: *const f32,
        lda: blasint,
        b: *const f32,
        ldb: blasint,
        beta: f32,
        c: *mut f32,
        ldc: blasint,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_sgemm(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

impl GEMMFuncAPI<f64> for DeviceOpenBLAS {
    unsafe fn cblas_gemm(
        order: CBLAS_ORDER,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: blasint,
        n: blasint,
        k: blasint,
        alpha: f64,
        a: *const f64,
        lda: blasint,
        b: *const f64,
        ldb: blasint,
        beta: f64,
        c: *mut f64,
        ldc: blasint,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_dgemm(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

impl GEMMFuncAPI<Complex<f32>> for DeviceOpenBLAS {
    unsafe fn cblas_gemm(
        order: CBLAS_ORDER,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: blasint,
        n: blasint,
        k: blasint,
        alpha: *const c_void,
        a: *const c_void,
        lda: blasint,
        b: *const c_void,
        ldb: blasint,
        beta: *const c_void,
        c: *mut c_void,
        ldc: blasint,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_cgemm(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

impl GEMMFuncAPI<Complex<f64>> for DeviceOpenBLAS {
    unsafe fn cblas_gemm(
        order: CBLAS_ORDER,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: blasint,
        n: blasint,
        k: blasint,
        alpha: *const c_void,
        a: *const c_void,
        lda: blasint,
        b: *const c_void,
        ldb: blasint,
        beta: *const c_void,
        c: *mut c_void,
        ldc: blasint,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_zgemm(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstsr_core::prelude_dev::*;

    #[test]
    fn playground() {
        let device = DeviceOpenBLAS::default();
        let a =
            linspace((0.0f32, 1., 1024 * 1024, &device)).into_shape([512, 2048]).into_dim::<Ix2>();
        let b =
            linspace((0.0f32, 1., 1024 * 1024, &device)).into_shape([2048, 512]).into_dim::<Ix2>();
        let driver = GEMMBuilder::default().a(a.view()).b(b.view()).build().unwrap();
        let c = driver.run().unwrap().into_owned();
        println!("{:?}", c);
    }
}
