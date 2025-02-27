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
    use rstsr_test_manifest::get_vec;

    #[test]
    fn playground() {
        let device = DeviceOpenBLAS::default();
        let la = [1024, 4096].c();
        let lb = [2048, 4096].c();
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);
        let b = Tensor::new(Storage::new(get_vec::<f64>('b').into(), device.clone()), lb);
        let driver = GEMMBuilder::default().a(a.view()).b(b.t()).build().unwrap();
        let c = driver.run().unwrap().into_owned();
        println!("{:?}", fingerprint(&c));
        assert!((fingerprint(&c) - -4118.154714656608).abs() < 1e-8);
    }
}
