use crate::DeviceOpenBLAS as DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::blas3::gemm::*;
use rstsr_core::flags::{FlagOrder, FlagTrans};

impl GEMMDriverAPI<f32> for DeviceBLAS {
    unsafe fn driver_gemm(
        order: FlagOrder,
        transa: FlagTrans,
        transb: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_sgemm(
            order as _,
            transa as _,
            transb as _,
            m as _,
            n as _,
            k as _,
            alpha,
            a,
            lda as _,
            b,
            ldb as _,
            beta,
            c,
            ldc as _,
        );
    }
}

impl GEMMDriverAPI<f64> for DeviceBLAS {
    unsafe fn driver_gemm(
        order: FlagOrder,
        transa: FlagTrans,
        transb: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_dgemm(
            order as _,
            transa as _,
            transb as _,
            m as _,
            n as _,
            k as _,
            alpha,
            a,
            lda as _,
            b,
            ldb as _,
            beta,
            c,
            ldc as _,
        );
    }
}

impl GEMMDriverAPI<Complex<f32>> for DeviceBLAS {
    unsafe fn driver_gemm(
        order: FlagOrder,
        transa: FlagTrans,
        transb: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: usize,
        b: *const Complex<f32>,
        ldb: usize,
        beta: Complex<f32>,
        c: *mut Complex<f32>,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_cgemm(
            order as _,
            transa as _,
            transb as _,
            m as _,
            n as _,
            k as _,
            &alpha as *const _ as *const _,
            a as *const _,
            lda as _,
            b as *const _,
            ldb as _,
            &beta as *const _ as *const _,
            c as *mut _,
            ldc as _,
        );
    }
}

impl GEMMDriverAPI<Complex<f64>> for DeviceBLAS {
    unsafe fn driver_gemm(
        order: FlagOrder,
        transa: FlagTrans,
        transb: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: usize,
        b: *const Complex<f64>,
        ldb: usize,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_zgemm(
            order as _,
            transa as _,
            transb as _,
            m as _,
            n as _,
            k as _,
            &alpha as *const _ as *const _,
            a as *const _,
            lda as _,
            b as *const _,
            ldb as _,
            &beta as *const _ as *const _,
            c as *mut _,
            ldc as _,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DeviceOpenBLAS;
    use rstsr_core::prelude_dev::*;
    use rstsr_test_manifest::get_vec;

    #[test]
    fn playground() {
        let device = DeviceOpenBLAS::default();
        let la = [1024, 4096].c();
        let lb = [2048, 4096].c();
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);
        let b = Tensor::new(Storage::new(get_vec::<f64>('b').into(), device.clone()), lb);
        let driver = GEMM::default().a(a.view()).b(b.t()).build().unwrap();
        let c = driver.run().unwrap().into_owned();
        assert!(c.c_contig());
        assert!((fingerprint(&c) - -4118.154714656608).abs() < 1e-8);
        let driver = GEMM::default().a(a.view()).b(b.view()).transb('T').build().unwrap();
        let c = driver.run().unwrap().into_owned();
        assert!(c.c_contig());
        assert!((fingerprint(&c) - -4118.154714656608).abs() < 1e-8);
        let driver = GEMM::default().a(a.t()).b(b.t()).transa('T').build().unwrap();
        let c = driver.run().unwrap().into_owned();
        assert!(c.f_contig());
        assert!((fingerprint(&c) - -4118.154714656608).abs() < 1e-8);
    }
}
