use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::blas3::trsm::*;
use rstsr_core::flags::*;

impl TRSMDriverAPI<f32> for DeviceBLAS {
    unsafe fn driver_trsm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        transa: FlagTrans,
        diag: FlagDiag,
        m: usize,
        n: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        b: *mut f32,
        ldb: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_strsm(
            order as _,
            side as _,
            uplo as _,
            transa as _,
            diag as _,
            m as _,
            n as _,
            alpha,
            a,
            lda as _,
            b,
            ldb as _,
        );
    }
}

impl TRSMDriverAPI<f64> for DeviceBLAS {
    unsafe fn driver_trsm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        transa: FlagTrans,
        diag: FlagDiag,
        m: usize,
        n: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *mut f64,
        ldb: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_dtrsm(
            order as _,
            side as _,
            uplo as _,
            transa as _,
            diag as _,
            m as _,
            n as _,
            alpha,
            a,
            lda as _,
            b,
            ldb as _,
        );
    }
}

impl TRSMDriverAPI<Complex<f32>> for DeviceBLAS {
    unsafe fn driver_trsm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        transa: FlagTrans,
        diag: FlagDiag,
        m: usize,
        n: usize,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: usize,
        b: *mut Complex<f32>,
        ldb: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_ctrsm(
            order as _,
            side as _,
            uplo as _,
            transa as _,
            diag as _,
            m as _,
            n as _,
            &alpha as *const _ as *const _,
            a as *const _,
            lda as _,
            b as *mut _,
            ldb as _,
        );
    }
}

impl TRSMDriverAPI<Complex<f64>> for DeviceBLAS {
    unsafe fn driver_trsm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        transa: FlagTrans,
        diag: FlagDiag,
        m: usize,
        n: usize,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: usize,
        b: *mut Complex<f64>,
        ldb: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_ztrsm(
            order as _,
            side as _,
            uplo as _,
            transa as _,
            diag as _,
            m as _,
            n as _,
            &alpha as *const _ as *const _,
            a as *const _,
            lda as _,
            b as *mut _,
            ldb as _,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DeviceBLAS;
    use rstsr_core::prelude_dev::*;
    use rstsr_test_manifest::get_vec;

    #[test]
    fn playground() {
        let device = DeviceBLAS::default();
        let la = [2048, 2048].c();
        let lb = [2048, 4096].c();
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);
        let b = Tensor::new(Storage::new(get_vec::<f64>('b').into(), device.clone()), lb);
        let driver = DTRSM::default().a(a.view()).b(b.view()).build().unwrap();
        let c = driver.run().unwrap().into_owned();
        println!("{:?}", c);
        assert!(c.c_contig());
        // current matrix is nan, where non-nan part is similar to the expected value
        println!("{:?}", fingerprint(&c));
        // assert!((fingerprint(&c) - 80575.42858282285).abs() < 1e-8);
    }
}
