use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::blas3::syhemm::*;
use rstsr_core::flags::*;

impl<const HERMI: bool> SYHEMMDriverAPI<f32, HERMI> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_ssymm(
            order as _, side as _, uplo as _, m as _, n as _, alpha, a, lda as _, b, ldb as _,
            beta, c, ldc as _,
        );
    }
}

impl<const HERMI: bool> SYHEMMDriverAPI<f64, HERMI> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_dsymm(
            order as _, side as _, uplo as _, m as _, n as _, alpha, a, lda as _, b, ldb as _,
            beta, c, ldc as _,
        );
    }
}

impl SYHEMMDriverAPI<Complex<f32>, false> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: usize,
        b: *const Complex<f32>,
        ldb: usize,
        beta: Complex<f32>,
        c: *mut Complex<f32>,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_csymm(
            order as _,
            side as _,
            uplo as _,
            m as _,
            n as _,
            &alpha as *const _ as *const _,
            a as *const _ as *const _,
            lda as _,
            b as *const _ as *const _,
            ldb as _,
            &beta as *const _ as *const _,
            c as *mut _ as *mut _,
            ldc as _,
        );
    }
}

impl SYHEMMDriverAPI<Complex<f32>, true> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
        alpha: Complex<f32>,
        a: *const Complex<f32>,
        lda: usize,
        b: *const Complex<f32>,
        ldb: usize,
        beta: Complex<f32>,
        c: *mut Complex<f32>,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_chemm(
            order as _,
            side as _,
            uplo as _,
            m as _,
            n as _,
            &alpha as *const _ as *const _,
            a as *const _ as *const _,
            lda as _,
            b as *const _ as *const _,
            ldb as _,
            &beta as *const _ as *const _,
            c as *mut _ as *mut _,
            ldc as _,
        );
    }
}

impl SYHEMMDriverAPI<Complex<f64>, false> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: usize,
        b: *const Complex<f64>,
        ldb: usize,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_zsymm(
            order as _,
            side as _,
            uplo as _,
            m as _,
            n as _,
            &alpha as *const _ as *const _,
            a as *const _ as *const _,
            lda as _,
            b as *const _ as *const _,
            ldb as _,
            &beta as *const _ as *const _,
            c as *mut _ as *mut _,
            ldc as _,
        );
    }
}

impl SYHEMMDriverAPI<Complex<f64>, true> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: usize,
        b: *const Complex<f64>,
        ldb: usize,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: usize,
    ) {
        rstsr_openblas_ffi::ffi::cblas::cblas_zhemm(
            order as _,
            side as _,
            uplo as _,
            m as _,
            n as _,
            &alpha as *const _ as *const _,
            a as *const _ as *const _,
            lda as _,
            b as *const _ as *const _,
            ldb as _,
            &beta as *const _ as *const _,
            c as *mut _ as *mut _,
            ldc as _,
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
        let lb = [4096, 2048].c();
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);
        let b = Tensor::new(Storage::new(get_vec::<f64>('b').into(), device.clone()), lb);
        let driver = DSYMM::default().a(a.view()).b(b.t()).build().unwrap();
        let c = driver.run().unwrap().into_owned();
        assert!(c.c_contig());
        assert!((fingerprint(&c) - 80575.42858282285).abs() < 1e-8);
    }
}
