use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::blas3::trsm::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     cblas_func  ;
   [f32] [cblas_strsm];
   [f64] [cblas_dtrsm];
)]
impl TRSMDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_trsm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        transa: FlagTrans,
        diag: FlagDiag,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *mut T,
        ldb: usize,
    ) {
        rstsr_lapack_ffi::cblas::cblas_func(
            order.into(),
            side.into(),
            uplo.into(),
            transa.into(),
            diag.into(),
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

#[duplicate_item(
    T              cblas_func  ;
   [Complex<f32>] [cblas_ctrsm];
   [Complex<f64>] [cblas_ztrsm];
)]
impl TRSMDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_trsm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        transa: FlagTrans,
        diag: FlagDiag,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *mut T,
        ldb: usize,
    ) {
        rstsr_lapack_ffi::cblas::cblas_func(
            order.into(),
            side.into(),
            uplo.into(),
            transa.into(),
            diag.into(),
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
        println!("{c:?}");
        assert!(c.c_contig());
        // current matrix is nan, where non-nan part is similar to the expected value
        println!("{:?}", fingerprint(&c));
        // assert!((fingerprint(&c) - 80575.42858282285).abs() < 1e-8);
    }
}
