use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::blas3::syhemm::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     cblas_func  ;
   [f32] [cblas_ssymm];
   [f64] [cblas_dsymm];
)]
impl<const HERMI: bool> SYHEMMDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *const T,
        ldb: usize,
        beta: T,
        c: *mut T,
        ldc: usize,
    ) {
        rstsr_lapack_ffi::cblas::cblas_func(
            order.into(),
            side.into(),
            uplo.into(),
            m as _,
            n as _,
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

#[duplicate_item(
    T              cblas_func    HERMI ;
   [Complex<f32>] [cblas_csymm] [false];
   [Complex<f32>] [cblas_chemm] [true ];
   [Complex<f64>] [cblas_zsymm] [false];
   [Complex<f64>] [cblas_zhemm] [true ];
)]
impl SYHEMMDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *const T,
        ldb: usize,
        beta: T,
        c: *mut T,
        ldc: usize,
    ) {
        rstsr_lapack_ffi::cblas::cblas_func(
            order.into(),
            side.into(),
            uplo.into(),
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
