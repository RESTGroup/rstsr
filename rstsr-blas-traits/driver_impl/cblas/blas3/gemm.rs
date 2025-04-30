use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::blas3::gemm::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     cblas_func  ;
   [f32] [cblas_sgemm];
   [f64] [cblas_dgemm];
)]
impl GEMMDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gemm(
        order: FlagOrder,
        transa: FlagTrans,
        transb: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
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
            transa.into(),
            transb.into(),
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

#[duplicate_item(
    T              cblas_func  ;
   [Complex<f32>] [cblas_cgemm];
   [Complex<f64>] [cblas_zgemm];
)]
impl GEMMDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gemm(
        order: FlagOrder,
        transa: FlagTrans,
        transb: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
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
            transa.into(),
            transb.into(),
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
    use crate::DeviceBLAS;
    use rstsr_core::prelude_dev::*;
    use rstsr_test_manifest::get_vec;

    #[test]
    fn playground() {
        let device = DeviceBLAS::default();
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
        assert!(c.c_contig());
        assert!((fingerprint(&c) - -4118.154714656608).abs() < 1e-8);
    }
}
