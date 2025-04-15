use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude::*;

impl SYEVDDriverAPI<f32> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut f32,
        lda: usize,
        w: *mut f32,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_ssyevd(
            order as _,
            jobz as _,
            uplo.into(),
            n as _,
            a,
            lda as _,
            w,
        )
    }
}

impl SYEVDDriverAPI<f64> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut f64,
        lda: usize,
        w: *mut f64,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_dsyevd(
            order as _,
            jobz as _,
            uplo.into(),
            n as _,
            a,
            lda as _,
            w,
        )
    }
}

impl SYEVDDriverAPI<Complex<f32>> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut Complex<f32>,
        lda: usize,
        w: *mut f32,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_cheevd(
            order as _,
            jobz as _,
            uplo.into(),
            n as _,
            a as *mut _,
            lda as _,
            w,
        )
    }
}

impl SYEVDDriverAPI<Complex<f64>> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut Complex<f64>,
        lda: usize,
        w: *mut f64,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_zheevd(
            order as _,
            jobz as _,
            uplo.into(),
            n as _,
            a as *mut _,
            lda as _,
            w,
        )
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
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);
        let driver = DSYEVD::default().a(a.view()).build().unwrap();
        let (c, w) = driver.run().unwrap();
        let c = c.into_owned();
        // println!("{:?}", c);
        println!("{:?}", w);
        assert!(c.c_contig());
        println!("{:?}", fingerprint(&c));
        // assert!((fingerprint(&c) - -8345.684144788995).abs() < 1e-8);
    }
}
