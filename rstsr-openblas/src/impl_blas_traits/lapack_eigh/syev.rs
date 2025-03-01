use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::{syev::SYEVDriverAPI, *};
use rstsr_core::flags::*;

impl SYEVDriverAPI<f32> for DeviceBLAS {
    unsafe fn driver_syev(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut f32,
        lda: usize,
        w: *mut f32,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_ssyev(
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

impl SYEVDriverAPI<f64> for DeviceBLAS {
    unsafe fn driver_syev(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut f64,
        lda: usize,
        w: *mut f64,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_dsyev(
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

impl SYEVDriverAPI<Complex<f32>> for DeviceBLAS {
    unsafe fn driver_syev(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut Complex<f32>,
        lda: usize,
        w: *mut f32,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_cheev(
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

impl SYEVDriverAPI<Complex<f64>> for DeviceBLAS {
    unsafe fn driver_syev(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut Complex<f64>,
        lda: usize,
        w: *mut f64,
    ) -> blasint {
        rstsr_openblas_ffi::ffi::lapacke::LAPACKE_zheev(
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
        let driver = DSYEV::default().a(a.view()).build().unwrap();
        let (c, w) = driver.run().unwrap();
        let c = c.into_owned();
        // println!("{:?}", c);
        println!("{:?}", w);
        assert!(c.c_contig());
        println!("{:?}", fingerprint(&c));
        // assert!((fingerprint(&c) - -8345.684144788995).abs() < 1e-8);
    }
}
