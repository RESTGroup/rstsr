use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func   ;
   [f32] [LAPACKE_ssyevd];
   [f64] [LAPACKE_dsyevd];
)]
impl SYEVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut T,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(
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

#[duplicate_item(
    T              lapacke_func   ;
   [Complex<f32>] [LAPACKE_cheevd];
   [Complex<f64>] [LAPACKE_zheevd];
)]
impl SYEVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut <T as ComplexFloat>::Real,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(
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
