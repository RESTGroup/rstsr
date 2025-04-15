use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude::*;

impl SYGVDDriverAPI<f32> for DeviceBLAS {
    unsafe fn driver_sygvd(
        order: FlagOrder,
        itype: i32,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut f32,
        lda: usize,
        b: *mut f32,
        ldb: usize,
        w: *mut f32,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_ssygvd(
            order as _,
            itype as _,
            jobz as _,
            uplo.into(),
            n as _,
            a,
            lda as _,
            b,
            ldb as _,
            w,
        )
    }
}

impl SYGVDDriverAPI<f64> for DeviceBLAS {
    unsafe fn driver_sygvd(
        order: FlagOrder,
        itype: i32,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut f64,
        lda: usize,
        b: *mut f64,
        ldb: usize,
        w: *mut f64,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_dsygvd(
            order as _,
            itype as _,
            jobz as _,
            uplo.into(),
            n as _,
            a,
            lda as _,
            b,
            ldb as _,
            w,
        )
    }
}

impl SYGVDDriverAPI<Complex<f32>> for DeviceBLAS {
    unsafe fn driver_sygvd(
        order: FlagOrder,
        itype: i32,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut Complex<f32>,
        lda: usize,
        b: *mut Complex<f32>,
        ldb: usize,
        w: *mut f32,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_chegvd(
            order as _,
            itype as _,
            jobz as _,
            uplo.into(),
            n as _,
            a as _,
            lda as _,
            b as _,
            ldb as _,
            w,
        )
    }
}

impl SYGVDDriverAPI<Complex<f64>> for DeviceBLAS {
    unsafe fn driver_sygvd(
        order: FlagOrder,
        itype: i32,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut Complex<f64>,
        lda: usize,
        b: *mut Complex<f64>,
        ldb: usize,
        w: *mut f64,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::LAPACKE_zhegvd(
            order as _,
            itype as _,
            jobz as _,
            uplo.into(),
            n as _,
            a as _,
            lda as _,
            b as _,
            ldb as _,
            w,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DeviceBLAS;
    use rstsr_core::prelude_dev::*;

    #[test]
    fn playground() {
        let device = DeviceBLAS::default();
        let vec_a = [0, 1, 9, 1, 5, 3, 9, 3, 6].iter().map(|&x| x as f64).collect::<Vec<_>>();
        let a = asarray((vec_a, [3, 3].c(), &device)).into_dim::<Ix2>();
        let vec_b = [1, 1, 2, 1, 3, 1, 2, 1, 8].iter().map(|&x| x as f64).collect::<Vec<_>>();
        let b = asarray((vec_b, [3, 3].c(), &device)).into_dim::<Ix2>();
        let result = DSYGVD::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
        println!("{:?}", result.0.into_owned());
        println!("{:?}", result.1);
    }
}
