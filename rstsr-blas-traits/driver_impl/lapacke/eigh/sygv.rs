use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_ssygv];
   [f64] [LAPACKE_dsygv];
)]
impl SYGVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygv(
        order: FlagOrder,
        itype: i32,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        w: *mut T,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(
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

#[duplicate_item(
    T              lapacke_func  ;
   [Complex<f32>] [LAPACKE_chegv];
   [Complex<f64>] [LAPACKE_zhegv];
)]
impl SYGVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygv(
        order: FlagOrder,
        itype: i32,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        w: *mut <T as ComplexFloat>::Real,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(
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
        let result = DSYGV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
        println!("{:?}", result.0.into_owned());
        println!("{:?}", result.1);
    }
}
