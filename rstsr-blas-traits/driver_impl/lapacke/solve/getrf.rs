use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func   ;
   [f32] [LAPACKE_sgetrf];
   [f64] [LAPACKE_dgetrf];
)]
impl GETRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(order as _, m as _, n as _, a, lda as _, ipiv)
    }
}

#[duplicate_item(
    T              lapacke_func   ;
   [Complex<f32>] [LAPACKE_cgetrf];
   [Complex<f64>] [LAPACKE_zgetrf];
)]
impl GETRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        rstsr_lapack_ffi::lapacke::lapacke_func(
            order as _,
            m as _,
            n as _,
            a as *mut _,
            lda as _,
            ipiv,
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
        let driver = DGETRF::default().a(a.view()).build().unwrap();
        let (c, ipiv) = driver.run().unwrap();
        let c = c.into_owned();
        // println!("{:?}", c);
        println!("{ipiv:?}");
        assert!(c.c_contig());
        assert!((fingerprint(&c) - -8345.684144788995).abs() < 1e-8);
    }
}
