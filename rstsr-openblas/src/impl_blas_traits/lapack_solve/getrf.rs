use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude::*;

impl GETRFDriverAPI<f32> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut f32,
        lda: usize,
        ipiv: *mut blasint,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_sgetrf(order as _, m as _, n as _, a, lda as _, ipiv)
    }
}

impl GETRFDriverAPI<f64> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut f64,
        lda: usize,
        ipiv: *mut blasint,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_dgetrf(order as _, m as _, n as _, a, lda as _, ipiv)
    }
}

impl GETRFDriverAPI<Complex<f32>> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut Complex<f32>,
        lda: usize,
        ipiv: *mut blasint,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_cgetrf(
            order as _,
            m as _,
            n as _,
            a as *mut _,
            lda as _,
            ipiv,
        )
    }
}

impl GETRFDriverAPI<Complex<f64>> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut Complex<f64>,
        lda: usize,
        ipiv: *mut blasint,
    ) -> blasint {
        rstsr_lapack_ffi::lapacke::LAPACKE_zgetrf(
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
        println!("{:?}", ipiv);
        assert!(c.c_contig());
        assert!((fingerprint(&c) - -8345.684144788995).abs() < 1e-8);
    }
}
