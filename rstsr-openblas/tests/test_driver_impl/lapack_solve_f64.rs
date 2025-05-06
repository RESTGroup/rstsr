use rstsr_blas_traits::lapack_solve::*;
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
use rstsr_test_manifest::get_vec;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dgesv() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = &get_vec::<f64>('b')[..1024 * 512];
        let b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DGESV::default().a(a.view()).b(b.view()).build().unwrap();
        let (lu, piv, x) = driver.run().unwrap();
        let lu = lu.into_owned();
        let x = x.into_owned();
        let fpiv = piv.map(|&v| v as f64);
        assert!((fingerprint(&lu) - 5397.198541468395).abs() < 1e-8);
        assert!((fingerprint(&fpiv) - -14.694714160751573).abs() < 1e-8);
        assert!((fingerprint(&x) - -1951.253447757597).abs() < 1e-8);
    }
}
