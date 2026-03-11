use rstsr_blas_traits::lapack_solve::*;
use rstsr_blis::DeviceBLIS as DeviceBLAS;
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
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

    #[test]
    fn test_dgetrf_dgetri() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DGETRF::default().a(a.view()).build().unwrap();
        let (lu, piv) = driver.run().unwrap();
        let lu = lu.into_owned();
        let fpiv = piv.map(|&v| v as f64);
        assert!((fingerprint(&lu) - 5397.198541468395).abs() < 1e-8);
        assert!((fingerprint(&fpiv) - -14.694714160751573).abs() < 1e-8);

        let driver = DGETRI::default().a(lu.view()).ipiv(piv.view()).build().unwrap();
        let inv_a = driver.run().unwrap();
        let inv_a = inv_a.into_owned();
        assert!((fingerprint(&inv_a) - 143.3900557703788).abs() < 1e-8);
    }

    #[test]
    fn test_dpotrf() {
        let device = DeviceBLAS::default();
        let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DPOTRF::default().a(b.view()).build().unwrap();
        let c = driver.run().unwrap();
        let c = c.into_owned();
        println!("fingerprint {:?}", fingerprint(&c));
        assert!((fingerprint(&c) - 35.17266259472725).abs() < 1e-8);

        // upper
        let driver = DPOTRF::default().a(b.view()).uplo(Upper).build().unwrap();
        let c = driver.run().unwrap();
        let c = c.into_owned();
        println!("fingerprint {:?}", fingerprint(&c));
        assert!((fingerprint(&c) - -53.53353704132017).abs() < 1e-8);
    }

    #[test]
    fn test_dsysv() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = &get_vec::<f64>('b')[..1024 * 512];
        let b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DSYSV::default().a(a.view()).b(b.view()).build().unwrap();
        let (udut, piv, x) = driver.run().unwrap();
        let udut = udut.into_owned();
        let x = x.into_owned();
        let fpiv = piv.map(|&v| v as f64);
        assert!((fingerprint(&udut) - -1201.6472395568974).abs() < 1e-8);
        assert!((fingerprint(&fpiv) - -16668.7094872639).abs() < 1e-8);
        assert!((fingerprint(&x) - -397.12032355166446).abs() < 1e-8);

        // upper
        let driver = DSYSV::default().a(a.view()).b(b.view()).uplo(Upper).build().unwrap();
        let (udut, piv, x) = driver.run().unwrap();
        let udut = udut.into_owned();
        let x = x.into_owned();
        let fpiv = piv.map(|&v| v as f64);
        assert!((fingerprint(&udut) - 1182.7836118324408).abs() < 1e-8);
        assert!((fingerprint(&fpiv) - 11905.503011559245).abs() < 1e-8);
        assert!((fingerprint(&x) - -314.4502289190444).abs() < 1e-8);
    }
}
