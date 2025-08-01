use rstsr_blas_traits::lapack_svd::*;
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_kml::DeviceKML as DeviceBLAS;
use rstsr_test_manifest::get_vec;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dgesvd() {
        let device = DeviceBLAS::default();
        let a_vec = &get_vec::<f64>('a')[..1024 * 512];
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DGESVD::default().a(a.view()).build().unwrap();
        if let (s, Some(u), Some(vt), _) = driver.run().unwrap() {
            assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
            assert!((fingerprint(&u.abs()) - -1.9368850983570982).abs() < 1e-8);
            assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);
        } else {
            panic!("DGESVD did not return expected output");
        }

        // full_matrices = false
        let driver = DGESVD::default().a(a.view()).full_matrices(false).build().unwrap();
        if let (s, Some(u), Some(vt), _) = driver.run().unwrap() {
            assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
            assert!((fingerprint(&u.abs()) - -9.144981428076894).abs() < 1e-8);
            assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);
        } else {
            panic!("DGESVD did not return expected output");
        }

        // full_matrices = false, compute_uv = false
        let driver = DGESVD::default().a(a.view()).full_matrices(false).compute_uv(false).build().unwrap();
        if let (s, None, None, _) = driver.run().unwrap() {
            assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
        } else {
            panic!("DGESVD did not return expected output");
        }
    }

    #[test]
    fn test_dgesdd() {
        let device = DeviceBLAS::default();
        let a_vec = &get_vec::<f64>('a')[..1024 * 512];
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DGESDD::default().a(a.view()).build().unwrap();
        if let (s, Some(u), Some(vt)) = driver.run().unwrap() {
            assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
            assert!((fingerprint(&u.abs()) - -1.9368850983570982).abs() < 1e-8);
            assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);
        } else {
            panic!("DGESDD did not return expected output");
        }

        // full_matrices = false
        let driver = DGESDD::default().a(a.view()).full_matrices(false).build().unwrap();
        if let (s, Some(u), Some(vt)) = driver.run().unwrap() {
            assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
            assert!((fingerprint(&u.abs()) - -9.144981428076894).abs() < 1e-8);
            assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);
        } else {
            panic!("DGESDD did not return expected output");
        }

        // full_matrices = false, compute_uv = false
        let driver = DGESDD::default().a(a.view()).full_matrices(false).compute_uv(false).build().unwrap();
        if let (s, None, None) = driver.run().unwrap() {
            assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
        } else {
            panic!("DGESDD did not return expected output");
        }
    }
}
