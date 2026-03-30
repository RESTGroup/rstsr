use rstsr_blas_traits::lapack_qr::*;
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
use rstsr_test_manifest::get_vec;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dgeqrf() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [512, 512].c(), &device)).into_dim::<Ix2>();

        // Test GEQRF
        let driver = DGEQRF::default().a(a.view()).build().unwrap();
        let (qr, tau) = driver.run().unwrap();
        let qr = qr.into_owned();

        // Verify fingerprints (from Python validation: dgeqrf)
        assert!((fingerprint(&qr.abs()) - 104.85191447485762).abs() < 1e-8);
        assert!((fingerprint(&tau) - 2.599625942247915).abs() < 1e-8);
    }

    #[test]
    fn test_dorgqr() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [512, 512].c(), &device)).into_dim::<Ix2>();

        // First do GEQRF
        let driver = DGEQRF::default().a(a.view()).build().unwrap();
        let (qr, tau) = driver.run().unwrap();

        // Then do ORGQR to get Q
        let driver = DORGQR::default().a(qr.view()).tau(tau.view()).build().unwrap();
        let q = driver.run().unwrap();
        let q = q.into_owned();

        // Verify fingerprint (from Python validation: dorgqr)
        assert!((fingerprint(&q.abs()) - (-1.6083348348387112)).abs() < 1e-8);
    }

    #[test]
    fn test_dgeqp3() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [512, 512].c(), &device)).into_dim::<Ix2>();

        // Test GEQP3 (QR with pivoting)
        let driver = DGEQP3::default().a(a.view()).build().unwrap();
        let (qr, jpvt, tau) = driver.run().unwrap();
        let qr = qr.into_owned();

        // Verify fingerprints (from Python validation: dgeqp3)
        assert!((fingerprint(&qr.abs()) - 82.18652098057946).abs() < 1e-8);
        assert!((fingerprint(&tau) - 0.9651077849682483).abs() < 1e-8);

        // jpvt should be a permutation (0-indexed after conversion)
        let mut jpvt_sorted: Vec<i32> = jpvt.into_vec();
        jpvt_sorted.sort();
        for (i, &j) in jpvt_sorted.iter().enumerate() {
            assert_eq!(j, i as i32);
        }
    }

    #[test]
    fn test_dormqr() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [512, 512].c(), &device)).into_dim::<Ix2>();
        let c = rt::asarray((get_vec::<f64>('b'), [512, 256].c(), &device)).into_dim::<Ix2>();

        // First do GEQRF
        let driver = DGEQRF::default().a(a.view()).build().unwrap();
        let (qr, tau) = driver.run().unwrap();

        // Then do ORMQR: Q^T * C
        let driver =
            DORMQR::default().a(qr.view()).tau(tau.view()).c(c.view()).side(Left).trans(Trans).build().unwrap();
        let result = driver.run().unwrap();
        let result = result.into_owned();

        // Verify fingerprint (from Python validation: dormqr)
        assert!((fingerprint(&result.abs()) - 14.485921558590004).abs() < 1e-6);
    }
}
