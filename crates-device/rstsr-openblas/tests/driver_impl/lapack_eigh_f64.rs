use rstsr_blas_traits::lapack_eigh::*;
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
use rstsr_test_manifest::get_vec;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dsyevd() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DSYEVD::default().a(a.view()).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -9.903934930318247).abs() < 1e-8);

        // upper for c-contiguous
        let driver = DSYEVD::default().a(a.view()).uplo(Upper).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 6.973792268793419).abs() < 1e-8);

        // transpose upper for c-contiguous
        let driver = DSYEVD::default().a(a.t()).uplo(Upper).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -9.903934930318247).abs() < 1e-8);
    }

    #[test]
    fn test_dsyev() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DSYEV::default().a(a.view()).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -9.903934930318247).abs() < 1e-8);

        // upper for c-contiguous
        let driver = DSYEV::default().a(a.view()).uplo(Upper).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 6.973792268793419).abs() < 1e-8);

        // transpose upper for c-contiguous
        let driver = DSYEV::default().a(a.t()).uplo(Upper).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -9.903934930318247).abs() < 1e-8);
    }

    #[test]
    fn test_dsygvd() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DSYGVD::default().a(a.view()).b(b.view()).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -5.243112559130817).abs() < 1e-8);

        // upper for c-contiguous
        let driver = DSYGVD::default().a(a.view()).b(b.view()).uplo(Upper).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -65.27252612342873).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -7.0849504857534535).abs() < 1e-8);

        // transpose upper for c-contiguous
        let driver = DSYGVD::default().a(a.t()).b(b.t()).uplo(Upper).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -5.243112559130817).abs() < 1e-8);

        // itype 2
        let driver = DSYGVD::default().a(a.view()).b(b.view()).itype(2).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -2437.094304861363).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -4.108281604767547).abs() < 1e-8);

        // itype 3
        let driver = DSYGVD::default().a(a.view()).b(b.view()).itype(3).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -2437.094304861363).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 30.756098926747757).abs() < 1e-8);
    }

    #[test]
    fn test_dsygv() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let driver = DSYGV::default().a(a.view()).b(b.view()).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -5.243112559130817).abs() < 1e-8);

        // upper for c-contiguous
        let driver = DSYGV::default().a(a.view()).b(b.view()).uplo(Upper).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -65.27252612342873).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -7.0849504857534535).abs() < 1e-8);

        // transpose upper for c-contiguous
        let driver = DSYGV::default().a(a.t()).b(b.t()).uplo(Upper).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -5.243112559130817).abs() < 1e-8);

        // itype 2
        let driver = DSYGV::default().a(a.view()).b(b.view()).itype(2).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -2437.094304861363).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -4.108281604767547).abs() < 1e-8);

        // itype 3
        let driver = DSYGV::default().a(a.view()).b(b.view()).itype(3).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&w) - -2437.094304861363).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 30.756098926747757).abs() < 1e-8);
    }
}
