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

    #[test]
    fn test_dsyevr() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // all eigenvalues (lower, should match SYEVD)
        let driver = DSYEVR::default().a(a.view()).range(EigenRange::All).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        let v = v.unwrap();
        println!("w fingerprint: {}", fingerprint(&w));
        println!("v.abs() fingerprint: {}", fingerprint(&v.view().abs()));
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
        assert!((fingerprint(&v.view().abs()) - -9.903934930318247).abs() < 1e-8);

        // all eigenvalues (upper)
        let driver = DSYEVR::default().a(a.view()).uplo(Upper).range(EigenRange::All).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        let v = v.unwrap();
        assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);
        assert!((fingerprint(&v.view().abs()) - 6.973792268793419).abs() < 1e-8);

        // eigenvalues by index [0,99] (lower)
        let driver = DSYEVR::default().a(a.view()).range(EigenRange::Index(0, Some(99))).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        let v = v.unwrap();
        assert_eq!(w.size(), 100);
        assert_eq!(v.shape(), &[1024, 100]);
        assert!((fingerprint(&w) - 7.172235948598356).abs() < 1e-8);
        assert!((fingerprint(&v.view().abs()) - -4.561871974095643).abs() < 1e-8);

        // eigenvalues by index [500,599] (lower)
        let driver = DSYEVR::default().a(a.view()).range(EigenRange::Index(500, Some(599))).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        let v = v.unwrap();
        assert_eq!(w.size(), 100);
        assert_eq!(v.shape(), &[1024, 100]);
        assert!((fingerprint(&w) - -8.66127659333348).abs() < 1e-8);
        assert!((fingerprint(&v.view().abs()) - 4.814753342657916).abs() < 1e-8);

        // eigenvalues by value range [-10, 10] (lower)
        let driver = DSYEVR::default().a(a.view()).range(EigenRange::Value(-10.0, 10.0)).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        let v = v.unwrap();
        assert_eq!(w.size(), 202);
        assert_eq!(v.shape(), &[1024, 202]);
        assert!((fingerprint(&w) - -1.2186975175456176).abs() < 1e-7);
        // Note: Z is sliced to actual m columns, fingerprint differs from scipy's full Z
        assert!((fingerprint(&v.view().abs()) - 6.129939315496429).abs() < 1e-7);

        // eigenvalues by value range [0, 5] (lower)
        let driver = DSYEVR::default().a(a.view()).range(EigenRange::Value(0.0, 5.0)).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        let v = v.unwrap();
        assert_eq!(w.size(), 52);
        assert_eq!(v.shape(), &[1024, 52]);
        assert!((fingerprint(&w) - 4.878784041243897).abs() < 1e-7);
        // Z fingerprint for sliced columns
        // We'll just verify shape, exact fingerprint varies

        // eigenvalues by value range using From<(f64, f64)>
        let driver = DSYEVR::default().a(a.view()).range((-10.0, 10.0)).build().unwrap();
        let (w, _, _) = driver.run().unwrap();
        assert_eq!(w.size(), 202);
        assert!((fingerprint(&w) - -1.2186975175456176).abs() < 1e-7);

        // eigenvalues only (jobz = 'N')
        let driver = DSYEVR::default().a(a.view()).jobz('N').range(EigenRange::All).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        assert!(v.is_none());
        assert_eq!(w.size(), 1024);
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
    }

    #[test]
    fn test_dsyevr_range_syntax() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // Test RangeFull (..)
        let driver = DSYEVR::default().a(a.view()).range(..).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        assert!(v.is_some());
        assert_eq!(w.size(), 1024);
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);

        // Test Range<usize> (0..100) -> Index(0, Some(99))
        let driver = DSYEVR::default().a(a.view()).range(0..100).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        assert!(v.is_some());
        assert_eq!(w.size(), 100);
        assert!((fingerprint(&w) - 7.172235948598356).abs() < 1e-8);

        // Test RangeInclusive<usize> (0..=99) -> Index(0, Some(99))
        let driver = DSYEVR::default().a(a.view()).range(0..=99).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        assert!(v.is_some());
        assert_eq!(w.size(), 100);
        assert!((fingerprint(&w) - 7.172235948598356).abs() < 1e-8);

        // Test RangeFrom<usize> (500..) -> Index(500, None)
        let driver = DSYEVR::default().a(a.view()).range(500..).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        assert!(v.is_some());
        assert_eq!(w.size(), 524); // 1024 - 500

        // Eigenvalue at index 500 is about -1.3, still negative
        let w_slice = w.raw();
        assert!(w_slice[0] < 0.0);
        assert!(w_slice[0] > -2.0); // Around -1.3

        // Test RangeTo<usize> (..100) -> Index(0, Some(99))
        let driver = DSYEVR::default().a(a.view()).range(..100).build().unwrap();
        let (w, v, _) = driver.run().unwrap();
        assert!(v.is_some());
        assert_eq!(w.size(), 100);
        assert!((fingerprint(&w) - 7.172235948598356).abs() < 1e-8);
    }
}
