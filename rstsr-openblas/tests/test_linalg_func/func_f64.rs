use rstsr::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_test_manifest::get_vec;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cholesky() {
        let device = DeviceBLAS::default();
        let mut b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device));

        // default
        let c = rt::linalg::cholesky(b.view());
        assert!((fingerprint(&c) - 43.21904478556176).abs() < 1e-8);

        // upper
        let c = rt::linalg::cholesky((b.view(), Upper));
        assert!((fingerprint(&c) - -25.925655124816647).abs() < 1e-8);

        // mutable changes itself
        rt::linalg::cholesky((b.view_mut(), Upper));
        assert!((fingerprint(&b) - -25.925655124816647).abs() < 1e-8);
    }

    #[test]
    fn test_eigh() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));
        let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device));

        // default, a
        let (w, v) = rt::linalg::eigh(a.view()).into();
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -9.903934930318247).abs() < 1e-8);

        // upper, a
        let (w, v) = rt::linalg::eigh((a.view(), Upper)).into();
        assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 6.973792268793419).abs() < 1e-8);

        // default, a b
        let (w, v) = rt::linalg::eigh((a.view(), b.view())).into();
        assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -5.243112559130817).abs() < 1e-8);

        // upper, a b, itype=3
        let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 3)).into();
        assert!((fingerprint(&w) - -2503.84161931662).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 152.17700520642055).abs() < 1e-8);

        // mutable changes a
        let (w, _) = rt::linalg::eigh(a.view_mut()).into();
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
        assert!((fingerprint(&a.abs()) - -9.903934930318247).abs() < 1e-8);
    }

    #[test]
    fn test_inv() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));

        // immutable
        let a_inv = rt::linalg::inv(a.view());
        assert!((fingerprint(&a_inv) - 143.39005577037764).abs() < 1e-8);

        // mutable
        rt::linalg::inv(a.view_mut());
        assert!((fingerprint(&a) - 143.39005577037764).abs() < 1e-8);
    }
}
