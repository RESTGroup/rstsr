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

    #[test]
    fn test_solve_general() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<f64>('b')[..1024 * 512].to_vec();
        let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_general((a.view(), b.view()));
        assert!((fingerprint(&x) - -1951.253447757597).abs() < 1e-8);

        // mutable changes itself
        rt::linalg::solve_general((a.view_mut(), b.view_mut()));
        assert!((fingerprint(&b) - -1951.253447757597).abs() < 1e-8);
    }

    #[test]
    fn test_solve_symmetric() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<f64>('b')[..1024 * 512].to_vec();
        let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_symmetric((a.view(), b.view()));
        assert!((fingerprint(&x) - -397.1203235513806).abs() < 1e-8);

        // upper, mutable changes b
        rt::linalg::solve_symmetric((a.view(), b.view_mut(), Upper));
        assert!((fingerprint(&b) - -314.45022891879034).abs() < 1e-8);
    }

    #[test]
    fn test_solve_triangular() {
        let device = DeviceBLAS::default();
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let mut a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_triangular((b.view(), a.view()));
        assert!((fingerprint(&x) - -2.6133848012216587).abs() < 1e-8);

        // upper, mutable changes a
        rt::linalg::solve_triangular((b.view(), a.view_mut(), Upper));
        assert!((fingerprint(&a) - 5.112256818100785).abs() < 1e-8);
    }
}
