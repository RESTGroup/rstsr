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
    fn test_det() {
        let device = DeviceBLAS::default();
        let a_vec = get_vec::<f64>('a')[..5 * 5].to_vec();
        let mut a = rt::asarray((a_vec, [5, 5].c(), &device));

        let det = rt::linalg::det(a.view_mut());
        assert!((det - 3.9699917597338046).abs() < 1e-8);
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
    fn test_eigvalsh() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));
        let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device));

        // default, a
        let w = rt::linalg::eigvalsh(a.view());
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);

        // upper, a
        let w = rt::linalg::eigvalsh((a.view(), Upper));
        assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);

        // default, a b
        let w = rt::linalg::eigvalsh((a.view(), b.view()));
        assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);

        // upper, a b, itype=3
        let w = rt::linalg::eigvalsh((a.view(), b.view(), Upper, 3));
        assert!((fingerprint(&w) - -2503.84161931662).abs() < 1e-8);

        // mutable changes a
        let w = rt::linalg::eigvalsh(a.view_mut());
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
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
    fn test_pinv() {
        let device = DeviceBLAS::default();

        // 1024 x 512
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        assert!((fingerprint(&a_pinv) - 0.0878262837784408).abs() < 1e-8);
        assert_eq!(rank, 163);

        // 512 x 1024
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        assert!((fingerprint(&a_pinv) - -0.3244041253699862).abs() < 1e-8);
        assert_eq!(rank, 161);
    }

    #[test]
    fn test_slogdet() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));

        let (sign, logabsdet) = rt::linalg::slogdet(a.view_mut()).into();
        assert!(sign - -1.0 < 1e-8);
        assert!(logabsdet - 3031.1259211802403 < 1e-8);
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

    #[test]
    fn test_svd() {
        let device = DeviceBLAS::default();
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let (u, s, vt) = rt::linalg::svd(a.view()).into();
        assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -1.9368850983570982).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);

        // full_matrices = false
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -9.144981428076894).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);

        // m < n, full_matrices = false
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 32.27742168207757).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -3.716931052161584).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - -0.32301437281530243).abs() < 1e-8);
    }

    #[test]
    fn test_svdvals() {
        let device = DeviceBLAS::default();
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);

        // m < n, full_matrices = false
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 32.27742168207757).abs() < 1e-8);
    }
}
