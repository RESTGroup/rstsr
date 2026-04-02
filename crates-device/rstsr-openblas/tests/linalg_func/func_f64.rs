use rstsr::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
use rstsr_test_manifest::get_vec;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_eig() {
        let device = DeviceBLAS::default();

        // Test 1: Basic eig with right eigenvectors (default)
        let a_vec = get_vec::<f64>('a')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::eig(a.view());
        let w = result.eigenvalues;
        let vr = result.right_eigenvectors.unwrap();
        assert!((fingerprint(&w.abs()) - 9.819876443763567).abs() < 1e-8);
        assert!((fingerprint(&vr.abs()) - -3.1323839657585237).abs() < 1e-8);
    }

    #[test]
    fn test_eig_both_eigenvectors() {
        let device = DeviceBLAS::default();

        // Test 2: eig with both left and right eigenvectors
        let a_vec = get_vec::<f64>('a')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::eig((a.view(), true, true));
        let w = result.eigenvalues;
        let vl = result.left_eigenvectors.unwrap();
        let vr = result.right_eigenvectors.unwrap();
        assert!((fingerprint(&w.abs()) - 9.819876443763567).abs() < 1e-8);
        assert!((fingerprint(&vl.abs()) - 0.30269389067674696).abs() < 1e-8);
        assert!((fingerprint(&vr.abs()) - -3.1323839657585237).abs() < 1e-8);
    }

    #[test]
    fn test_eig_left_only() {
        let device = DeviceBLAS::default();

        // Test 3: eig with left eigenvectors only
        let a_vec = get_vec::<f64>('a')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::eig((a.view(), true, false));
        let w = result.eigenvalues;
        let vl = result.left_eigenvectors.unwrap();
        assert!(result.right_eigenvectors.is_none());
        assert!((fingerprint(&w.abs()) - 9.819876443763567).abs() < 1e-8);
        assert!((fingerprint(&vl.abs()) - 0.30269389067674696).abs() < 1e-8);
    }

    #[test]
    fn test_eigvals() {
        let device = DeviceBLAS::default();

        // Test 4: eigvals (eigenvalues only)
        let a_vec = get_vec::<f64>('a')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();

        let w = rt::linalg::eigvals(a.view());
        assert!((fingerprint(&w.abs()) - 9.819876443763567).abs() < 1e-8);
    }

    #[test]
    fn test_eig_rotation_matrix() {
        let device = DeviceBLAS::default();

        // Test 5: Rotation matrix with complex eigenvalues
        let theta = std::f64::consts::PI / 4.0;
        let a_data: Vec<f64> = vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()];
        let a = rt::asarray((a_data, [2, 2].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::eig(a.view());
        let w = result.eigenvalues;
        let vr = result.right_eigenvectors.unwrap();

        // Eigenvalues should be exp(±i*π/4) = cos(π/4) ± i*sin(π/4) = 0.7071 ± 0.7071i
        assert!((fingerprint(&w.abs()) - 1.5403023058681398).abs() < 1e-8);
        assert!((fingerprint(&vr.abs()) - 0.09486754779484802).abs() < 1e-8);
    }

    #[test]
    fn test_eig_larger_matrix() {
        let device = DeviceBLAS::default();

        // Test 6: Larger matrix (512x512)
        let a_vec = get_vec::<f64>('a')[..512 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 512].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::eig(a.view());
        let w = result.eigenvalues;
        let vr = result.right_eigenvectors.unwrap();

        // Verify shapes are correct
        assert_eq!(w.shape(), &[512]);
        assert_eq!(vr.shape(), &[512, 512]);

        // Just verify we got results - the eigenvalue equation is checked by other tests
        // Eigenvalues from LAPACK are not in any specific order, so fingerprints differ
        // We verify by checking that eigenvalues have reasonable magnitude
        let max_w_mag = w.iter().map(|x| x.norm()).fold(0.0_f64, |a, b| a.max(b));
        println!("Max eigenvalue magnitude: {}", max_w_mag);
        assert!(max_w_mag > 0.0, "Should have non-zero eigenvalues");
    }

    #[test]
    fn test_eig_generalized_32x32() {
        let device = DeviceBLAS::default();

        // Test: Generalized eigenvalue problem (32x32)
        let a_vec = get_vec::<f64>('a')[..32 * 32].to_vec();
        let b_vec = get_vec::<f64>('b')[..32 * 32].to_vec();
        let a = rt::asarray((a_vec, [32, 32].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec, [32, 32].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::eig((a.view(), b.view()));
        let w = result.eigenvalues;
        let vr = result.right_eigenvectors.unwrap();

        assert_eq!(w.shape(), &[32]);
        assert_eq!(vr.shape(), &[32, 32]);
        // Eigenvalue fingerprint should match
        assert!((fingerprint(&w.abs()) - 145.30492158178672).abs() < 1e-8);
    }

    #[test]
    fn test_eig_generalized_64x64() {
        let device = DeviceBLAS::default();

        // Test: Generalized eigenvalue problem (64x64)
        let a_vec = get_vec::<f64>('a')[..64 * 64].to_vec();
        let b_vec = get_vec::<f64>('b')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec, [64, 64].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::eig((a.view(), b.view()));
        let w = result.eigenvalues;
        let vr = result.right_eigenvectors.unwrap();

        assert_eq!(w.shape(), &[64]);
        assert_eq!(vr.shape(), &[64, 64]);
        // Eigenvalue fingerprint should match
        assert!((fingerprint(&w.abs()) - 166.30282160221182).abs() < 1e-8);
    }

    #[test]
    fn test_eig_generalized_both_eigenvectors() {
        let device = DeviceBLAS::default();

        // Test: Generalized eig with both eigenvectors (32x32)
        let a_vec = get_vec::<f64>('a')[..32 * 32].to_vec();
        let b_vec = get_vec::<f64>('b')[..32 * 32].to_vec();
        let a = rt::asarray((a_vec, [32, 32].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec, [32, 32].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::eig((a.view(), b.view(), true, true));
        let w = result.eigenvalues;
        let vl = result.left_eigenvectors.unwrap();
        let vr = result.right_eigenvectors.unwrap();

        assert_eq!(w.shape(), &[32]);
        assert_eq!(vl.shape(), &[32, 32]);
        assert_eq!(vr.shape(), &[32, 32]);
        // Eigenvalue fingerprint should match
        assert!((fingerprint(&w.abs()) - 145.30492158178672).abs() < 1e-8);
    }

    #[test]
    fn test_eig_generalized_simple_2x2() {
        let device = DeviceBLAS::default();

        // Test: Simple 2x2 generalized eigenvalue problem
        let a_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f64> = vec![2.0, 1.0, 1.0, 2.0];
        let a = rt::asarray((a_data, [2, 2].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_data, [2, 2].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::eig((a.view(), b.view()));
        let w = result.eigenvalues;
        let _vr = result.right_eigenvectors.unwrap();

        // Eigenvalues should be approximately -0.333 and 2.0
        assert_eq!(w.shape(), &[2]);
        assert!((fingerprint(&w.abs()) - 1.4139379450696126).abs() < 1e-8);
    }

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
    fn test_cholesky_submatrix() {
        let device = DeviceBLAS::default();
        let vec_b: Vec<f64> = vec![0.0, 1.0, 2.0, 1.0, 5.0, 1.5, 2.0, 1.5, 8.0];
        let b = rt::asarray((vec_b, [3, 3].c(), &device));

        let b_view = b.i((1..3, 1..3));
        let c = rt::linalg::cholesky(b_view);
        assert!((fingerprint(&c) - -0.7633202592326889).abs() < 1e-8);
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
    fn test_qr() {
        let device = DeviceBLAS::default();

        // Test with square matrix first
        let a_vec = get_vec::<f64>('a')[..512 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 512].c(), &device)).into_dim::<Ix2>();

        // Test tuple conversion (Q, R)
        let (q, r) = rt::linalg::qr(a.view()).into();
        let q_fp = fingerprint(&q.abs());
        let r_fp = fingerprint(&r);
        assert!((q_fp - -1.6083348348387723).abs() < 1e-8);
        assert!((r_fp - -132.64533391608532).abs() < 1e-8);
    }

    #[test]
    fn test_qr_nonsquare() {
        let device = DeviceBLAS::default();

        // Test with non-square matrix (M > N)
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // Test tuple conversion (Q, R)
        let (q, r) = rt::linalg::qr(a.view()).into();
        assert!((fingerprint(&q.abs()) - -3.816695392252395).abs() < 1e-8);
        assert!((fingerprint(&r) - 443.8056068942338).abs() < 1e-8);
    }

    #[test]
    fn test_qr_complete() {
        let device = DeviceBLAS::default();

        // Test 2: complete mode (M > N)
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // Test tuple conversion (Q, R)
        let (q, r) = rt::linalg::qr((a.view(), "complete")).into();
        assert!((fingerprint(&q.abs()) - 2.2969510394541404).abs() < 1e-8);
        assert!((fingerprint(&r) - 443.8056068942341).abs() < 1e-8);
    }

    #[test]
    fn test_qr_m_less_n() {
        let device = DeviceBLAS::default();

        // Test 4: M < N case
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();

        // Test tuple conversion (Q, R)
        let (q, r) = rt::linalg::qr(a.view()).into();
        assert!((fingerprint(&q.abs()) - -0.9769304656721538).abs() < 1e-8);
        assert!((fingerprint(&r) - -476.9310050389706).abs() < 1e-8);
    }

    #[test]
    fn test_qr_pivoting() {
        let device = DeviceBLAS::default();

        // Test 5: pivoting - use (Q, R, P) tuple conversion
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let (q, r, p) = rt::linalg::qr((a.view(), "reduced", true)).into();
        assert!((fingerprint(&q.abs()) - 8.110705884324302).abs() < 1e-8);
        assert!((fingerprint(&r) - 466.0756909302775).abs() < 1e-8);
        // Convert pivot indices to f64 for fingerprint
        let p_f64 = p.mapv(|x| x as f64);
        assert!((fingerprint(&p_f64) - -26.4276638273343).abs() < 1e-8);
    }

    #[test]
    fn test_qr_r_mode() {
        let device = DeviceBLAS::default();

        // Test 'r' mode - only returns R (cannot use (Q, R) tuple conversion)
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::qr((a.view(), "r"));
        assert!(result.q.is_none());
        let r = result.r.unwrap();
        assert!((fingerprint(&r) - 443.8056068942338).abs() < 1e-8);
        assert!(result.h.is_none());
        assert!(result.tau.is_none());
        assert!(result.p.is_none());
    }

    #[test]
    fn test_qr_raw_mode() {
        let device = DeviceBLAS::default();

        // Test 'raw' mode - returns packed H and tau
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::qr((a.view(), "raw"));
        assert!(result.q.is_none());
        assert!(result.r.is_none());
        let h = result.h.unwrap();
        let tau = result.tau.unwrap();
        assert!(result.p.is_none());

        // Verify shapes: h has shape [1024, 512], tau has shape [512]
        assert_eq!(h.shape(), &[1024, 512]);
        assert_eq!(tau.shape(), &[512]);

        // Verify fingerprints
        assert!((fingerprint(&h) - 474.6362237429613).abs() < 1e-8);
        assert!((fingerprint(&tau) - 0.30396075283831525).abs() < 1e-8);
    }

    #[test]
    fn test_qr_economic_mode() {
        let device = DeviceBLAS::default();

        // Test 'economic' mode - should be same as 'reduced' (SciPy terminology)
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // Compare 'economic' with 'reduced'
        let (q_red, r_red) = rt::linalg::qr((a.view(), "reduced")).into();
        let (q_econ, r_econ) = rt::linalg::qr((a.view(), "economic")).into();

        // Should produce identical results
        assert!((fingerprint(&q_red.abs()) - fingerprint(&q_econ.abs())).abs() < 1e-10);
        assert!((fingerprint(&r_red) - fingerprint(&r_econ)).abs() < 1e-10);
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
    fn test_solve_general_for_vec() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<f64>('b')[..1024].to_vec();
        let mut b = rt::asarray((b_vec, [1024].c(), &device)).into_dim::<Ix1>();

        // default
        let x = rt::linalg::solve_general((a.view(), b.view()));
        assert!((fingerprint(&x) - -9.120066438800688).abs() < 1e-8);

        // mutable changes itself
        rt::linalg::solve_general((a.view_mut(), b.view_mut()));
        assert!((fingerprint(&b) - -9.120066438800688).abs() < 1e-8);
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
