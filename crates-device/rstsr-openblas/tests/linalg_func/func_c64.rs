use rstsr::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
use rstsr_test_manifest::get_vec;

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

macro_rules! c64 {
    ($real:expr, $imag:expr) => {
        c64::new($real, $imag)
    };
    ($real:expr) => {
        c64::new($real, 0.0)
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cholesky() {
        let device = DeviceBLAS::default();
        let mut b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device));

        // default
        let c = rt::linalg::cholesky(b.view());
        assert!((fingerprint(&c) - c64!(62.89494065393874, -73.47055443374522)).norm() < 1e-8);

        // upper
        let c = rt::linalg::cholesky((b.view(), Upper));
        assert!((fingerprint(&c) - c64!(13.720509103165073, -1.8066465348490963)).norm() < 1e-8);

        // mutable changes itself
        rt::linalg::cholesky((b.view_mut(), Upper));
        assert!((fingerprint(&b) - c64!(13.720509103165073, -1.8066465348490963)).norm() < 1e-8);
    }

    #[test]
    fn test_det() {
        let device = DeviceBLAS::default();
        let a_vec = get_vec::<c64>('a')[..5 * 5].to_vec();
        let mut a = rt::asarray((a_vec, [5, 5].c(), &device));

        let det = rt::linalg::det(a.view_mut());
        assert!((det - c64!(-24.808965756481086, 11.800248863799464)).norm() < 1e-8);
    }

    #[test]
    fn test_eigh() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));
        let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device));

        // default, a
        let (w, v) = rt::linalg::eigh(a.view()).into();
        assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -7.450761195788254).abs() < 1e-8);

        // upper, a
        let (w, v) = rt::linalg::eigh((a.view(), Upper)).into();
        assert!((fingerprint(&w) - -103.99103522434956).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -12.184946930165328).abs() < 1e-8);

        // default, a b
        let (w, v) = rt::linalg::eigh((a.view(), b.view())).into();
        assert!((fingerprint(&w) - -97.43376763322635).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -4.3181177983574255).abs() < 1e-8);

        // upper, a b, itype=3
        let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 3)).into();
        assert!((fingerprint(&w) - -4656.824753078057).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -0.15861903557045487).abs() < 1e-8);

        // mutable changes a
        let (w, _) = rt::linalg::eigh(a.view_mut()).into();
        assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);
        assert!((fingerprint(&a.abs()) - -7.450761195788254).abs() < 1e-8);
    }

    #[test]
    fn test_eigvalsh() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));
        let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device));

        // default, a
        let w = rt::linalg::eigvalsh(a.view());
        assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);

        // upper, a
        let w = rt::linalg::eigvalsh((a.view(), Upper));
        assert!((fingerprint(&w) - -103.99103522434956).abs() < 1e-8);

        // default, a b
        let w = rt::linalg::eigvalsh((a.view(), b.view()));
        assert!((fingerprint(&w) - -97.43376763322635).abs() < 1e-8);

        // upper, a b, itype=3
        let w = rt::linalg::eigvalsh((a.view(), b.view(), Upper, 3));
        assert!((fingerprint(&w) - -4656.824753078057).abs() < 1e-8);

        // mutable changes a
        let w = rt::linalg::eigvalsh(a.view_mut());
        assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);
    }

    #[test]
    fn test_inv() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

        // immutable
        let a_inv = rt::linalg::inv(a.view());
        assert!((fingerprint(&a_inv) - c64!(-11.836382515156183, 8.250167298349842)).norm() < 1e-8);

        // mutable
        rt::linalg::inv(a.view_mut());
        assert!((fingerprint(&a) - c64!(-11.836382515156183, 8.250167298349842)).norm() < 1e-8);
    }

    #[test]
    fn test_pinv() {
        let device = DeviceBLAS::default();

        // 1024 x 512
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        assert!((fingerprint(&a_pinv) - c64!(-0.03454885412959018, -0.023651876085623254)).norm() < 1e-8);
        assert_eq!(rank, 240);

        // 512 x 1024
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        assert!((fingerprint(&a_pinv) - c64!(-0.2814806469687325, -0.15198888300458474)).norm() < 1e-8);
        assert_eq!(rank, 240);
    }

    #[test]
    fn test_slogdet() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

        let (sign, logabsdet) = rt::linalg::slogdet(a.view_mut()).into();
        assert!((sign - c64!(-0.44606842323663365, 0.8949988613351316)).norm() < 1e-8);
        assert!(logabsdet - 3393.6720579594585 < 1e-8);
    }

    #[test]
    fn test_solve_general() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<c64>('b')[..1024 * 512].to_vec();
        let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_general((a.view(), b.view()));
        assert!((fingerprint(&x) - c64!(404.1900761036138, -258.5602505551204)).norm() < 1e-8);

        // mutable changes itself
        rt::linalg::solve_general((a.view_mut(), b.view_mut()));
        assert!((fingerprint(&b) - c64!(404.1900761036138, -258.5602505551204)).norm() < 1e-8);
    }

    #[test]
    fn test_solve_general_for_vec() {
        let device = DeviceBLAS::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<c64>('b')[..1024].to_vec();
        let mut b = rt::asarray((b_vec, [1024].c(), &device)).into_dim::<Ix1>();

        // default
        let x = rt::linalg::solve_general((a.view(), b.view()));
        assert!((fingerprint(&x) - c64!(-15.070310793269726, -1.987917054716041)).norm() < 1e-8);

        // mutable changes itself
        rt::linalg::solve_general((a.view_mut(), b.view_mut()));
        assert!((fingerprint(&b) - c64!(-15.070310793269726, -1.987917054716041)).norm() < 1e-8);
    }

    #[test]
    fn test_solve_symmetric() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<c64>('b')[..1024 * 512].to_vec();
        let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default (hermi)
        let x = rt::linalg::solve_symmetric((a.view(), b.view()));
        assert!((fingerprint(&x) - c64!(-1053.7242100144504, -559.2846004618166)).norm() < 1e-8);

        // upper (hermi)
        let x = rt::linalg::solve_symmetric((a.view(), b.view(), Upper));
        assert!((fingerprint(&x) - c64!(674.2725854112028, -68.55236080351166)).norm() < 1e-8);

        // default (symm)
        let x = rt::linalg::solve_symmetric((a.view(), b.view(), false));
        assert!((fingerprint(&x) - c64!(401.05642312535775, -805.8028453625365)).norm() < 1e-8);

        // upper, mutable changes b (symm)
        rt::linalg::solve_symmetric((a.view(), b.view_mut(), false, Upper));
        assert!((fingerprint(&b) - c64!(141.70122084637046, -829.609691493499)).norm() < 1e-8);
    }

    #[test]
    fn test_solve_triangular() {
        let device = DeviceBLAS::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let mut a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_triangular((b.view(), a.view()));
        assert!((fingerprint(&x) - c64!(-8.433708003916948, 20.578272827017052)).norm() < 1e-8);

        // upper, mutable changes a
        rt::linalg::solve_triangular((b.view(), a.view_mut(), Upper));
        assert!((fingerprint(&a) - c64!(0.1778922244846507, 11.42463765128442)).norm() < 1e-8);
    }

    #[test]
    fn test_svd() {
        let device = DeviceBLAS::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let (u, s, vt) = rt::linalg::svd(a.view()).into();
        assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -15.44133470545584).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 2.1605324161714172).abs() < 1e-8);

        // full_matrices = false
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -1.9516528722381659).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 2.1605324161714172).abs() < 1e-8);

        // m < n, full_matrices = false
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 47.599274835886646).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - 4.636614351700778).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 1.4497879458575658).abs() < 1e-8);
    }

    #[test]
    fn test_svdvals() {
        let device = DeviceBLAS::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);

        // m < n, full_matrices = false
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 47.599274835886646).abs() < 1e-8);
    }

    #[test]
    fn test_qr() {
        let device = DeviceBLAS::default();

        // Test with square matrix first
        let a_vec = get_vec::<c64>('a')[..512 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 512].c(), &device)).into_dim::<Ix2>();

        // Test tuple conversion (Q, R)
        let (q, r) = rt::linalg::qr(a.view()).into();
        assert!((fingerprint(&q.abs()) - 5.130192364112329).abs() < 1e-8);
        assert!((fingerprint(&r) - c64!(75.80674992700851, 164.615763513737)).norm() < 1e-8);
    }

    #[test]
    fn test_qr_nonsquare() {
        let device = DeviceBLAS::default();

        // Test with non-square matrix (M > N)
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // Test tuple conversion (Q, R)
        let (q, r) = rt::linalg::qr(a.view()).into();
        assert!((fingerprint(&q.abs()) - -10.06474255929977).abs() < 1e-8);
        assert!((fingerprint(&r) - c64!(-494.4676746210691, 38.38755919576653)).norm() < 1e-8);
    }

    #[test]
    fn test_qr_complete() {
        let device = DeviceBLAS::default();

        // Test 2: complete mode (M > N)
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // Test tuple conversion (Q, R)
        let (q, r) = rt::linalg::qr((a.view(), "complete")).into();
        assert!((fingerprint(&q.abs()) - 4.063591959250008).abs() < 1e-8);
        assert!((fingerprint(&r) - c64!(-494.46767462106914, 38.38755919576663)).norm() < 1e-8);
    }

    #[test]
    fn test_qr_m_less_n() {
        let device = DeviceBLAS::default();

        // Test 4: M < N case
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();

        // Test tuple conversion (Q, R)
        let (q, r) = rt::linalg::qr(a.view()).into();
        assert!((fingerprint(&q.abs()) - -0.7203599314373275).abs() < 1e-8);
        assert!((fingerprint(&r) - c64!(-582.4241295882298, 202.54615825826306)).norm() < 1e-8);
    }

    #[test]
    fn test_qr_pivoting() {
        let device = DeviceBLAS::default();

        // Test 5: pivoting - use (Q, R, P) tuple conversion
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let (q, r, p) = rt::linalg::qr((a.view(), "reduced", true)).into();
        assert!((fingerprint(&q.abs()) - 3.478992345356496).abs() < 1e-8);
        assert!((fingerprint(&r) - c64!(-1068.7288017773842, -37.65296060898508)).norm() < 1e-8);
        // Convert pivot indices to f64 for fingerprint
        let p_f64 = p.mapv(|x| x as f64);
        assert!((fingerprint(&p_f64) - -1940.0388752749188).abs() < 1e-8);
    }

    #[test]
    fn test_qr_r_mode() {
        let device = DeviceBLAS::default();

        // Test 'r' mode - only returns R (cannot use (Q, R) tuple conversion)
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let result = rt::linalg::qr((a.view(), "r"));
        assert!(result.q.is_none());
        let r = result.r.unwrap();
        assert!((fingerprint(&r) - c64!(-494.46767462106914, 38.38755919576663)).norm() < 1e-8);
        assert!(result.h.is_none());
        assert!(result.tau.is_none());
        assert!(result.p.is_none());
    }

    #[test]
    fn test_qr_raw_mode() {
        let device = DeviceBLAS::default();

        // Test 'raw' mode - returns packed H and tau
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
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
        assert!((fingerprint(&h) - c64!(-501.61932484481514, 34.79514296888962)).norm() < 1e-8);
        assert!((fingerprint(&tau) - c64!(0.895022938093218, -0.8172306823248144)).norm() < 1e-8);
    }

    #[test]
    fn test_qr_economic_mode() {
        let device = DeviceBLAS::default();

        // Test 'economic' mode - should be same as 'reduced' (SciPy terminology)
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // Compare 'economic' with 'reduced'
        let (q_red, r_red) = rt::linalg::qr((a.view(), "reduced")).into();
        let (q_econ, r_econ) = rt::linalg::qr((a.view(), "economic")).into();

        // Should produce identical results
        assert!((fingerprint(&q_red.abs()) - fingerprint(&q_econ.abs())).abs() < 1e-10);
        assert!((fingerprint(&r_red) - fingerprint(&r_econ)).norm() < 1e-10);
    }
}

#[cfg(test)]
mod test_eig {
    use super::*;

    #[test]
    fn test_eig() {
        let device = DeviceBLAS::default();
        // Test 1: Basic eig with right eigenvectors (default) - 64x64
        let a_vec = get_vec::<c64>('a')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();
        let result = rt::linalg::eig(a.view());
        let w = result.eigenvalues;
        let vr = result.right_eigenvectors.unwrap();
        assert!((fingerprint(&w.abs()) - 10.237632603014857).abs() < 1e-8);
        assert!((fingerprint(&vr.abs()) - -0.3765273892608302).abs() < 1e-8);
    }

    #[test]
    fn test_eig_both_eigenvectors() {
        let device = DeviceBLAS::default();
        // Test 2: eig with both left and right eigenvectors - 64x64
        let a_vec = get_vec::<c64>('a')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();
        let result = rt::linalg::eig((a.view(), true, true));
        let w = result.eigenvalues;
        let vl = result.left_eigenvectors.unwrap();
        let vr = result.right_eigenvectors.unwrap();
        assert!((fingerprint(&w.abs()) - 10.237632603014857).abs() < 1e-8);
        assert!((fingerprint(&vl.abs()) - 5.802237057668709).abs() < 1e-8);
        assert!((fingerprint(&vr.abs()) - -0.3765273892608302).abs() < 1e-8);
    }

    #[test]
    fn test_eig_left_only() {
        let device = DeviceBLAS::default();
        // Test 3: eig with left eigenvectors only - 64x64
        let a_vec = get_vec::<c64>('a')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();
        let result = rt::linalg::eig((a.view(), true, false));
        let w = result.eigenvalues;
        let vl = result.left_eigenvectors.unwrap();
        assert!(result.right_eigenvectors.is_none());
        assert!((fingerprint(&w.abs()) - 10.237632603014857).abs() < 1e-8);
        assert!((fingerprint(&vl.abs()) - 5.802237057668709).abs() < 1e-8);
    }

    #[test]
    fn test_eigvals() {
        let device = DeviceBLAS::default();
        // Test 4: eigvals (eigenvalues only) - 64x64
        let a_vec = get_vec::<c64>('a')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();
        let w = rt::linalg::eigvals(a.view());
        assert!((fingerprint(&w.abs()) - 10.237632603014857).abs() < 1e-8);
    }

    #[test]
    fn test_eig_larger_matrix() {
        let device = DeviceBLAS::default();
        // Test 5: Larger matrix - 512x512
        let a_vec = get_vec::<c64>('a')[..512 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 512].c(), &device)).into_dim::<Ix2>();
        let result = rt::linalg::eig(a.view());
        let w = result.eigenvalues;
        let vr = result.right_eigenvectors.unwrap();
        assert!((fingerprint(&w.abs()) - 16.84923087126611).abs() < 1e-8);
        assert!((fingerprint(&vr.abs()) - (-0.2098623126454835)).abs() < 1e-8);
    }

    #[test]
    fn test_eig_generalized_32x32() {
        let device = DeviceBLAS::default();
        // Test 6: Generalized eigenvalue problem - 32x32
        let a_vec = get_vec::<c64>('a')[..32 * 32].to_vec();
        let b_vec = get_vec::<c64>('b')[..32 * 32].to_vec();
        let a = rt::asarray((a_vec, [32, 32].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec, [32, 32].c(), &device)).into_dim::<Ix2>();
        let result = rt::linalg::eig((a.view(), b.view()));
        let w = result.eigenvalues;
        assert!((fingerprint(&w.abs()) - 9.593381733904854).abs() < 1e-8);
    }

    #[test]
    fn test_eig_generalized_64x64() {
        let device = DeviceBLAS::default();
        // Test 7: Generalized eigenvalue problem - 64x64
        let a_vec = get_vec::<c64>('a')[..64 * 64].to_vec();
        let b_vec = get_vec::<c64>('b')[..64 * 64].to_vec();
        let a = rt::asarray((a_vec, [64, 64].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec, [64, 64].c(), &device)).into_dim::<Ix2>();
        let result = rt::linalg::eig((a.view(), b.view()));
        let w = result.eigenvalues;
        assert!((fingerprint(&w.abs()) - 15.169874441187222).abs() < 1e-8);
    }

    #[test]
    fn test_eig_generalized_both_eigenvectors() {
        let device = DeviceBLAS::default();
        // Test 8: Generalized eig with left and right eigenvectors - 32x32
        let a_vec = get_vec::<c64>('a')[..32 * 32].to_vec();
        let b_vec = get_vec::<c64>('b')[..32 * 32].to_vec();
        let a = rt::asarray((a_vec, [32, 32].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec, [32, 32].c(), &device)).into_dim::<Ix2>();
        let result = rt::linalg::eig((a.view(), b.view(), true, true));
        let w = result.eigenvalues;
        let vl = result.left_eigenvectors.unwrap();
        let vr = result.right_eigenvectors.unwrap();
        assert!((fingerprint(&w.abs()) - 9.593381733904858).abs() < 1e-8);
        assert!((fingerprint(&vl.abs()) - 3.7152914807178323).abs() < 1e-8);
        assert!((fingerprint(&vr.abs()) - 2.6849725877054844).abs() < 1e-8);
    }

    #[test]
    fn test_eig_generalized_128x128() {
        let device = DeviceBLAS::default();
        // Test 9: Generalized eigenvalue problem - 128x128
        let a_vec = get_vec::<c64>('a')[..128 * 128].to_vec();
        let b_vec = get_vec::<c64>('b')[..128 * 128].to_vec();
        let a = rt::asarray((a_vec, [128, 128].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec, [128, 128].c(), &device)).into_dim::<Ix2>();
        let result = rt::linalg::eig((a.view(), b.view()));
        let w = result.eigenvalues;
        assert!((fingerprint(&w.abs()) - 21.069841281001473).abs() < 1e-8);
    }

    #[test]
    fn test_eig_generalized_simple_2x2() {
        let device = DeviceBLAS::default();
        // Test 10: Simple 2x2 generalized eigenvalue problem (complex)
        let a_simple =
            rt::asarray((vec![c64!(1.0, 0.5), c64!(2.0, -0.3), c64!(3.0, 0.2), c64!(4.0, -0.1)], [2, 2].c(), &device))
                .into_dim::<Ix2>();
        let b_simple =
            rt::asarray((vec![c64!(2.0, 0.1), c64!(1.0, -0.2), c64!(1.0, 0.3), c64!(2.0, -0.4)], [2, 2].c(), &device))
                .into_dim::<Ix2>();
        let result = rt::linalg::eig((a_simple.view(), b_simple.view()));
        let w = result.eigenvalues;
        assert!((fingerprint(&w.abs()) - 2.224853629259969).abs() < 1e-8);
    }
}

#[cfg(test)]
mod test_generalized_eigh {
    use super::*;

    #[test]
    fn test_generalized_eigh() {
        let device = DeviceBLAS::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 1024].to_vec();
        let b_vec = get_vec::<c64>('b')[..1024 * 1024].to_vec();
        let a = rt::asarray((a_vec, [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec, [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // 1, lower
        let (w, v) = rt::linalg::eigh((a.view(), b.view(), Lower, 1)).into();
        println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
        assert!((fingerprint(&w) - -97.43376763322635).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -4.3181177983574255).abs() < 1e-8);

        // 1, upper
        let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 1)).into();
        println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
        assert!((fingerprint(&w) - -54.81859256480441).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -1.4841788446757156).abs() < 1e-8);

        // 2, lower
        let (w, v) = rt::linalg::eigh((a.view(), b.view(), Lower, 2)).into();
        println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
        assert!((fingerprint(&w) - -4967.627482507203).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 5.541034627252399).abs() < 1e-8);

        // 2, upper
        let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 2)).into();
        println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
        assert!((fingerprint(&w) - -4656.824753078057).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 1.0609263552377188).abs() < 1e-8);

        // 3, lower
        let (w, v) = rt::linalg::eigh((a.view(), b.view(), Lower, 3)).into();
        println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
        assert!((fingerprint(&w) - -4967.627482507203).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 118.76501084045631).abs() < 1e-8);

        // 3, upper
        let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 3)).into();
        println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
        assert!((fingerprint(&w) - -4656.824753078057).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -0.15861903557045487).abs() < 1e-8);
    }
}
