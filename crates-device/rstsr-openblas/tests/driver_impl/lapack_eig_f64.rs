use num::Complex;
use rstsr_blas_traits::lapack_eig::*;
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
use rstsr_test_manifest::get_vec;

/// Compute fingerprint of eigenvalues after sorting by complex value.
/// This makes the comparison implementation-agnostic (works with both MKL and OpenBLAS).
fn sorted_eigenvalue_fingerprint(w: &Tensor<Complex<f64>, DeviceBLAS, Ix1>) -> f64 {
    let w_vec = w.raw();
    let mut eigenvalues: Vec<Complex<f64>> = w_vec.clone();
    // Sort by real part first, then imaginary part
    eigenvalues.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap().then_with(|| a.im.partial_cmp(&b.im).unwrap()));
    // Compute fingerprint on sorted eigenvalues (use real parts)
    let n = eigenvalues.len();
    (0..n).map(|i| (i as f64).cos() * eigenvalues[i].re).sum()
}

/// Compute fingerprint of generalized eigenvalues (alpha/beta) after sorting.
/// Eigenvalues are computed as alpha/beta.
fn sorted_ggev_eigenvalue_fingerprint(
    alpha: &Tensor<Complex<f64>, DeviceBLAS, Ix1>,
    beta: &Tensor<Complex<f64>, DeviceBLAS, Ix1>,
) -> f64 {
    let alpha_vec = alpha.raw();
    let beta_vec = beta.raw();
    let n = alpha_vec.len();
    let mut eigenvalues: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let a = alpha_vec[i];
            let b = beta_vec[i];
            if b.norm() > 1e-15 {
                a / b
            } else {
                // Handle infinite eigenvalues
                Complex::new(f64::INFINITY, 0.0)
            }
        })
        .collect();
    // Filter out infinite eigenvalues for fingerprint
    eigenvalues.retain(|e| e.re.is_finite());
    // Sort by real part first, then imaginary part
    eigenvalues.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap().then_with(|| a.im.partial_cmp(&b.im).unwrap()));
    // Compute fingerprint on sorted eigenvalues
    let n = eigenvalues.len();
    (0..n).map(|i| (i as f64).cos() * eigenvalues[i].re).sum()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dgeev_simple() {
        // Simple 2x2 matrix with known eigenvalues
        // Matrix: [[1, 2], [3, 4]]
        // Eigenvalues: (5±sqrt(33))/2 ≈ 5.372, -0.372
        let device = DeviceBLAS::default();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let a = rt::asarray((data.clone(), [2, 2].c(), &device)).into_dim::<Ix2>();

        let driver = GEEV::default().a(a.view()).left(false).right(true).build().unwrap();
        let (w, _vl, _vr, _a) = driver.run().unwrap();

        // Check sorted eigenvalue fingerprint
        let fp = sorted_eigenvalue_fingerprint(&w);
        // Expected: sum of cos(i) * sorted_eigenvalues for eigenvalues 5.372 and -0.372
        assert!((fp - 2.5303746634655755).abs() < 1e-8);
    }

    #[test]
    fn test_dgeev_right_only() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [512, 512].c(), &device)).into_dim::<Ix2>();

        // Default (right eigenvectors only)
        let driver = GEEV::default().a(a.view()).left(false).right(true).build().unwrap();
        let (w, _vl, vr, _a_out) = driver.run().unwrap();

        let sorted_fp = sorted_eigenvalue_fingerprint(&w);
        assert!((sorted_fp - (-0.20985324861994137)).abs() < 1e-8);

        // Verify eigenvector was computed
        let _vr = vr.unwrap();
    }

    #[test]
    fn test_dgeev_left_only() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [512, 512].c(), &device)).into_dim::<Ix2>();

        let driver = GEEV::default().a(a.view()).left(true).right(false).build().unwrap();
        let (w, vl, vr, _a) = driver.run().unwrap();

        // Use sorted eigenvalue fingerprint (implementation-agnostic)
        let sorted_fp = sorted_eigenvalue_fingerprint(&w);
        assert!((sorted_fp - (-0.20985324861994137)).abs() < 1e-8);
        assert!(vr.is_none());

        // Verify left eigenvector was computed
        let _vl = vl.unwrap();
    }

    #[test]
    fn test_dgeev_both() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [512, 512].c(), &device)).into_dim::<Ix2>();

        let driver = GEEV::default().a(a.view()).left(true).right(true).build().unwrap();
        let (w, vl, vr, _a) = driver.run().unwrap();

        // Use sorted eigenvalue fingerprint (implementation-agnostic)
        let sorted_fp = sorted_eigenvalue_fingerprint(&w);
        assert!((sorted_fp - (-0.20985324861994137)).abs() < 1e-8);

        // Verify eigenvectors were computed
        let _vl = vl.unwrap();
        let _vr = vr.unwrap();
    }

    #[test]
    fn test_dgeev_column_major() {
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [512, 512].f(), &device)).into_dim::<Ix2>();

        let driver = GEEV::default().a(a.view()).left(false).right(true).build().unwrap();
        let (w, _vl, vr, _a_out) = driver.run().unwrap();

        // Same sorted eigenvalue fingerprint as row-major
        let sorted_fp = sorted_eigenvalue_fingerprint(&w);
        assert!((sorted_fp - (-0.20985324861994137)).abs() < 1e-8);

        // Verify eigenvector was computed
        let _vr = vr.unwrap();
    }
}

#[cfg(test)]
mod test_ggev {
    use super::*;

    #[test]
    fn test_dggev_simple() {
        // Simple 2x2 generalized eigenvalue problem
        // A = [[1, 2], [3, 4]], B = [[2, 0], [0, 2]]
        // Eigenvalues of A/B = eigenvalues of A/2
        let device = DeviceBLAS::default();
        let data_a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let data_b: Vec<f64> = vec![2.0, 0.0, 0.0, 2.0];
        let a = rt::asarray((data_a.clone(), [2, 2].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((data_b.clone(), [2, 2].c(), &device)).into_dim::<Ix2>();

        let driver = GGEV::default().a(a.view()).b(b.view()).left(false).right(true).build().unwrap();
        let (alpha, beta, _vl, _vr, _a, _b) = driver.run().unwrap();

        // Check sorted eigenvalue fingerprint
        let fp = sorted_ggev_eigenvalue_fingerprint(&alpha, &beta);
        // Expected: eigenvalues are -0.186 and 2.686 (sorted: [-0.186, 2.686])
        assert!((fp - 1.2651873317327877).abs() < 1e-8);
    }

    #[test]
    fn test_dggev_identity_b() {
        // When B = I, GGEV should give same eigenvalues as GEEV
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [64, 64].c(), &device)).into_dim::<Ix2>();

        // Create identity matrix B
        let mut b_data = vec![0.0; 64 * 64];
        for i in 0..64 {
            b_data[i * 64 + i] = 1.0;
        }
        let b = rt::asarray((b_data, [64, 64].c(), &device)).into_dim::<Ix2>();

        // Run GGEV with identity B
        let driver_ggev = GGEV::default().a(a.view()).b(b.view()).left(false).right(true).build().unwrap();
        let (alpha_ggev, beta_ggev, _vl, vr_ggev, _a_out, _b_out) = driver_ggev.run().unwrap();

        // Check sorted eigenvalue fingerprint
        let fp_ggev = sorted_ggev_eigenvalue_fingerprint(&alpha_ggev, &beta_ggev);
        assert!((fp_ggev - (-1.3665882148626807)).abs() < 1e-8);

        // Verify eigenvector was computed
        let _vr = vr_ggev.unwrap();
    }

    #[test]
    fn test_dggev_random_b() {
        // Test GGEV with random B matrix
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [32, 32].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<f64>('b'), [32, 32].c(), &device)).into_dim::<Ix2>();

        // Row-major
        let driver = GGEV::default().a(a.view()).b(b.view()).left(false).right(true).build().unwrap();
        let (alpha, beta, _vl, vr, _a_out, _b_out) = driver.run().unwrap();

        let fp = sorted_ggev_eigenvalue_fingerprint(&alpha, &beta);
        assert!((fp - (-146.02668700723964)).abs() < 1e-8);

        // Verify eigenvector was computed
        let _vr = vr.unwrap();
    }

    #[test]
    fn test_dggev_column_major() {
        // Test GGEV with column-major input
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [32, 32].f(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<f64>('b'), [32, 32].f(), &device)).into_dim::<Ix2>();

        let driver = GGEV::default().a(a.view()).b(b.view()).left(false).right(true).build().unwrap();
        let (alpha, beta, _vl, vr, _a_out, _b_out) = driver.run().unwrap();

        // Same fingerprint as row-major
        let fp = sorted_ggev_eigenvalue_fingerprint(&alpha, &beta);
        assert!((fp - (-146.02668700723964)).abs() < 1e-8);

        // Verify eigenvector was computed
        let _vr = vr.unwrap();
    }

    #[test]
    fn test_dggev_both_eigenvectors() {
        // Test computing both left and right eigenvectors
        let device = DeviceBLAS::default();
        let a = rt::asarray((get_vec::<f64>('a'), [32, 32].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<f64>('b'), [32, 32].c(), &device)).into_dim::<Ix2>();

        let driver = GGEV::default().a(a.view()).b(b.view()).left(true).right(true).build().unwrap();
        let (_alpha, _beta, vl, vr, _a_out, _b_out) = driver.run().unwrap();

        // Verify both eigenvectors were computed
        let vl = vl.unwrap();
        let vr = vr.unwrap();

        // Check eigenvector fingerprints
        assert!((fingerprint(&vl.abs()) - 2.767725200811872).abs() < 1e-8);
        assert!((fingerprint(&vr.abs()) - 6.185473827740987).abs() < 1e-8);
    }
}
