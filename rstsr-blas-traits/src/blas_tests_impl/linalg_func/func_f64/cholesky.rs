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
fn test_cholesky_submatrix() {
    let device = DeviceBLAS::default();
    let vec_b: Vec<f64> = vec![0.0, 1.0, 2.0, 1.0, 5.0, 1.5, 2.0, 1.5, 8.0];
    let b = rt::asarray((vec_b, [3, 3].c(), &device));

    let b_view = b.i((1..3, 1..3));
    let c = rt::linalg::cholesky(b_view);
    assert!((fingerprint(&c) - -0.7633202592326889).abs() < 1e-8);
}

#[test]
fn test_cholesky_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    // distinct per-slot SPD matrices: slot k is (Bᵀ B + (1 + k) I) for a
    // fixed well-conditioned B, so each slot has a different factor
    let n = 5;
    let b = get_vec::<f64>('b')[..n * n].to_vec();
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] = (0..n).map(|l| b[l * n + i] * b[l * n + j]).sum();
        }
    }
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n).map(|j| gram[j] + if j / n == j % n { 1.0 + k as f64 } else { 0.0 }).collect()
    };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [3, n, n].c(), &device));

    // allocating: L Lᵀ = A per slot, and per-slot parity with the 2-D call
    let c = rt::linalg::cholesky(a.view());
    assert_eq!(c.shape(), &[3, n, n]);
    for k in 0..3 {
        let slot = a.i(k).into_dim::<Ix2>();
        let slot_c = rt::linalg::cholesky(&slot);
        assert!(rt::allclose(c.i(k).into_dim::<Ix2>(), &slot_c, None));
        let llt = &c.i(k).into_dim::<Ix2>() % c.i(k).into_dim::<Ix2>().t();
        assert!(rt::allclose(&llt, &slot, Some(1e-10)));
    }

    // in-place: each slot is overwritten with its own factor, written back
    let mut a = a;
    rt::linalg::cholesky(a.view_mut());
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [n, n].c(), &device));
        let llt = &a.i(k).into_dim::<Ix2>() % a.i(k).into_dim::<Ix2>().t();
        assert!(rt::allclose(&llt, &slot, Some(1e-10)));
    }
}

#[test]
fn test_cholesky_nd_submatrix_inplace() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    // A is [2, 5, 5] row-major; factor only the (3, 3) trailing submatrix of
    // each slot: A[:, 2:, 2:] is a strided mutable view of shape [2, 3, 3]
    let n = 5;
    let slot = |k: usize| -> Vec<f64> {
        // SPD submatrix B at rows/cols 2..5: diag(4, 5, 6) + offdiag 1
        let mut b = vec![0.0; n * n];
        for i in 2..n {
            for j in 2..n {
                b[i * n + j] = if i == j { 3.0 + i as f64 } else { 1.0 };
            }
        }
        // marker values outside the submatrix, so a stray write is visible
        b[0] = 99.0 + k as f64;
        b[7] = 99.0 + k as f64;
        b
    };
    let stack: Vec<f64> = (0..2).flat_map(slot).collect();
    let mut a = rt::asarray((stack, [2, n, n].c(), &device));

    // in-place cholesky on the strided submatrix view
    let view = a.i_mut((.., 2.., 2..));
    rt::linalg::cholesky(view);
    let expect_a = |slot_idx: usize| -> Vec<f64> {
        // rebuild the original slot, then overwrite the factor region
        let mut b = slot(slot_idx);
        let sub: Vec<f64> = (0..9).map(|j| b[(2 + j / 3) * n + 2 + j % 3]).collect();
        let l = rt::linalg::cholesky(rt::asarray((sub, [3, 3].c(), &device)));
        let l = l.raw().to_vec();
        for i in 0..3 {
            for j in 0..3 {
                b[(2 + i) * n + 2 + j] = l[i * 3 + j];
            }
        }
        b
    };
    for k in 0..2 {
        let want = expect_a(k);
        assert_eq!(a.raw().to_vec()[k * n * n..(k + 1) * n * n], want, "slot {k}");
    }
}
