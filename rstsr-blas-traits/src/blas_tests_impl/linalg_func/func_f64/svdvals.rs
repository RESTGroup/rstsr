use super::*;

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

#[test]
fn test_svdvals_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let m = 4;
    let n = 5;
    let a0 = get_vec::<f64>('a')[..m * n].to_vec();
    // rectangular stack (3, 4, 5): each slot is (A + 3k B) with a fixed
    // distinct B, so singular values differ per slot
    let b0: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 - 5.0).collect();
    let slot_data = |k: usize| -> Vec<f64> { (0..m * n).map(|j| a0[j] + 3.0 * k as f64 * b0[j]).collect() };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [3, m, n].c(), &device));

    let s = rt::linalg::svdvals(a.view());
    assert_eq!(s.shape(), &[3, m]);
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [m, n].c(), &device));
        let s2 = rt::linalg::svdvals(&slot);
        assert!(rt::allclose(s.i(k).into_dim::<Ix1>(), &s2, None));
    }
}

#[test]
fn test_svdvals_nd_col_major() {
    // col-major device: matrix axes leading, batch trailing, shape [4, 5, 3]
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    let m = 4;
    let n = 5;
    let a0 = get_vec::<f64>('a')[..m * n].to_vec();
    let b0: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 - 5.0).collect();
    let slot_data = |k: usize| -> Vec<f64> { (0..m * n).map(|j| a0[j] + 3.0 * k as f64 * b0[j]).collect() };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [m, n, 3].f(), &device));

    let s = rt::linalg::svdvals(a.view());
    assert_eq!(s.shape(), &[m, 3]);
    for k in 0..3 {
        let slot = a.i((.., .., k)).into_dim::<Ix2>();
        let s2 = rt::linalg::svdvals(&slot);
        assert!(rt::allclose(s.i((.., k)).into_dim::<Ix1>(), &s2, None));
    }
}
