use super::*;

#[test]
fn test_det() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..5 * 5].to_vec();
    let mut a = rt::asarray((a_vec, [5, 5].c(), &device));

    let det = rt::linalg::det(a.view_mut());
    assert!((det.to_scalar() - 3.9699917597338046).abs() < 1e-8);
}

#[test]
fn test_det_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let n = 5;
    let a0 = get_vec::<f64>('a')[..n * n].to_vec();
    // slot k is (A + k I); det scales with the diagonal shift
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n).map(|j| a0[j] + if j / n == j % n { 3.0 * k as f64 } else { 0.0 }).collect()
    };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [3, n, n].c(), &device));

    // det of a stack is batch-shaped; per-slot parity with the 2-D call
    let dets = rt::linalg::det(a.view());
    assert_eq!(dets.shape(), &[3]);
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [n, n].c(), &device));
        let d2 = rt::linalg::det(&slot);
        assert!((dets.i(k).to_scalar() - d2.to_scalar()).abs() < 1e-8);
    }

    // the mutable form may destroy the input but still yields the same dets
    let mut a = a;
    let dets = rt::linalg::det(a.view_mut());
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [n, n].c(), &device));
        let d2 = rt::linalg::det(&slot);
        assert!((dets.i(k).to_scalar() - d2.to_scalar()).abs() < 1e-8);
    }
}

#[test]
fn test_det_nd_col_major() {
    // on a col-major device the matrix axes are the first two shape axes,
    // so a stack of 3 (n, n) matrices is shaped [n, n, 3] with batch trailing
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    let n = 5;
    let a0 = get_vec::<f64>('a')[..n * n].to_vec();
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n).map(|j| a0[j] + if j / n == j % n { 3.0 * k as f64 } else { 0.0 }).collect()
    };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [n, n, 3].f(), &device));

    let dets = rt::linalg::det(a.view());
    assert_eq!(dets.shape(), &[3]);
    for k in 0..3 {
        let slot = a.i((.., .., k)).into_dim::<Ix2>();
        let d2 = rt::linalg::det(&slot);
        assert!((dets.i(k).to_scalar() - d2.to_scalar()).abs() < 1e-8);
    }
}
