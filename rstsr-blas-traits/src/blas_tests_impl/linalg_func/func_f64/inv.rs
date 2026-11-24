use super::*;

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
fn test_inv_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    // distinct per-slot data: slot k is (A + 3k I), A well-conditioned
    let n = 6;
    let a0 = get_vec::<f64>('a')[..n * n].to_vec();
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n).map(|j| a0[j] + if j / n == j % n { 3.0 * k as f64 } else { 0.0 }).collect()
    };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [3, n, n].c(), &device));
    let eye: Tensor<f64, DeviceBLAS, IxD> = rt::eye((n, &device));

    // allocating: each slot inverted, with per-slot parity against the 2-D call
    let a_inv = rt::linalg::inv(a.view());
    assert_eq!(a_inv.shape(), &[3, n, n]);
    for k in 0..3 {
        let slot = a.i(k).into_dim::<Ix2>();
        let slot_inv = rt::linalg::inv(&slot);
        assert!(rt::allclose(a_inv.i(k).into_dim::<Ix2>(), &slot_inv, None));
        let prod = &slot % a_inv.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&prod, &eye, Some(1e-10)));
    }

    // in-place: the stack is overwritten with its own inverse, written back
    let mut a = a;
    rt::linalg::inv(a.view_mut());
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [n, n].c(), &device));
        let prod = &slot % a.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&prod, &eye, Some(1e-10)));
    }
}

#[test]
fn test_inv_nd_col_major() {
    // on a col-major device, the matrix axes are the FIRST two shape axes,
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
    let eye: Tensor<f64, DeviceBLAS, IxD> = rt::eye((n, &device));

    // slot k lives at the trailing batch axis: a[(.., .., k)]
    let a_inv = rt::linalg::inv(a.view());
    assert_eq!(a_inv.shape(), &[n, n, 3]);
    for k in 0..3 {
        let slot = a.i((.., .., k)).into_dim::<Ix2>();
        let slot_inv = rt::linalg::inv(&slot);
        assert!(rt::allclose(a_inv.i((.., .., k)).into_dim::<Ix2>(), &slot_inv, None));
        let prod = &slot % a_inv.i((.., .., k)).into_dim::<Ix2>();
        assert!(rt::allclose(&prod, &eye, Some(1e-10)));
    }
}
