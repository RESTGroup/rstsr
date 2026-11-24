use super::*;

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
fn test_solve_general_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let n = 5;
    let a0 = get_vec::<f64>('a')[..n * n].to_vec();
    let b0 = get_vec::<f64>('b')[..n * 3].to_vec();
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n).map(|j| a0[j] + if j / n == j % n { 3.0 * k as f64 } else { 0.0 }).collect()
    };
    let stack_a: Vec<f64> = (0..2).flat_map(slot_data).collect();
    let stack_b: Vec<f64> = (0..2).flat_map(|_| b0.iter().copied()).collect();
    let a = rt::asarray((stack_a, [2, n, n].c(), &device));
    let b = rt::asarray((stack_b, [2, n, 3].c(), &device));

    // allocating: x [2, 5, 3]; per-slot parity and A x = b
    let x = rt::linalg::solve_general((a.view(), b.view()));
    assert_eq!(x.shape(), &[2, n, 3]);
    for k in 0..2 {
        let slot = a.i(k).into_dim::<Ix2>();
        let slot_b = b.i(k).into_dim::<Ix2>();
        let slot_x = rt::linalg::solve_general((&slot, &slot_b));
        assert!(rt::allclose(x.i(k).into_dim::<Ix2>(), &slot_x, None));
        let ax = &slot % x.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&ax, &slot_b, Some(1e-8)));
    }

    // broadcast: b with a singleton batch axis broadcasts to x [2, 5, 3]
    let b_single = rt::asarray((b0.clone(), [1, n, 3].c(), &device));
    let x = rt::linalg::solve_general((a.view(), b_single.view()));
    assert_eq!(x.shape(), &[2, n, 3]);
    for k in 0..2 {
        let slot = a.i(k).into_dim::<Ix2>();
        let ax = &slot % x.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&ax, b_single.i(0).into_dim::<Ix2>(), Some(1e-8)));
    }

    // batched vector form: b [2, 5] solves to x [2, 5]
    let b_vec = b.i((.., .., 0)).into_dim::<IxD>();
    let x = rt::linalg::solve_general((a.view(), b_vec.view()));
    assert_eq!(x.shape(), &[2, n]);
    for k in 0..2 {
        let slot = a.i(k).into_dim::<Ix2>();
        let ax = &slot % x.i(k).into_dim::<Ix1>();
        assert!(rt::allclose(&ax, b_vec.i(k).into_dim::<Ix1>(), Some(1e-8)));
    }

    // in-place: b is overwritten with the solution, written back
    let mut b = b;
    rt::linalg::solve_general((a.view(), b.view_mut()));
    for k in 0..2 {
        let slot = a.i(k).into_dim::<Ix2>();
        let ax = &slot % b.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&ax, rt::asarray((b0.clone(), [n, 3].c(), &device)), Some(1e-8)));
    }

    // in-place cannot grow: a batched a with an unbatched mut-b is rejected
    let mut b_single_mut = b_single.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rt::linalg::solve_general((a.view(), b_single_mut.view_mut()));
    }));
    assert!(result.is_err());
}
