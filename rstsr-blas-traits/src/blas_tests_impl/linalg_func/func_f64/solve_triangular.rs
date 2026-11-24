use super::*;

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
fn test_solve_triangular_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let n = 5;
    // lower-triangular slots: slot k is L + k I with unit-free diagonal
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    if j < i {
                        1.0 + ((i * n + j) % 5) as f64
                    } else if i == j {
                        3.0 + k as f64
                    } else {
                        0.0
                    }
                })
            })
            .collect()
    };
    let stack_a: Vec<f64> = (0..2).flat_map(slot_data).collect();
    let b0 = get_vec::<f64>('b')[..n * 2].to_vec();
    let stack_b: Vec<f64> = (0..2).flat_map(|_| b0.iter().copied()).collect();
    let a = rt::asarray((stack_a, [2, n, n].c(), &device));
    let b = rt::asarray((stack_b, [2, n, 2].c(), &device));

    // allocating: x [2, 5, 2]; A x = b per slot
    let x = rt::linalg::solve_triangular((a.view(), b.view()));
    assert_eq!(x.shape(), &[2, n, 2]);
    for k in 0..2 {
        let slot = a.i(k).into_dim::<Ix2>();
        let slot_b = b.i(k).into_dim::<Ix2>();
        let ax = &slot % x.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&ax, &slot_b, Some(1e-8)));
    }

    // in-place: b is overwritten with the solution
    let mut b = b;
    rt::linalg::solve_triangular((a.view(), b.view_mut()));
    for k in 0..2 {
        let slot = a.i(k).into_dim::<Ix2>();
        let ax = &slot % b.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&ax, rt::asarray((b0.clone(), [n, 2].c(), &device)), Some(1e-8)));
    }
}

#[test]
fn test_solve_triangular_nd_col_major() {
    // col-major device: matrix axes leading, batch trailing
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    let n = 5;
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    if j < i {
                        1.0 + ((i * n + j) % 5) as f64
                    } else if i == j {
                        3.0 + k as f64
                    } else {
                        0.0
                    }
                })
            })
            .collect()
    };
    let stack_a: Vec<f64> = (0..2).flat_map(slot_data).collect();
    let b0 = get_vec::<f64>('b')[..n * 2].to_vec();
    let stack_b: Vec<f64> = (0..2).flat_map(|_| b0.iter().copied()).collect();
    let a = rt::asarray((stack_a, [n, n, 2].f(), &device));
    let b = rt::asarray((stack_b, [n, 2, 2].f(), &device));

    let x = rt::linalg::solve_triangular((a.view(), b.view()));
    assert_eq!(x.shape(), &[n, 2, 2]);
    for k in 0..2 {
        let slot = a.i((.., .., k)).into_dim::<Ix2>();
        let ax = &slot % x.i((.., .., k)).into_dim::<Ix2>();
        assert!(rt::allclose(&ax, b.i((.., .., k)).into_dim::<Ix2>(), Some(1e-8)));
    }
}
