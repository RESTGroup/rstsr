use super::*;

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
fn test_solve_symmetric_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let n = 5;
    // symmetric positive-definite slots: slot k is S + (k+3) I
    let slot_a = |k: usize| -> Vec<f64> {
        let raw = get_vec::<f64>('a')[..n * n].to_vec();
        (0..n * n)
            .map(|j| {
                let (i, jj) = (j / n, j % n);
                let s = raw[i * n + jj] + raw[jj * n + i];
                if i == jj {
                    s + (k + 3) as f64
                } else {
                    s
                }
            })
            .collect()
    };
    let slot_b = |k: usize| -> Vec<f64> { get_vec::<f64>('b')[k * n * n..(k + 1) * n * n].to_vec() };
    let stack_a: Vec<f64> = (0..2).flat_map(slot_a).collect();
    let stack_b: Vec<f64> = (0..2).flat_map(slot_b).collect();
    let a = rt::asarray((stack_a, [2, n, n].c(), &device));
    let b = rt::asarray((stack_b, [2, n, n].c(), &device));
    let eye: Tensor<f64, DeviceBLAS, IxD> = rt::eye((n, &device));

    // allocating: per-slot parity with 2-D and A X = B
    let x = rt::linalg::solve_symmetric((a.view(), b.view()));
    assert_eq!(x.shape(), &[2, n, n]);
    for k in 0..2 {
        let sa = rt::asarray((slot_a(k), [n, n].c(), &device));
        let sb = rt::asarray((slot_b(k), [n, n].c(), &device));
        let slot_x = rt::linalg::solve_symmetric((&sa, &sb));
        assert!(rt::allclose(x.i(k).into_dim::<Ix2>(), &slot_x, None));
        let ax = &sa % x.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&ax, &sb, Some(1e-8)));
    }

    // broadcast: a is a single matrix, batch (1) grows to (2)
    let a_single = rt::asarray((slot_a(0), [1, n, n].c(), &device));
    let x_bcast = rt::linalg::solve_symmetric((a_single.view(), b.view()));
    assert_eq!(x_bcast.shape(), &[2, n, n]);
    for k in 0..2 {
        let sb = rt::asarray((slot_b(k), [n, n].c(), &device));
        let ax = &a_single.i((0, .., ..)).into_dim::<Ix2>() % x_bcast.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&ax, &sb, Some(1e-8)));
    }

    // batched-vector form: b of shape (2, n)
    let b_vec = rt::asarray((get_vec::<f64>('b')[..2 * n].to_vec(), [2, n].c(), &device));
    let x_vec = rt::linalg::solve_symmetric((a.view(), b_vec.view()));
    assert_eq!(x_vec.shape(), &[2, n]);
    for k in 0..2 {
        let sa = rt::asarray((slot_a(k), [n, n].c(), &device));
        let bv = b_vec.i(k).into_dim::<Ix1>();
        let xv = &sa % x_vec.i(k).into_dim::<Ix1>();
        assert!(rt::allclose(&xv, &bv, Some(1e-8)));
    }

    // in-place b: write-back into the original buffer
    let mut b_inplace = b.view().to_owned();
    rt::linalg::solve_symmetric((a.view(), b_inplace.view_mut()));
    for k in 0..2 {
        let slot_x = x.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(b_inplace.i(k).into_dim::<Ix2>(), &slot_x, Some(1e-10)));
    }

    // in-place a and b: a is destroyed, b holds the solution
    let mut a_inplace = a;
    let mut b_inplace2 = b.view().to_owned();
    rt::linalg::solve_symmetric((a_inplace.view_mut(), b_inplace2.view_mut()));
    for k in 0..2 {
        let slot_x = x.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(b_inplace2.i(k).into_dim::<Ix2>(), &slot_x, Some(1e-10)));
    }
    let _ = eye;
}

#[test]
fn test_solve_symmetric_nd_col_major() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    let n = 5;
    let slot_a = |k: usize| -> Vec<f64> {
        let raw = get_vec::<f64>('a')[..n * n].to_vec();
        (0..n * n)
            .map(|j| {
                let (i, jj) = (j / n, j % n);
                let s = raw[i * n + jj] + raw[jj * n + i];
                if i == jj {
                    s + (k + 3) as f64
                } else {
                    s
                }
            })
            .collect()
    };
    let slot_b = |k: usize| -> Vec<f64> { get_vec::<f64>('b')[k * n * n..(k + 1) * n * n].to_vec() };
    let stack_a: Vec<f64> = (0..2).flat_map(slot_a).collect();
    let stack_b: Vec<f64> = (0..2).flat_map(slot_b).collect();
    let a = rt::asarray((stack_a, [n, n, 2].f(), &device));
    let b = rt::asarray((stack_b, [n, n, 2].f(), &device));

    let x = rt::linalg::solve_symmetric((a.view(), b.view()));
    assert_eq!(x.shape(), &[n, n, 2]);
    for k in 0..2 {
        // note: slot_a is symmetric (layout-interpretation invariant), but
        // slot_b is not: references must use the same f-interpretation
        let sa = rt::asarray((slot_a(k), [n, n].f(), &device));
        let sb = rt::asarray((slot_b(k), [n, n].f(), &device));
        let slot_x = rt::linalg::solve_symmetric((&sa, &sb));
        assert!(rt::allclose(x.i((.., .., k)).into_dim::<Ix2>(), &slot_x, None));
        let ax = &sa % x.i((.., .., k)).into_dim::<Ix2>();
        assert!(rt::allclose(&ax, &sb, Some(1e-8)));
    }
}
