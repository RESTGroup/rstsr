use super::*;
use rstsr::prelude::linalg::EighResult;

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
fn test_eigh_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let n = 5;
    let a0 = get_vec::<f64>('a')[..n * n].to_vec();
    let mut sym0 = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            sym0[i * n + j] = a0[i * n + j] + a0[j * n + i];
        }
    }
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n).map(|j| sym0[j] + if j / n == j % n { 3.0 * k as f64 } else { 0.0 }).collect()
    };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [3, n, n].c(), &device));
    let eye: Tensor<f64, DeviceBLAS, IxD> = rt::eye((n, &device));

    // allocating: A V = V diag(W), V orthogonal, per-slot parity with 2-D
    let result = rt::linalg::eigh(a.view());
    assert_eq!(result.eigenvalues.shape(), &[3, n]);
    assert_eq!(result.eigenvectors.shape(), &[3, n, n]);
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [n, n].c(), &device));
        let slot_eigh = rt::linalg::eigh(&slot);
        assert!(rt::allclose(result.eigenvalues.i(k).into_dim::<Ix1>(), &slot_eigh.eigenvalues, None));
        let av = &slot % result.eigenvectors.i(k).into_dim::<Ix2>();
        let vd = &result.eigenvectors.i(k).into_dim::<Ix2>() % rt::diag(&result.eigenvalues.i(k));
        assert!(rt::allclose(&av, &vd, Some(1e-8)));
        let vtv = &result.eigenvectors.i(k).into_dim::<Ix2>().t() % result.eigenvectors.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&vtv, &eye, Some(1e-10)));
    }

    // in-place: a is overwritten with the eigenvectors, written back
    let mut a = a;
    let EighResult { eigenvalues: w, eigenvectors: _ } = rt::linalg::eigh(a.view_mut());
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [n, n].c(), &device));
        let av = &slot % a.i(k).into_dim::<Ix2>();
        let vd = &a.i(k).into_dim::<Ix2>() % rt::diag(&w.i(k));
        assert!(rt::allclose(&av, &vd, Some(1e-8)));
    }
}

#[test]
fn test_eigh_nd_col_major() {
    // col-major device: matrix axes leading, batch trailing, shape [n, n, 3]
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    let n = 5;
    let a0 = get_vec::<f64>('a')[..n * n].to_vec();
    let mut sym0 = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            sym0[i * n + j] = a0[i * n + j] + a0[j * n + i];
        }
    }
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n).map(|j| sym0[j] + if j / n == j % n { 3.0 * k as f64 } else { 0.0 }).collect()
    };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [n, n, 3].f(), &device));

    let result = rt::linalg::eigh(a.view());
    assert_eq!(result.eigenvalues.shape(), &[n, 3]);
    assert_eq!(result.eigenvectors.shape(), &[n, n, 3]);
    for k in 0..3 {
        let slot = a.i((.., .., k)).into_dim::<Ix2>();
        let slot_eigh = rt::linalg::eigh(&slot);
        assert!(rt::allclose(result.eigenvalues.i((.., k)).into_dim::<Ix1>(), &slot_eigh.eigenvalues, None));
    }
}

#[test]
fn test_eigh_nd_generalized() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let n = 5;
    let a0 = get_vec::<f64>('a')[..n * n].to_vec();
    let b0 = get_vec::<f64>('b')[..n * n].to_vec();
    let mut sym0 = vec![0.0; n * n];
    let mut pd0 = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            sym0[i * n + j] = a0[i * n + j] + a0[j * n + i];
            pd0[i * n + j] = b0[i * n + j] + b0[j * n + i];
        }
        pd0[i * n + i] += n as f64;
    }
    let slot_a = |k: usize| -> Vec<f64> {
        (0..n * n).map(|j| sym0[j] + if j / n == j % n { 3.0 * k as f64 } else { 0.0 }).collect()
    };
    let slot_b =
        |k: usize| -> Vec<f64> { (0..n * n).map(|j| pd0[j] + if j / n == j % n { k as f64 } else { 0.0 }).collect() };
    let stack_a: Vec<f64> = (0..3).flat_map(slot_a).collect();
    let stack_b: Vec<f64> = (0..3).flat_map(slot_b).collect();
    let a = rt::asarray((stack_a, [3, n, n].c(), &device));
    let b = rt::asarray((stack_b, [3, n, n].c(), &device));
    let eye: Tensor<f64, DeviceBLAS, IxD> = rt::eye((n, &device));

    // same batch: per-slot parity with the 2-D entry, A V = lambda B V, B-orthonormal V
    let result = rt::linalg::eigh((a.view(), b.view()));
    assert_eq!(result.eigenvalues.shape(), &[3, n]);
    assert_eq!(result.eigenvectors.shape(), &[3, n, n]);
    for k in 0..3 {
        let sa = rt::asarray((slot_a(k), [n, n].c(), &device));
        let sb = rt::asarray((slot_b(k), [n, n].c(), &device));
        let slot_eigh = rt::linalg::eigh((&sa, &sb));
        assert!(rt::allclose(result.eigenvalues.i(k).into_dim::<Ix1>(), &slot_eigh.eigenvalues, None));
        let v = result.eigenvectors.i(k).into_dim::<Ix2>();
        let lam = result.eigenvalues.i(k).into_dim::<Ix1>();
        let av = &sa % &v;
        let bv = &sb % &v;
        let lbv = &bv % rt::diag(&lam);
        assert!(rt::allclose(&av, &lbv, Some(1e-8)));
        let vbv = v.t() % &bv;
        assert!(rt::allclose(&vbv, &eye, Some(1e-10)));
    }

    // broadcast: b is a single matrix, batch (1) grows to (3)
    let b_single = rt::asarray((slot_b(0), [1, n, n].c(), &device));
    let result = rt::linalg::eigh((a.view(), b_single.view()));
    assert_eq!(result.eigenvalues.shape(), &[3, n]);
    for k in 0..3 {
        let sa = rt::asarray((slot_a(k), [n, n].c(), &device));
        let sb = rt::asarray((slot_b(0), [n, n].c(), &device));
        let slot_eigh = rt::linalg::eigh((&sa, &sb));
        assert!(rt::allclose(result.eigenvalues.i(k).into_dim::<Ix1>(), &slot_eigh.eigenvalues, None));
    }

    // itype = 3: A B x = lambda x
    let result3 = rt::linalg::eigh((a.view(), b.view(), Upper, 3));
    for k in 0..3 {
        let sa = rt::asarray((slot_a(k), [n, n].c(), &device));
        let sb = rt::asarray((slot_b(k), [n, n].c(), &device));
        let slot_eigh = rt::linalg::eigh((&sa, &sb, Upper, 3));
        assert!(rt::allclose(result3.eigenvalues.i(k).into_dim::<Ix1>(), &slot_eigh.eigenvalues, None));
    }
}
