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
