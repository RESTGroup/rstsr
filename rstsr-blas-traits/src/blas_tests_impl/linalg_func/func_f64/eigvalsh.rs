use super::*;

#[test]
fn test_eigvalsh() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));
    let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device));

    // default, a
    let w = rt::linalg::eigvalsh(a.view());
    assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);

    // upper, a
    let w = rt::linalg::eigvalsh((a.view(), Upper));
    assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);

    // default, a b
    let w = rt::linalg::eigvalsh((a.view(), b.view()));
    assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);

    // upper, a b, itype=3
    let w = rt::linalg::eigvalsh((a.view(), b.view(), Upper, 3));
    assert!((fingerprint(&w) - -2503.84161931662).abs() < 1e-8);

    // mutable changes a
    let w = rt::linalg::eigvalsh(a.view_mut());
    assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
}

#[test]
fn test_eigvalsh_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let n = 5;
    let a0 = get_vec::<f64>('a')[..n * n].to_vec();
    // slot k is symmetrized (A + k I): distinct spectra per slot
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

    // allocating: eigenvalues are [3, n]; per-slot parity with the 2-D call,
    // and each slot satisfies A v = w v round-trip via inv-free check
    let w = rt::linalg::eigvalsh(a.view());
    assert_eq!(w.shape(), &[3, n]);
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [n, n].c(), &device));
        let w2 = rt::linalg::eigvalsh(&slot);
        assert!(rt::allclose(w.i(k).into_dim::<Ix1>(), &w2, None));
    }

    // in-place: a is destroyed, eigenvalues unchanged
    let mut a = a;
    let w = rt::linalg::eigvalsh(a.view_mut());
    assert_eq!(w.shape(), &[3, n]);
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [n, n].c(), &device));
        let w2 = rt::linalg::eigvalsh(&slot);
        assert!(rt::allclose(w.i(k).into_dim::<Ix1>(), &w2, None));
    }
}

#[test]
fn test_eigvalsh_nd_col_major() {
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

    let w = rt::linalg::eigvalsh(a.view());
    assert_eq!(w.shape(), &[n, 3]);
    for k in 0..3 {
        let slot = a.i((.., .., k)).into_dim::<Ix2>();
        let w2 = rt::linalg::eigvalsh(&slot);
        assert!(rt::allclose(w.i((.., k)).into_dim::<Ix1>(), &w2, None));
    }
}
