use super::*;
use rstsr::prelude::linalg::{QrMode, QrResult};

/// R with a real-positive diagonal (QR is unique up to a sign per column).
fn canonical_r(r: &Tensor<f64, DeviceBLAS, Ix2>) -> Tensor<f64, DeviceBLAS, Ix2> {
    let mut out = r.clone();
    for i in 0..r.nrow().min(r.ncol()) {
        let s = if r[[i, i]] < 0.0 { -1.0 } else { 1.0 };
        out.i_mut((i, i..r.ncol())).assign_f(r.view().i((i, i..r.ncol())) * s).unwrap();
    }
    out
}

#[test]
fn test_qr() {
    let device = DeviceBLAS::default();
    let a_vec = &get_vec::<f64>('a')[..1024 * 512];
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // tall, reduced (default): Q (1024, 512), R (512, 512)
    let (q, r) = rt::linalg::qr(&a).into();
    assert_eq!(q.shape(), &[1024, 512]);
    assert_eq!(r.shape(), &[512, 512]);
    assert!((fingerprint(&q.abs()) - -3.816695392252389).abs() < 1e-8);
    assert!((fingerprint(&canonical_r(&r)) - -245.99004994559718).abs() < 1e-8);

    // tall, complete: Q (1024, 1024); thin part and R unchanged
    let (qc, rc) = rt::linalg::qr((a.view(), QrMode::Complete)).into();
    assert_eq!(qc.shape(), &[1024, 1024]);
    assert_eq!(rc.shape(), &[1024, 512]);
    assert!((fingerprint(&qc.i((.., 0..512)).abs()) - -3.816695392252389).abs() < 1e-8);
    let rc = canonical_r(&rc);
    assert!((fingerprint(&rc.i((0..512, ..))) - -245.99004994559718).abs() < 1e-8);

    // wide, reduced: Q (512, 512), R (512, 1024)
    let a_vec = &get_vec::<f64>('a')[..1024 * 512];
    let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
    let (q, r) = rt::linalg::qr((a.view(), "reduced")).into();
    assert_eq!(q.shape(), &[512, 512]);
    assert_eq!(r.shape(), &[512, 1024]);
    assert!((fingerprint(&q.abs()) - -0.9769304656721726).abs() < 1e-8);
    assert!((fingerprint(&canonical_r(&r)) - 405.0109813743518).abs() < 1e-8);

    // wide, complete equals reduced (Q (512, 512), R (512, 1024))
    let (q, r) = rt::linalg::qr((a.view(), "complete")).into();
    assert_eq!(q.shape(), &[512, 512]);
    assert_eq!(r.shape(), &[512, 1024]);
    assert!((fingerprint(&q.abs()) - -0.9769304656721726).abs() < 1e-8);
    assert!((fingerprint(&canonical_r(&r)) - 405.0109813743518).abs() < 1e-8);
}

/// Self-validating checks: shapes, `A = Q R`, `QᴴQ = I`, `tril(R, -1) = 0`.
fn assert_qr_correct(a: &Tensor<f64, DeviceBLAS, Ix2>, mode: QrMode) {
    let (m, n) = (a.nrow(), a.ncol());
    let k = m.min(n);
    let QrResult { q, r } = rt::linalg::qr((a.view(), mode));
    let (qc, r_rows) = match mode {
        QrMode::Complete => (m, m),
        _ => (k, k),
    };
    assert_eq!(q.shape(), &[m, qc]);
    assert_eq!(r.shape(), &[r_rows, n]);

    // A = Q[:, :k] R[:k, :]
    let q_k = q.i((.., 0..k)).into_dim::<Ix2>().into_owned();
    let r_k = r.i((0..k, ..)).into_dim::<Ix2>().into_owned();
    let diff = (q_k.view() % r_k.view()) - a.view();
    let ones: Tensor<f64, DeviceBLAS, _> = rt::ones((a.shape().to_vec(), a.device()));
    let scale = (a.view().abs() + &ones).sum_all();
    assert!(((diff.abs()).sum_all() / scale) < 1e-12, "A != Q R");

    // Q^H Q = I (conjugate transpose for complex)
    let qh = q.view().t().conj().into_owned();
    let g = qh.view() % q.view();
    let eye: Tensor<f64, _, Ix2> = rt::eye((qc, a.device())).into_dim::<Ix2>();
    let d = g.abs() - eye.view();
    assert!((d.abs()).sum_all() < 1e-10, "Q is not orthonormal");

    // lower triangle of R is zero
    let rl = rt::tril((r.view(), -1));
    assert!((rl.abs()).sum_all() == 0.0, "R is not upper triangular");
}

#[test]
fn test_qr_layout_and_stack() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let a0 = get_vec::<f64>('b')[..7 * 5].to_vec();

    // 2-D: both orders of the input, strided view
    let a = rt::asarray((a0.clone(), [7, 5].c(), &device)).into_dim::<Ix2>();
    assert_qr_correct(&a, QrMode::Reduced);
    assert_qr_correct(&a, QrMode::Complete);
    let af = rt::asarray((a0.clone(), [7, 5].f(), &device)).into_dim::<Ix2>();
    assert_qr_correct(&af, QrMode::Reduced);
    let buf = rt::asarray((get_vec::<f64>('b')[..9 * 7].to_vec(), [9, 7].c(), &device)).into_dim::<Ix2>();
    let strided: Tensor<f64, _, Ix2> =
        buf.i((slice!(None, None, 1), SliceI::new(Some(0), Some(5), 2))).into_dim::<Ix2>().into_owned();
    assert_qr_correct(&strided, QrMode::Complete);

    // wide and square
    let aw = rt::asarray((get_vec::<f64>('b')[..5 * 8].to_vec(), [5, 8].c(), &device)).into_dim::<Ix2>();
    assert_qr_correct(&aw, QrMode::Reduced);
    assert_qr_correct(&aw, QrMode::Complete);

    // 3-D stack (row-major: matrix axes last): per-slot parity with the 2-D call
    let (m, n) = (4usize, 3usize);
    let b0: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 - 5.0).collect();
    let slot_data = |kk: usize| -> Vec<f64> { (0..m * n).map(|j| a0[j % (m * n)] + 3.0 * kk as f64 * b0[j]).collect() };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [3, m, n].c(), &device));
    for mode in [QrMode::Reduced, QrMode::Complete] {
        let result = rt::linalg::qr((a.view(), mode));
        let k = m.min(n);
        assert_eq!(result.q.shape(), &[3, m, if matches!(mode, QrMode::Complete) { m } else { k }]);
        assert_eq!(result.r.shape(), &[3, if matches!(mode, QrMode::Complete) { m } else { k }, n]);
        for kk in 0..3 {
            let slot = rt::asarray((slot_data(kk), [m, n].c(), &device)).into_dim::<Ix2>();
            let slot_qr = rt::linalg::qr((&slot, mode));
            assert!(rt::allclose(result.q.i(kk).into_dim::<Ix2>(), &slot_qr.q, None));
            assert!(rt::allclose(result.r.i(kk).into_dim::<Ix2>(), &slot_qr.r, None));
        }
    }
}

#[test]
fn test_qr_owned_inputs() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let v = get_vec::<f64>('b')[..6 * 4].to_vec();

    // owned: consumed
    let a = rt::asarray((v.clone(), [6, 4].c(), &device)).into_dim::<Ix2>();
    let (q, r): (Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>) = rt::linalg::qr(a.view()).into();
    assert_eq!(q.shape(), &[6, 4]);
    assert_eq!(r.shape(), &[4, 4]);

    // mutable: consumed as scratch (destroyed), same result
    let a_ref = rt::asarray((v.clone(), [6, 4].c(), &device)).into_dim::<Ix2>();
    let (q_ref, r_ref) = rt::linalg::qr((a_ref.view(), QrMode::Complete)).into();
    let mut a = rt::asarray((v, [6, 4].c(), &device)).into_dim::<Ix2>();
    let snapshot = a.clone();
    let (q, r) = rt::linalg::qr((a.view_mut(), QrMode::Complete)).into();
    assert!(rt::allclose(&q, &q_ref, None));
    assert!(rt::allclose(&r, &r_ref, None));
    let _ = snapshot;
}

#[test]
fn test_qr_zero_size() {
    let device = DeviceBLAS::default();
    for (m, n) in [(0usize, 4usize), (4, 0), (0, 0)] {
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([m, n], &device)).into_dim::<Ix2>();
        for mode in [QrMode::Reduced, QrMode::Complete] {
            let (q, r) = rt::linalg::qr((a.view(), mode)).into();
            let k = m.min(n);
            let qc = if matches!(mode, QrMode::Complete) { m } else { k };
            let rr = if matches!(mode, QrMode::Complete) { m } else { k };
            assert_eq!(q.shape(), &[m, qc]);
            assert_eq!(r.shape(), &[rr, n]);
            // numpy: Q is the identity completion when only columns are missing
            if qc > 0 && k == 0 {
                let eye: Tensor<f64, DeviceBLAS, _> = rt::eye((m, &device));
                assert!(rt::allclose(&q, &eye, None));
            }
        }
    }
}

#[test]
fn test_qr_ndim_error() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((vec![1.0, 2.0, 3.0], [3], &device));
    assert!(rt::linalg::qr_f(&a).is_err());
}
