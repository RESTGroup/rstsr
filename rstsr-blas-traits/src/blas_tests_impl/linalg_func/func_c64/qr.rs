use super::*;
use rstsr::prelude::linalg::QrMode;

/// R with a real-positive diagonal (QR is unique up to a unit factor per column).
fn canonical_r(r: &Tensor<c64, DeviceBLAS, Ix2>) -> Tensor<c64, DeviceBLAS, Ix2> {
    let mut out = r.clone();
    for i in 0..r.nrow().min(r.ncol()) {
        let d = r[[i, i]];
        let s = c64::new(1.0, 0.0) / (d / d.norm());
        out.i_mut((i, i..r.ncol())).assign_f(r.view().i((i, i..r.ncol())) * s).unwrap();
    }
    out
}

#[test]
fn test_qr() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // tall (default reduced and complete): Q (1024, 512), R (512, 512)
    let (q, r) = rt::linalg::qr(&a).into();
    assert_eq!(q.shape(), &[1024, 512]);
    assert_eq!(r.shape(), &[512, 512]);
    assert!((fingerprint(&q.abs()) - -10.064742559299749).abs() < 1e-8);
    assert!((fingerprint(&canonical_r(&r)) - c64::new(0.2403951123793533, 239.69822983660924)).norm() < 1e-8);

    let (qc, rc) = rt::linalg::qr((a.view(), QrMode::Complete)).into();
    assert_eq!(qc.shape(), &[1024, 1024]);
    assert_eq!(rc.shape(), &[1024, 512]);
    assert!((fingerprint(&qc.i((.., 0..512)).abs()) - -10.064742559299749).abs() < 1e-8);
    assert!(
        (fingerprint(&canonical_r(&rc).i((0..512, ..))) - c64::new(0.2403951123793533, 239.69822983660924)).norm()
            < 1e-8
    );

    // wide: Q (512, 512), R (512, 1024)
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
    let (q, r) = rt::linalg::qr((a.view(), "reduced")).into();
    assert_eq!(q.shape(), &[512, 512]);
    assert_eq!(r.shape(), &[512, 1024]);
    assert!((fingerprint(&q.abs()) - -0.7203599314373182).abs() < 1e-8);
    assert!((fingerprint(&canonical_r(&r)) - c64::new(-47.17594962775998, 808.8519384383669)).norm() < 1e-8);

    // A = Q R and Q^H Q = I (unitary: conjugate transpose)
    let qh = q.view().t().conj().into_owned();
    let diff = (q.view() % r.view()) - a.view();
    let ones: Tensor<f64, DeviceBLAS, _> = rt::ones(([512, 1024], &device));
    let scale = (a.view().abs() + &ones).sum_all();
    assert!((diff.abs()).sum_all() / scale < 1e-12, "A != Q R");
    let g = qh.view() % q.view();
    let eye: Tensor<f64, DeviceBLAS, _> = rt::eye((512, &device));
    let d = g.abs() - eye.view();
    assert!((d.abs()).sum_all() < 1e-10, "Q is not unitary");
}
