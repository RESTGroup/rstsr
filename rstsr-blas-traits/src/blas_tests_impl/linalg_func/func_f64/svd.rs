use super::*;
use rstsr::prelude::linalg::SVDResult;

#[test]
fn test_svd() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // default
    let (u, s, vt) = rt::linalg::svd(a.view()).into();
    assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -1.9368850983570982).abs() < 1e-8);
    assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);

    // full_matrices = false
    let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
    assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -9.144981428076894).abs() < 1e-8);
    assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);

    // m < n, full_matrices = false
    let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
    let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
    assert!((fingerprint(&s) - 32.27742168207757).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -3.716931052161584).abs() < 1e-8);
    assert!((fingerprint(&vt.abs()) - -0.32301437281530243).abs() < 1e-8);
}

#[test]
fn test_svd_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let m = 4;
    let n = 5;
    let a0 = get_vec::<f64>('a')[..m * n].to_vec();
    let b0: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 - 5.0).collect();
    let slot_data = |k: usize| -> Vec<f64> { (0..m * n).map(|j| a0[j] + 3.0 * k as f64 * b0[j]).collect() };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [3, m, n].c(), &device));

    // reduced form: u [3, 4, 4], s [3, 4], vt [3, 4, 5]
    let result = rt::linalg::svd((a.view(), false));
    let SVDResult { u, s, vt } = result;
    assert_eq!(u.shape(), &[3, m, m]);
    assert_eq!(s.shape(), &[3, m]);
    assert_eq!(vt.shape(), &[3, m, n]);
    let eye4: Tensor<f64, DeviceBLAS, IxD> = rt::eye((m, &device));
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [m, n].c(), &device));
        let slot_svd = rt::linalg::svd((&slot, false));
        assert!(rt::allclose(u.i(k).into_dim::<Ix2>(), &slot_svd.u, None));
        // A = U diag(S) Vt and U, V orthonormal
        let rebuild = &u.i(k).into_dim::<Ix2>() % rt::diag(&s.i(k)) % vt.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&rebuild, &slot, Some(1e-10)));
        let uut = &u.i(k).into_dim::<Ix2>().t() % u.i(k).into_dim::<Ix2>();
        assert!(rt::allclose(&uut, &eye4, Some(1e-10)));
    }

    // full form: u [3, 4, 4], s [3, 4], vt [3, 5, 5]
    let result = rt::linalg::svd((a.view(), true));
    assert_eq!(result.u.shape(), &[3, m, m]);
    assert_eq!(result.vt.shape(), &[3, n, n]);
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [m, n].c(), &device));
        let slot_svd = rt::linalg::svd((&slot, true));
        assert!(rt::allclose(result.u.i(k).into_dim::<Ix2>(), &slot_svd.u, None));
    }
}
