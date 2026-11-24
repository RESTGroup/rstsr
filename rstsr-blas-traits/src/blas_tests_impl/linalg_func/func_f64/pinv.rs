use super::*;

#[test]
fn test_pinv() {
    let device = DeviceBLAS::default();

    // 1024 x 512
    let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
    assert!((fingerprint(&a_pinv) - 0.0878262837784408).abs() < 1e-8);
    assert_eq!(rank, 163);

    // 512 x 1024
    let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();

    let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
    assert!((fingerprint(&a_pinv) - -0.3244041253699862).abs() < 1e-8);
    assert_eq!(rank, 161);
}

#[test]
fn test_pinv_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let m = 4;
    let n = 5;
    let a0 = get_vec::<f64>('a')[..m * n].to_vec();
    let b0: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 - 5.0).collect();
    let slot_data = |k: usize| -> Vec<f64> { (0..m * n).map(|j| a0[j] + 3.0 * k as f64 * b0[j]).collect() };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [3, m, n].c(), &device));

    // pinv of a (4, 5) slot is (5, 4); A P A = A per slot
    let result = rt::linalg::pinv(a.view());
    assert_eq!(result.pinv.shape(), &[3, n, m]);
    assert_eq!(result.rank, m);
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [m, n].c(), &device));
        let slot_pinv = rt::linalg::pinv(&slot);
        assert!(rt::allclose(result.pinv.i(k).into_dim::<Ix2>(), &slot_pinv.pinv, None));
        let apa = &(&slot % result.pinv.i(k).into_dim::<Ix2>()) % &slot;
        assert!(rt::allclose(&apa, &slot, Some(1e-8)));
    }

    // a rank-deficient slot reports the largest per-matrix rank
    let mut rank_def = slot_data(0);
    for j in 0..n {
        rank_def[2 * n + j] = rank_def[j] + rank_def[n + j];
    }
    let stack: Vec<f64> = [slot_data(0), rank_def.clone(), slot_data(2)].concat();
    let a = rt::asarray((stack, [3, m, n].c(), &device));
    let result = rt::linalg::pinv(a.view());
    assert_eq!(result.rank, m);
    let apa = &(&(&a % result.pinv.view()) % &a);
    assert!(rt::allclose(apa, &a, Some(1e-8)));
}
