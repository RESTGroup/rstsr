use super::*;

#[test]
fn test_slogdet() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));

    let (sign, logabsdet) = rt::linalg::slogdet(a.view()).into();
    assert!(sign.to_scalar() - -1.0 < 1e-8);
    assert!(logabsdet.to_scalar() - 3031.1259211802403 < 1e-8);
}

#[test]
fn test_slogdet_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let n = 5;
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n)
            .map(|j| {
                let (i, jj) = (j / n, j % n);
                if i == jj {
                    (k as f64) - 1.5 // slot 0: negative det, slots 1-2: positive
                } else if i > jj {
                    1.0 + ((i * n + jj) % 3) as f64 * 0.5 // lower-triangular: det = product of
                                                          // diagonal
                } else {
                    0.0
                }
            })
            .collect()
    };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [3, n, n].c(), &device));

    let (sign, logabsdet) = rt::linalg::slogdet(a.view()).into();
    assert_eq!(sign.shape(), &[3]);
    assert_eq!(logabsdet.shape(), &[3]);
    for k in 0..3 {
        let slot = rt::asarray((slot_data(k), [n, n].c(), &device));
        let (s_ref, l_ref) = rt::linalg::slogdet(&slot).into();
        assert!((sign.i(k).to_scalar() - s_ref.to_scalar()).abs() < 1e-10);
        assert!((logabsdet.i(k).to_scalar() - l_ref.to_scalar()).abs() < 1e-10);
    }
    // triangular slots: sign = sign(diagonal product) = (-1)^n for slots 0-1
    assert!(sign.i(0).to_scalar() < 0.0);
    assert!(sign.i(1).to_scalar() < 0.0);
    assert!(sign.i(2).to_scalar() > 0.0);
}

#[test]
fn test_slogdet_nd_col_major() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    let n = 5;
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n)
            .map(|j| {
                let (i, jj) = (j / n, j % n);
                if i == jj {
                    2.0 + k as f64
                } else {
                    ((i * 2 + jj) % 5) as f64 * 0.25
                }
            })
            .collect()
    };
    let stack: Vec<f64> = (0..3).flat_map(slot_data).collect();
    let a = rt::asarray((stack, [n, n, 3].f(), &device));

    let (sign, logabsdet) = rt::linalg::slogdet(a.view()).into();
    assert_eq!(sign.shape(), &[3]);
    for k in 0..3 {
        let slot = a.i((.., .., k)).into_dim::<Ix2>();
        let (s_ref, l_ref) = rt::linalg::slogdet(&slot).into();
        assert!((sign.i(k).to_scalar() - s_ref.to_scalar()).abs() < 1e-10);
        assert!((logabsdet.i(k).to_scalar() - l_ref.to_scalar()).abs() < 1e-10);
    }
}
