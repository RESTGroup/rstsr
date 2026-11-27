use super::*;

// NumPy golden values from `func_validation_f64.py` (`# ### matrix_power`);
// they equal the constants of the faer entry's tests (same manifest data)
#[test]
fn test_matrix_power() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..16].to_vec();
    let a = rt::asarray((a_vec, [4, 4].c(), &device));

    // n = 0 -> identity, in the input dtype
    let p = rt::linalg::matrix_power((a.view(), 0));
    assert!((fingerprint(&p) - -0.3150972564720475).abs() < 1e-8);

    // n = 1 -> a itself
    let p = rt::linalg::matrix_power((a.view(), 1));
    assert!((fingerprint(&p) - -0.5537625095088874).abs() < 1e-8);

    // shortcuts (2, 3) and the squaring loop (> 3)
    for (n, golden) in [(2isize, -3.3066512682104), (3, 4.312505984257974), (5, -2.904867370507913)] {
        let p = rt::linalg::matrix_power((a.view(), n));
        assert!((fingerprint(&p) - golden).abs() < 1e-8, "n = {n}");
    }

    // negative exponent: invert once, then raise the inverse to |n|
    for (n, golden) in [(-1isize, 1.5318824349546132), (-2, -4.11401929372348)] {
        let p = rt::linalg::matrix_power((a.view(), n));
        assert!((fingerprint(&p) - golden).abs() < 1e-8, "n = {n}");
    }
}

// the full-size fingerprints: n = 2 one GEMM, n = -1 one GETRF + GETRI (whose
// value is the `inv` test's constant), n = 0 the identity, n = 1 the input
#[test]
fn test_matrix_power_fingerprint() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..1024 * 1024].to_vec();
    let a = rt::asarray((a_vec, [1024, 1024].c(), &device)).into_dim::<Ix2>();

    let p1 = rt::linalg::matrix_power((a.view(), 1));
    assert!((fingerprint(&p1) - 191.28900005103065).abs() < 1e-6);
    let p0 = rt::linalg::matrix_power((a.view(), 0));
    assert!((fingerprint(&p0) - 0.20548286571538066).abs() < 1e-8);
    let p2 = rt::linalg::matrix_power((a.view(), 2));
    assert!((fingerprint(&p2) - 12608.641272488418).abs() < 1e-6);
    let pm1 = rt::linalg::matrix_power((a.view(), -1));
    assert!((fingerprint(&pm1) - 143.39005577037764).abs() < 1e-8);
}

#[test]
fn test_matrix_power_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let a_vec = get_vec::<f64>('a')[..16].to_vec();

    // stack: slice 0 is a, slice 1 is 2*a
    let mut stacked = a_vec.clone();
    stacked.extend(a_vec.iter().map(|x| 2.0 * x));
    let b = rt::asarray((stacked, [2, 4, 4].c(), &device));

    let p = rt::linalg::matrix_power((b.view(), 0));
    assert_eq!(p.shape(), &[2, 4, 4]);
    assert!((fingerprint(&p) - -0.2588243168865289).abs() < 1e-8);

    let p = rt::linalg::matrix_power((b.view(), 2));
    assert!((fingerprint(&p) - 1.5180278106564264).abs() < 1e-8);

    let p = rt::linalg::matrix_power((b.view(), -1));
    assert!((fingerprint(&p) - 0.5342839334664118).abs() < 1e-8);
}

#[test]
fn test_matrix_power_nd_colmajor() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    let a_vec = get_vec::<f64>('a')[..16].to_vec();
    // F-order of the operand, so the 2-D and stacked inputs share one layout
    let mut va = Vec::with_capacity(16);
    for j in 0..4 {
        for i in 0..4 {
            va.push(a_vec[i * 4 + j]);
        }
    }
    let v2a: Vec<f64> = va.iter().map(|x| 2.0 * x).collect();
    let a2 = rt::asarray((va.clone(), [4, 4], &device));
    let b2 = rt::asarray((v2a.clone(), [4, 4], &device));

    // stack: slice 0 = a, slice 1 = 2*a (batch axis trails under ColMajor)
    let mut stacked = va;
    stacked.extend_from_slice(&v2a);
    let b_t = rt::asarray((stacked, [4, 4, 2], &device));

    for n in [0isize, 2, 5, -1] {
        let p_nd = rt::linalg::matrix_power((b_t.view(), n));
        assert_eq!(p_nd.shape(), &[4, 4, 2]);
        let p0 = rt::linalg::matrix_power((a2.view(), n));
        let p1 = rt::linalg::matrix_power((b2.view(), n));
        assert!(rt::allclose(p_nd.i((.., .., 0)), &p0, None), "n = {n}");
        assert!(rt::allclose(p_nd.i((.., .., 1)), &p1, None), "n = {n}");
    }
}

// stacked powers agree with the per-slot 2-D calls on both storage orders,
// including off-order and strided input
#[test]
fn test_matrix_power_layouts() {
    // slot k's row-major 3x3 data
    let slot = |k: usize| -> Vec<f64> { (0..9).map(|i| ((i * 5 + 2) % 7) as f64 - 3.0 + 2.0 * k as f64).collect() };
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        // place the slots at the C-order positions of the stacked shape
        let mut v = Vec::new();
        match order {
            RowMajor => {
                for k in 0..3 {
                    v.extend(slot(k));
                }
            },
            ColMajor => {
                for r in 0..3 {
                    for c in 0..3 {
                        for k in 0..3 {
                            v.push(slot(k)[r * 3 + c]);
                        }
                    }
                }
            },
        }
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        for stored in [order, other] {
            let stacked = stack_tensor(&v, &[3], &[3, 3], order, stored, &device);
            let s0 = stack_tensor(&slot(0), &[], &[3, 3], order, stored, &device);
            let s1 = stack_tensor(&slot(1), &[], &[3, 3], order, stored, &device);
            let s2 = stack_tensor(&slot(2), &[], &[3, 3], order, stored, &device);
            for n in [0isize, 2, 5, -1] {
                let nd = rt::linalg::matrix_power((stacked.view(), n));
                for (k, slot_t) in [(0usize, &s0), (1, &s1), (2, &s2)] {
                    let d = rt::linalg::matrix_power((slot_t.view(), n));
                    // the batch axis is first under RowMajor and last under ColMajor
                    let nd_slot =
                        if order == RowMajor { nd.i(k).into_dim::<Ix2>() } else { nd.i((.., .., k)).into_dim::<Ix2>() };
                    assert!(rt::allclose(nd_slot.view(), &d, (1e-10, 1e-10)), "{order:?}/{stored:?} n = {n} slot {k}");
                }
            }
        }

        // strided batch: every second slot of a 4-slot buffer (slots {0, 2}
        // hold slot(0) scaled by 1 and 3; the data sits at the C-order
        // positions of shape [4, 3, 3] on a row-major device and [3, 3, 4]
        // on a column-major one)
        let mut wide = Vec::with_capacity(36);
        let slot_of = |k: usize| -> usize {
            if k < 2 {
                k
            } else {
                k - 2
            }
        };
        match order {
            RowMajor => {
                for k in 0..4 {
                    for x in &slot(slot_of(k)) {
                        wide.push(x * (k as f64 + 1.0));
                    }
                }
            },
            ColMajor => {
                for r in 0..3 {
                    for c in 0..3 {
                        for k in 0..4 {
                            wide.push(slot(slot_of(k))[r * 3 + c] * (k as f64 + 1.0));
                        }
                    }
                }
            },
        }
        let wide_t = match order {
            RowMajor => rt::asarray((wide, [4, 3, 3].c(), &device)),
            ColMajor => rt::asarray((wide, [3, 3, 4].c(), &device)),
        };
        let picked = match order {
            RowMajor => wide_t.i((slice!(None, None, 2), .., ..)).into_dyn(),
            ColMajor => wide_t.i((.., .., slice!(None, None, 2))).into_dyn(),
        };
        let nd = rt::linalg::matrix_power((picked.view(), 2));
        let wide0 = rt::asarray((slot(0).iter().map(|x| *x).collect::<Vec<f64>>(), [3, 3].c(), &device));
        let wide2 = rt::asarray((slot(0).iter().map(|x| 3.0 * x).collect::<Vec<f64>>(), [3, 3].c(), &device));
        let d0 = rt::linalg::matrix_power((wide0.view(), 2));
        let d1 = rt::linalg::matrix_power((wide2.view(), 2));
        let nd0 = if order == RowMajor { nd.i(0).into_dim::<Ix2>() } else { nd.i((.., .., 0)).into_dim::<Ix2>() };
        let nd1 = if order == RowMajor { nd.i(1).into_dim::<Ix2>() } else { nd.i((.., .., 1)).into_dim::<Ix2>() };
        assert!(rt::allclose(nd0.view(), &d0, (1e-10, 1e-10)));
        assert!(rt::allclose(nd1.view(), &d1, (1e-10, 1e-10)));
    }
}

// an owned or mutable input is raised in its own storage; `a.view()` keeps it
#[test]
fn test_matrix_power_inplace() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..16].to_vec();

    // an in-order owned input: the power lands in the same buffer
    let owned_in = rt::asarray((a_vec.clone(), [4, 4].c(), &device));
    let ptr_owned = owned_in.raw().as_ptr();
    let expected = rt::linalg::matrix_power((owned_in.view(), 2));
    let owned_out = rt::linalg::matrix_power((owned_in, 2));
    assert_eq!(owned_out.raw().as_ptr(), ptr_owned, "owned input is raised in place");
    assert!(rt::allclose(&owned_out, &expected, (1e-12, 1e-12)));

    // a mutable view: the power lands in the viewed storage
    let mut b = rt::asarray((a_vec.clone(), [4, 4].c(), &device));
    let ptr_b = b.raw().as_ptr();
    rt::linalg::matrix_power((b.view_mut(), 2));
    assert_eq!(b.raw().as_ptr(), ptr_b);
    assert!(rt::allclose(&b, &expected, (1e-12, 1e-12)));

    // an off-order mutable input: computed in a work buffer, written back
    let mut device_col = DeviceBLAS::default();
    device_col.set_default_order(ColMajor);
    let a_col_vec: Vec<f64> = {
        let mut va = Vec::with_capacity(16);
        for j in 0..4 {
            for i in 0..4 {
                va.push(a_vec[i * 4 + j]);
            }
        }
        va
    };
    let mut c = rt::asarray((a_col_vec.clone(), [4, 4].f(), &device_col));
    let ptr_c = c.raw().as_ptr();
    rt::linalg::matrix_power((c.view_mut(), 2));
    assert_eq!(c.raw().as_ptr(), ptr_c, "off-order write-back stays in the input buffer");
    let c_expected = rt::linalg::matrix_power((rt::asarray((a_col_vec.clone(), [4, 4].f(), &device_col)).view(), 2));
    assert!(rt::allclose(&c, &c_expected, (1e-12, 1e-12)));

    // a view keeps the input intact
    let d = rt::asarray((a_vec.clone(), [4, 4].c(), &device));
    let d_before = d.raw().to_vec();
    let _ = rt::linalg::matrix_power((d.view(), 3));
    assert_eq!(d.raw().to_vec(), d_before, "a view does not consume the input");
}

// a singular base with n < 0 errors (GETRF info > 0) where faer returns NaN
// and NumPy raises; n = 0 and n > 0 work regardless
#[test]
fn test_matrix_power_singular_negative() {
    let device = DeviceBLAS::default();
    // the second row is twice the first: an exactly singular matrix
    let a = rt::asarray((vec![1.0f64, 2.0, 2.0, 4.0], [2, 2].c(), &device));

    assert!(rt::linalg::matrix_power_f((a.view(), -1)).is_err());
    assert!(rt::linalg::matrix_power_f((a.view(), -2)).is_err());
    // a stacked singular slot errors as well
    let stacked = rt::asarray((vec![1.0f64, 2.0, 2.0, 4.0, 1.0, 0.0, 0.0, 1.0], [2, 2, 2].c(), &device));
    assert!(rt::linalg::matrix_power_f((stacked.view(), -1)).is_err());

    // a nearly singular base still returns a result, as NumPy does
    let near = rt::asarray((vec![1.0f64, 2.0, 2.0, 4.0 + 1e-10], [2, 2].c(), &device));
    assert!(rt::linalg::matrix_power_f((near.view(), -1)).is_ok());

    // nonnegative exponents are unaffected by singularity
    let p0 = rt::linalg::matrix_power((a.view(), 0));
    assert_eq!(p0.shape(), &[2, 2]);
    let p2 = rt::linalg::matrix_power((a.view(), 2));
    let a2 = &a % &a;
    assert!(rt::allclose(&p2, &a2, None));
}

#[test]
fn test_matrix_power_errors_zero() {
    let device = DeviceBLAS::default();
    // non-square input
    let a = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3].c(), &device));
    assert!(rt::linalg::matrix_power_f((a.view(), 2)).is_err());
    // 1-D input
    let v = rt::asarray((vec![1.0f64, 2.0], [2].c(), &device));
    assert!(rt::linalg::matrix_power_f((v.view(), 2)).is_err());

    // (0, 0) input: the identity of shape (0, 0)
    let z = rt::asarray((vec![0.0f64; 0], [0, 0].c(), &device));
    let p = rt::linalg::matrix_power((z.view(), 3));
    assert_eq!(p.shape(), &[0, 0]);

    // an empty batch stays empty
    let eb = rt::asarray((vec![0.0f64; 0], [0, 2, 2].c(), &device));
    let p = rt::linalg::matrix_power((eb.view(), -1));
    assert_eq!(p.shape(), &[0, 2, 2]);

    // a zero-size square axis with n < 0 skips the inversion (the empty
    // 0x0 inverse is the 0x0 identity)
    let za = rt::asarray((vec![0.0f64; 0], [2, 0, 0].c(), &device));
    let p = rt::linalg::matrix_power((za.view(), -1));
    assert_eq!(p.shape(), &[2, 0, 0]);
}

// every overload form on a small input; result types per input kind
#[test]
fn test_matrix_power_overloads() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..16].to_vec();
    let a = rt::asarray((a_vec.clone(), [4, 4].c(), &device));
    let expected = rt::linalg::matrix_power((a.view(), 2));

    // read-only kinds return a fresh tensor of the input's dim
    let r1 = rt::linalg::matrix_power((&a, 2));
    let r2 = rt::linalg::matrix_power((a.view(), 2));
    assert!(rt::allclose(&r1, &expected, (1e-12, 1e-12)));
    assert!(rt::allclose(&r2, &expected, (1e-12, 1e-12)));

    // owned and TensorMut inputs are consumed and returned raised in place
    let mut a_mut = rt::asarray((a_vec.clone(), [4, 4].c(), &device));
    let r3 = rt::linalg::matrix_power((a_mut.clone(), 2));
    assert!(rt::allclose(&r3, &expected, (1e-12, 1e-12)));
    let r4 = rt::linalg::matrix_power((a_mut.view_mut(), 2));
    assert!(rt::allclose(r4.view(), &expected, (1e-12, 1e-12)));
    // a_mut now holds a^2; raising it once more gives a^4
    let expected2 = rt::linalg::matrix_power((expected.view(), 2));
    let r5 = rt::linalg::matrix_power((a_mut.view_mut(), 2));
    assert!(rt::allclose(r5.view(), &expected2, (1e-12, 1e-12)));

    // the n = 0 identity comes back in the input dtype and shape
    let p0 = rt::linalg::matrix_power((a.view(), 0));
    assert_eq!(p0.shape(), &[4, 4]);
    let eye4: Tensor<f64, DeviceBLAS, Ix2> = rt::eye((4, &device)).into_dim::<Ix2>();
    assert!(rt::allclose(&p0, &eye4, None));
}
