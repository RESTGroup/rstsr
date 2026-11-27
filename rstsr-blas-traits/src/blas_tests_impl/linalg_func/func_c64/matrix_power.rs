use super::*;
use rstsr_core::prelude_dev::RSTSRError;

// NumPy golden values from `func_validation_c64.py` (`# ### matrix_power`);
// they equal the constants of the faer entry's tests (same manifest data)
#[test]
fn test_matrix_power() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..16].to_vec();
    let a = rt::asarray((a_vec, [4, 4].c(), &device));

    // n = 0 -> identity, in the input dtype
    let p = rt::linalg::matrix_power((a.view(), 0));
    assert!((fingerprint(&p) - c64!(-0.3150972564720475)).norm() < 1e-8);

    // n = 1 -> a itself
    let p = rt::linalg::matrix_power((a.view(), 1));
    assert!((fingerprint(&p) - c64!(-0.5537625095088878, -3.0610178896100395)).norm() < 1e-8);

    // shortcuts (2, 3) and the squaring loop (> 3)
    for (n, golden) in [
        (2isize, c64!(-3.3981320140425257, 0.16690991094120955)),
        (3, c64!(-14.635121297638918, -1.0839909116245399)),
        (5, c64!(-7.659626934528033, 29.380294417607516)),
    ] {
        let p = rt::linalg::matrix_power((a.view(), n));
        assert!((fingerprint(&p) - golden).norm() < 1e-8, "n = {n}");
    }

    // negative exponent: invert once, then raise the inverse to |n|
    for (n, golden) in [
        (-1isize, c64!(0.23944130706023403, 0.38201618793714387)),
        (-2, c64!(-0.40283871318247577, 0.22747768835164975)),
    ] {
        let p = rt::linalg::matrix_power((a.view(), n));
        assert!((fingerprint(&p) - golden).norm() < 1e-8, "n = {n}");
    }
}

// the full-size fingerprints: n = 2 one GEMM, n = -1 one GETRF + GETRI
#[test]
fn test_matrix_power_fingerprint() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..1024 * 1024].to_vec();
    let a = rt::asarray((a_vec, [1024, 1024].c(), &device)).into_dim::<Ix2>();

    let p0 = rt::linalg::matrix_power((a.view(), 0));
    assert!((fingerprint(&p0) - c64!(0.20548286571538066)).norm() < 1e-8);
    let p1 = rt::linalg::matrix_power((a.view(), 1));
    assert!((fingerprint(&p1) - c64!(191.28900005103065, 217.50386287824938)).norm() < 1e-6);
    let p2 = rt::linalg::matrix_power((a.view(), 2));
    assert!((fingerprint(&p2) - c64!(1565.4184664848617, -66445.91164508468)).norm() < 1e-6);
    let pm1 = rt::linalg::matrix_power((a.view(), -1));
    assert!((fingerprint(&pm1) - c64!(-11.836382515155854, 8.250167298349126)).norm() < 1e-6);
}

// stacked powers agree with the per-slot 2-D calls in both storage orders
#[test]
fn test_matrix_power_nd() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..16].to_vec();

    // stack: slice 0 is a, slice 1 is 2*a
    let mut stacked = a_vec.clone();
    stacked.extend(a_vec.iter().map(|x| 2.0 * x));
    let b = rt::asarray((stacked, [2, 4, 4].c(), &device));

    let p = rt::linalg::matrix_power((b.view(), 0));
    assert_eq!(p.shape(), &[2, 4, 4]);
    assert!((fingerprint(&p) - c64!(-0.2588243168865289)).norm() < 1e-8);

    let p = rt::linalg::matrix_power((b.view(), 2));
    assert!((fingerprint(&p) - c64!(5.8067955793386306, -11.549488782937473)).norm() < 1e-8);

    let p = rt::linalg::matrix_power((b.view(), -1));
    assert!((fingerprint(&p) - c64!(-0.06327227000728392, 0.22792209504132835)).norm() < 1e-8);
}

#[test]
fn test_matrix_power_nd_colmajor() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    let a_vec = get_vec::<c64>('a')[..16].to_vec();
    // F-order of the operand, so the 2-D and stacked inputs share one layout
    let mut va = Vec::with_capacity(16);
    for j in 0..4 {
        for i in 0..4 {
            va.push(a_vec[i * 4 + j]);
        }
    }
    let v2a: Vec<c64> = va.iter().map(|x| 2.0 * x).collect();
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

// an owned or mutable input is raised in its own storage, off-order included
#[test]
fn test_matrix_power_inplace() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..16].to_vec();

    // an in-order owned input: the power lands in the same buffer
    let owned_in = rt::asarray((a_vec.clone(), [4, 4].c(), &device));
    let ptr_owned = owned_in.raw().as_ptr();
    let expected = rt::linalg::matrix_power((owned_in.view(), 2));
    let owned_out = rt::linalg::matrix_power((owned_in, 2));
    assert_eq!(owned_out.raw().as_ptr(), ptr_owned, "owned input is raised in place");
    assert!(rt::allclose(&owned_out, &expected, (1e-12, 1e-12)));

    // a mutable view: the power lands in the viewed storage
    let mut b = rt::asarray((a_vec.clone(), [4, 4].c(), &device));
    rt::linalg::matrix_power((b.view_mut(), 2));
    assert!(rt::allclose(b.view(), &expected, (1e-12, 1e-12)));

    // an off-order mutable input (column-major storage on the row-major
    // device): computed in a work buffer, written back into the input
    let mut a_col_vec = Vec::with_capacity(16);
    for j in 0..4 {
        for i in 0..4 {
            a_col_vec.push(a_vec[i * 4 + j]);
        }
    }
    let mut c = rt::asarray((a_col_vec, [4, 4].f(), &device));
    assert!(c.f_contig() && !c.c_contig(), "the fixture must not be in-place-eligible");
    let ptr_c = c.raw().as_ptr();
    rt::linalg::matrix_power((c.view_mut(), 2));
    assert_eq!(c.raw().as_ptr(), ptr_c, "off-order write-back stays in the input buffer");
    assert!(rt::allclose(c.view(), &expected, (1e-12, 1e-12)));
}

// a singular base with n < 0 errors (GETRF info > 0) where faer returns NaN
#[test]
fn test_matrix_power_singular_negative() {
    let device = DeviceBLAS::default();
    // a real exactly singular matrix embedded in the complex dtype
    let a = rt::asarray((vec![c64!(1.0), c64!(2.0), c64!(2.0), c64!(4.0)], [2, 2].c(), &device));
    let err = rt::linalg::matrix_power_f((a.view(), -1)).unwrap_err();
    assert!(matches!(err.inner, RSTSRError::InvalidValue(_)), "{err}");
    assert!(rt::linalg::matrix_power_f((a.view(), -2)).is_err());

    // a complex singular matrix: the second row is 2 times the first
    let b = rt::asarray((vec![c64!(1.0, 1.0), c64!(2.0), c64!(2.0, 2.0), c64!(4.0)], [2, 2].c(), &device));
    assert!(rt::linalg::matrix_power_f((b.view(), -1)).is_err());

    // a nearly singular base still returns a result, as NumPy does
    let near =
        rt::asarray((vec![c64!(1.0, 1.0), c64!(2.0), c64!(2.0, 2.0), c64!(4.0, 4.0000000001)], [2, 2].c(), &device));
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
    let a = rt::asarray((vec![c64!(1.0), c64!(2.0), c64!(3.0), c64!(4.0), c64!(5.0), c64!(6.0)], [2, 3].c(), &device));
    let err = rt::linalg::matrix_power_f((a.view(), 2)).unwrap_err();
    assert!(matches!(err.inner, RSTSRError::InvalidLayout(_)), "{err}");

    // (0, 0) input: the identity of shape (0, 0); an empty batch stays empty
    let z = rt::asarray((Vec::<c64>::new(), [0, 0].c(), &device));
    let p = rt::linalg::matrix_power((z.view(), 3));
    assert_eq!(p.shape(), &[0, 0]);
    let eb = rt::asarray((Vec::<c64>::new(), [0, 2, 2].c(), &device));
    let p = rt::linalg::matrix_power((eb.view(), -1));
    assert_eq!(p.shape(), &[0, 2, 2]);
    // a zero-size square axis with n < 0: the empty 0x0 inverse
    let za = rt::asarray((Vec::<c64>::new(), [2, 0, 0].c(), &device));
    let p = rt::linalg::matrix_power((za.view(), -1));
    assert_eq!(p.shape(), &[2, 0, 0]);
}

// every overload form on a small input; result types per input kind
#[test]
fn test_matrix_power_overloads() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..16].to_vec();
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
    let eye4: Tensor<c64, DeviceBLAS, Ix2> = rt::eye((4, &device)).into_dim::<Ix2>();
    assert!(rt::allclose(&p0, &eye4, None));
}

// a scalar matrix with an exact power ladder: ((1+i) I)^n = (1+i)^n I
#[test]
fn test_matrix_power_scalar_eye() {
    let device = DeviceBLAS::default();
    let scalar_eye = |x: c64| -> Tensor<c64, DeviceBLAS, Ix2> {
        let mut data = vec![c64!(0.0); 9];
        for i in 0..3 {
            data[i * 4] = x;
        }
        rt::asarray((data, [3, 3].c(), &device)).into_dim::<Ix2>()
    };
    let a = scalar_eye(c64!(1.0, 1.0));
    for (n, expected) in [
        (2isize, c64!(0.0, 2.0)),
        (3, c64!(-2.0, 2.0)),
        (4, c64!(-4.0)),
        // 7 = 0b111 runs the full bit loop
        (7, c64!(8.0, -8.0)),
        (-1, c64!(0.5, -0.5)),
        (-2, c64!(0.0, -0.5)),
    ] {
        let p = rt::linalg::matrix_power((a.view(), n));
        let want = scalar_eye(expected);
        assert!(rt::allclose(&p, &want, (1e-12, 1e-12)), "n = {n}");
    }
}
