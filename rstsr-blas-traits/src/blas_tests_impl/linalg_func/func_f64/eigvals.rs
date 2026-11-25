use super::*;
use rstsr::prelude::linalg::{EigSide, Homogeneous};

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

/// SciPy fingerprints of the full-size value-only solves: the standard and
/// the generalized spectrum.
#[test]
fn test_eigvals() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let fp = |w: &Tensor<c64, DeviceBLAS, Ix1>| fingerprint(&rt::asarray((super::eig::sorted_spectrum(w), &device)));
    let w = rt::linalg::eigvals(a.view());
    assert_eq!(w.shape(), &[1024]);
    assert!((fp(&w) - c64::new(-36.380652131982586, -116.87071686473277)).norm() < 1e-8);
    // generalized: a 256 x 256 pencil (the leading blocks; a full-size QZ
    // solve is far costlier than GEEV), SPD B for a well-conditioned problem
    let a = rt::asarray((get_vec::<f64>('a')[..256 * 256].to_vec(), [256, 256].c(), &device)).into_dim::<Ix2>();
    let b0 = rt::asarray((get_vec::<f64>('b')[..256 * 256].to_vec(), [256, 256].c(), &device)).into_dim::<Ix2>();
    let mut b = &b0 % b0.view().t();
    for i in 0..256 {
        b[[i, i]] += 256.0;
    }
    let b = &b;
    let wg = rt::linalg::eigvals((a.view(), b.view()));
    let (fp_re, fp_im) = super::eig::fingerprint_spectrum(wg.raw());
    assert!((fp_re - -0.08043969532621999).abs() < 1e-8 && (fp_im - -0.07080695445332981).abs() < 1e-8);
}

/// The overload matrix of `eigvals` on a small matrix (all sixteen operand
/// kind combinations): every form gives the spectrum of `eig` without
/// vectors, the homogeneous forms its `(alpha, beta)` rows; a mutable input
/// is consumed as scratch.
#[test]
fn test_eigvals_overloads() {
    let device = DeviceBLAS::default();
    let (a, b) = super::eig::small_operands(&device);
    let ws = rt::linalg::eig((a.view(), EigSide::None)).eigenvalues;
    let wgs = rt::linalg::eig((a.view(), b.view(), EigSide::None)).eigenvalues;
    let hs = rt::linalg::eigvals((a.view(), Homogeneous));
    let hgs = rt::linalg::eigvals((a.view(), b.view(), Homogeneous));
    let (mut am, mut bm) = (a.clone(), b.clone());

    macro_rules! forms {
        ($what:expr, $a:expr, $b:expr) => {{
            super::eig::assert_same_spectrum_within(rt::linalg::eigvals($a).raw(), ws.raw(), 1e-12, $what);
            super::eig::assert_same_spectrum_within(rt::linalg::eigvals(($a, $b)).raw(), wgs.raw(), 1e-12, $what);
            assert!(rt::allclose(&rt::linalg::eigvals(($a, Homogeneous)), &hs, None), $what);
            assert!(rt::allclose(&rt::linalg::eigvals(($a, $b, Homogeneous)), &hgs, None), $what);
        }};
    }
    macro_rules! with_b_kinds {
        ($what:expr, $a:expr) => {
            forms!(concat!($what, ", owned b"), $a, b.clone());
            forms!(concat!($what, ", ref b"), $a, &b);
            forms!(concat!($what, ", view b"), $a, b.view());
            forms!(concat!($what, ", mut b"), $a, {
                bm.clone_from(&b);
                bm.view_mut()
            });
        };
    }
    with_b_kinds!("owned a", a.clone());
    with_b_kinds!("ref a", &a);
    with_b_kinds!("view a", a.view());
    with_b_kinds!("mut a", {
        am.clone_from(&a);
        am.view_mut()
    });

    // a mutable input is consumed as scratch
    am.clone_from(&a);
    let w = rt::linalg::eigvals(am.view_mut());
    assert!(!rt::allclose(am.view(), a.view(), None), "the input must be destroyed");
    super::eig::assert_same_spectrum_within(w.raw(), ws.raw(), 1e-12, "in place");
}
#[test]
fn test_eigvals_nd() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let n = 5usize;
        let batch = [2usize, 3];
        let shape = |batch: &[usize], inner: &[usize]| super::eig::stack_shape(order, batch, inner);
        let slot = |data: char, k: usize| get_vec::<f64>(data)[k * n * n..(k + 1) * n * n].to_vec();
        let slots: Vec<Vec<f64>> = (0..6).map(|k| slot('b', k)).collect();
        // b carries one batch dim: pairing with the last axis of `batch` under a
        // row-major device, with the first one under a column-major device
        let slots_b: Vec<Vec<f64>> = (0..if order == RowMajor { 3 } else { 2 }).map(|k| slot('a', k)).collect();
        let nb = slots_b.len();
        // the physical index: `nb` batch dims, then (row-major) or before
        // (column-major) the matrix axes
        let data = |slots: &[Vec<f64>], batch: &[usize]| {
            let (nb, nd) = (batch.len(), batch.len() + 2);
            let (ib, im) = if order == RowMajor { (0, nb) } else { (nd - nb, 0) };
            super::eig::layout_data(&shape(batch, &[n, n]), order, |idx| {
                let bidx = (0..nb).fold(0usize, |acc, d| acc * batch[d] + idx[ib + d]);
                slots[bidx % slots.len()][idx[im] * n + idx[im + 1]]
            })
        };
        let a = rt::asarray((data(&slots, &batch), super::eig::tag(order, &shape(&batch, &[n, n])), &device));
        let b = rt::asarray((data(&slots_b, &[nb]), super::eig::tag(order, &shape(&[nb], &[n, n])), &device));

        // per-slot parity with the 2-D call
        let w = rt::linalg::eigvals(a.view());
        assert_eq!(w.shape(), &shape(&batch, &[n])[..]);
        super::eig::assert_parity(&batch, &slots, &[], &w, None, None);
        let wg = rt::linalg::eigvals((a.view(), b.view()));
        assert_eq!(wg.shape(), &shape(&batch, &[n])[..]);
        super::eig::assert_parity(&batch, &slots, &slots_b, &wg, None, None);

        // homogeneous
        let w2 = rt::linalg::eigvals((a.view(), Homogeneous));
        assert_eq!(w2.shape(), &shape(&batch, &[2, n])[..]);
        let w2g = rt::linalg::eigvals((a.view(), b.view(), Homogeneous));
        assert_eq!(w2g.shape(), &shape(&batch, &[2, n])[..]);

        // in place
        let mut a = rt::asarray((data(&slots, &batch), super::eig::tag(order, &shape(&batch, &[n, n])), &device));
        let snapshot = a.clone();
        let w3 = rt::linalg::eigvals(a.view_mut());
        assert!(!rt::allclose(a.view(), snapshot.view(), None));
        assert!(rt::allclose(&w3, &w, None));

        // zero-size
        let a0: Tensor<f64, DeviceBLAS, IxD> = rt::zeros((shape(&[3], &[0, 0]), &device));
        assert_eq!(rt::linalg::eigvals(a0.view()).shape(), &shape(&[3], &[0])[..]);
        assert_eq!(rt::linalg::eigvals((a0.view(), Homogeneous)).shape(), &shape(&[3], &[2, 0])[..]);
        let a0: Tensor<f64, DeviceBLAS, IxD> = rt::zeros((shape(&[0], &[n, n]), &device));
        assert_eq!(rt::linalg::eigvals(a0.view()).shape(), &shape(&[0], &[n])[..]);

        // non-square matrices are rejected
        let a_bad: Tensor<f64, DeviceBLAS, IxD> = rt::zeros((shape(&[2], &[3, 4]), &device));
        assert!(rt::linalg::eigvals_f(a_bad.view()).is_err());
    }
}

/// `lambda = alpha / beta` stays correct when `beta` is so large or so small
/// that its square over- or underflows (SciPy divides in the overflow-safe
/// form as well). The pencil is scaled as a whole: LAPACK's own deflation
/// tolerances are relative, so a `b` that is tiny next to `a` is a different
/// (singular) problem.
#[test]
fn test_eigvals_generalized_extreme_scale() {
    let device = DeviceBLAS::default();
    // eigenvalues of [[1, 2], [3, 4]] for the pencil `(s A, s I)`
    let unit = [c64::new(-0.3722813232690143, 0.0), c64::new(5.372281323269014, 0.0)];
    let mat = |s: f64| rt::asarray((vec![s, 2.0 * s, 3.0 * s, 4.0 * s], [2, 2].c(), &device)).into_dim::<Ix2>();
    let eye = |s: f64| rt::asarray((vec![s, 0.0, 0.0, s], [2, 2].c(), &device)).into_dim::<Ix2>();
    for s in [1e160, 1e-200] {
        let w = rt::linalg::eigvals((mat(s).view(), eye(s).view()));
        assert!(w.raw().iter().all(|z| z.re.is_finite() && z.im == 0.0), "scale {s}: {:?}", w.raw());
        super::eig::assert_same_spectrum(w.raw(), &unit, &format!("pencil scaled by {s}"));
    }
    // the homogeneous spectrum carries `(alpha, beta)` as LAPACK gives them
    let h = rt::linalg::eigvals((mat(1e160).view(), eye(1e160).view(), Homogeneous));
    assert!(h.raw().iter().all(|z| z.re.is_finite() && z.im.is_finite() && z.norm() > 0.0), "{:?}", h.raw());
}
