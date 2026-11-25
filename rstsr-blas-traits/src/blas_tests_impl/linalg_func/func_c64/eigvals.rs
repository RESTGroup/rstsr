use super::*;
use rstsr::prelude::linalg::Homogeneous;

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

#[test]
fn test_eigvals() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let fp = |w: &Tensor<c64, DeviceBLAS, Ix1>| fingerprint(&rt::asarray((super::eig::sorted_spectrum(w), &device)));

    // standard problem: the spectrum alone
    let w = rt::linalg::eigvals(a.view());
    assert_eq!(w.shape(), &[1024]);
    assert!((fp(&w) - c64::new(-53.51484095091083, -871.7616347408099)).norm() < 1e-8);
    let w2 = rt::linalg::eigvals(&a);
    assert!(rt::allclose(&w2, &w, None));

    // homogeneous: (alpha, beta) stacked as (2, n)
    let w2 = rt::linalg::eigvals((a.view(), Homogeneous));
    assert_eq!(w2.shape(), &[2, 1024]);
    assert!(rt::allclose(w2.i((0, ..)).into_dim::<Ix1>(), &w, None));
    assert!((0..1024).all(|j| w2[[1, j]] == c64::new(1.0, 0.0)));
    let w2b = rt::linalg::eigvals((&a, Homogeneous));
    assert!(rt::allclose(&w2b, &w2, None));
    assert!((super::eig::fingerprint_homogeneous(&w2) - -0.6191999034499596).abs() < 1e-8);

    // generalized: alpha / beta
    let wg = rt::linalg::eigvals((a.view(), b.view()));
    assert_eq!(wg.shape(), &[1024]);
    let (fp_re, fp_im) = super::eig::fingerprint_spectrum(wg.raw());
    assert!((fp_re - -27.59161731816745).abs() < 1e-8 && (fp_im - -19.504398285092513).abs() < 1e-8);
    let wg2 = rt::linalg::eigvals((&a, &b));
    assert!(rt::allclose(&wg2, &wg, None));
    // the same spectrum as eig
    let r = rt::linalg::eig((a.view(), b.view(), "none"));
    assert!(rt::allclose(&r.eigenvalues, &wg, None));

    // generalized homogeneous: rows (alpha, beta)
    let w2 = rt::linalg::eigvals((a.view(), b.view(), Homogeneous));
    assert_eq!(w2.shape(), &[2, 1024]);
    let mut lambda: Vec<c64> = (0..1024).map(|j| w2[[0, j]] / w2[[1, j]]).collect();
    super::eig::sort_in_place(&mut lambda);
    assert!(rt::allclose(
        rt::asarray((lambda, &device)),
        rt::asarray((super::eig::sorted_spectrum(&wg), &device)),
        Some(1e-12)
    ));
    let r = rt::linalg::eig((a.view(), b.view(), Homogeneous));
    assert!(rt::allclose(&r.eigenvalues, &w2, None));
    assert!((super::eig::fingerprint_homogeneous(&w2) - -1.3522939501084459).abs() < 1e-8);

    // mutable inputs are consumed as scratch
    let mut a2 = a.clone();
    let mut b2 = b.clone();
    let w3 = rt::linalg::eigvals((a2.view_mut(), b2.view_mut()));
    assert!(!rt::allclose(a2.view(), a.view(), None) && !rt::allclose(b2.view(), b.view(), None));
    assert!(rt::allclose(&w3, &wg, None));
}

#[test]
fn test_eigvals_nd() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let n = 5usize;
        let batch = [2usize, 3];
        let shape = |batch: &[usize], inner: &[usize]| super::eig::stack_shape(order, batch, inner);
        let slot = |data: char, k: usize| -> Vec<c64> { get_vec::<c64>(data)[k * n * n..(k + 1) * n * n].to_vec() };
        let slots: Vec<Vec<c64>> = (0..6).map(|k| slot('b', k)).collect();
        // b carries one batch dim: pairing with the last axis of `batch` under a
        // row-major device, with the first one under a column-major device
        let slots_b: Vec<Vec<c64>> = (0..if order == RowMajor { 3 } else { 2 }).map(|k| slot('a', k)).collect();
        let nb = slots_b.len();
        // the physical index: `nb` batch dims, then (row-major) or before
        // (column-major) the matrix axes
        let data = |slots: &[Vec<c64>], batch: &[usize]| {
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
        let a0: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[3], &[0, 0]), &device));
        assert_eq!(rt::linalg::eigvals(a0.view()).shape(), &shape(&[3], &[0])[..]);
        assert_eq!(rt::linalg::eigvals((a0.view(), Homogeneous)).shape(), &shape(&[3], &[2, 0])[..]);
        let a0: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[0], &[n, n]), &device));
        assert_eq!(rt::linalg::eigvals(a0.view()).shape(), &shape(&[0], &[n])[..]);

        // non-square matrices are rejected
        let a_bad: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[2], &[3, 4]), &device));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = rt::linalg::eigvals(a_bad.view());
        }));
        assert!(result.is_err());
    }
}
