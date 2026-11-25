use super::*;
use rstsr::prelude::linalg::{EigSide, Homogeneous};

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

/// Sort a spectrum lexicographically by real then imaginary part.
pub(super) fn sort_in_place(ws: &mut [c64]) {
    ws.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap().then(a.im.partial_cmp(&b.im).unwrap()));
}

/// Canonical order of an unordered spectrum: lexicographic by real then
/// imaginary part (as `np.argsort`).
pub(super) fn canonical_order<R: DataAPI<Data = Vec<c64>>>(w: &TensorAny<R, c64, DeviceBLAS, Ix1>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..w.shape()[0]).collect();
    let at = |i: usize| w.i(i).to_scalar();
    order.sort_by(|&i, &j| {
        let (wi, wj) = (at(i), at(j));
        wi.re.partial_cmp(&wj.re).unwrap().then(wi.im.partial_cmp(&wj.im).unwrap())
    });
    order
}

pub(super) fn sorted_spectrum<R: DataAPI<Data = Vec<c64>>>(w: &TensorAny<R, c64, DeviceBLAS, Ix1>) -> Vec<c64> {
    let mut ws: Vec<c64> = (0..w.shape()[0]).map(|i| w.i(i).to_scalar()).collect();
    sort_in_place(&mut ws);
    ws
}

/// Fingerprint of `|v|` with the columns reordered by `order` (phase- and
/// sign-invariant, as the eigenvector order is backend-dependent).
pub(super) fn fingerprint_abs_reordered<R: DataAPI<Data = Vec<c64>>>(
    v: &TensorAny<R, c64, DeviceBLAS, Ix2>,
    order: &[usize],
) -> f64 {
    let (m, n) = (v.nrow(), v.ncol());
    let mut va = Vec::with_capacity(m * n);
    for i in 0..m {
        for &j in order {
            va.push(v.i((i, j)).to_scalar().norm());
        }
    }
    fingerprint(&rt::asarray((va, [m, n].c(), v.device())))
}

/// Order-invariant spectrum fingerprints: the independently sorted real and
/// imaginary parts (the sorted complex order is not robust when distinct
/// eigenvalues share a real part).
pub(super) fn fingerprint_spectrum(ws: &[c64]) -> (f64, f64) {
    let device = DeviceBLAS::default();
    let mut re: Vec<f64> = ws.iter().map(|z| z.re).collect();
    let mut im: Vec<f64> = ws.iter().map(|z| z.im).collect();
    re.sort_by(|x, y| x.partial_cmp(y).unwrap());
    im.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (fingerprint(&rt::asarray((re, &device))), fingerprint(&rt::asarray((im, &device))))
}

/// Order-invariant eigenvector-block fingerprint: the per-column fingerprints
/// of `|v|`, sorted.
pub(super) fn fingerprint_abs_columns<R: DataAPI<Data = Vec<c64>>>(v: &TensorAny<R, c64, DeviceBLAS, Ix2>) -> f64 {
    let (m, n) = (v.nrow(), v.ncol());
    let mut fps: Vec<f64> = (0..n)
        .map(|j| {
            let col: Vec<f64> = (0..m).map(|i| v.i((i, j)).to_scalar().norm()).collect();
            fingerprint(&rt::asarray((col, v.device())))
        })
        .collect();
    fps.sort_by(|x, y| x.partial_cmp(y).unwrap());
    fingerprint(&rt::asarray((fps, v.device())))
}

/// Order-invariant fingerprint of a homogeneous `(alpha, beta)` stack of
/// shape `(2, n)`: `|alpha|` and `|beta|` scaled by `1 / (|alpha| + |beta|)`
/// per column, each row sorted.
pub(super) fn fingerprint_homogeneous<R: DataAPI<Data = Vec<c64>>>(w2: &TensorAny<R, c64, DeviceBLAS, Ix2>) -> f64 {
    let n = w2.shape()[1];
    let scaled = |row: usize| -> Vec<f64> {
        let mut v: Vec<f64> = (0..n)
            .map(|j| {
                let (al, be) = (w2.i((row, j)).to_scalar().norm(), w2.i((1 - row, j)).to_scalar().norm());
                al / (al + be)
            })
            .collect();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        v
    };
    let mut rows = scaled(0);
    rows.extend(scaled(1));
    fingerprint(&rt::asarray((rows, [2, n].c(), w2.device())))
}

/// The spectra agree as multisets (each expected value consumes its nearest
/// actual value, so a roundoff difference in the sort key does not matter).
fn assert_same_spectrum(ws: &[c64], expected: &[c64], what: &str) {
    assert_eq!(ws.len(), expected.len(), "{what}: spectrum length");
    let scale: f64 = expected.iter().map(|z| 1.0 + z.norm()).sum();
    let mut rest: Vec<c64> = ws.to_vec();
    for e in expected {
        let (k, _) = rest
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - *e).norm().partial_cmp(&(**b - *e).norm()).unwrap())
            .unwrap();
        let a = rest.swap_remove(k);
        assert!((a - *e).norm() < 1e-9 * scale, "{what}: {ws:?} vs {expected:?}");
    }
}

/// Physical shape of a stacked matrix array for the device order: the batch
/// dims precede the matrix axes under a row-major device and follow them
/// under a column-major one.
pub(super) fn stack_shape(order: FlagOrder, batch: &[usize], inner: &[usize]) -> Vec<usize> {
    match order {
        RowMajor => [batch, inner].concat(),
        ColMajor => [inner, batch].concat(),
    }
}

/// Dense layout of `shape` in `order`.
pub(super) fn tag(order: FlagOrder, shape: &[usize]) -> Layout<IxD> {
    shape.to_vec().new_contig(None, order)
}

/// The elements of a dense array of the physical `shape` in the iteration
/// order of `order`.
pub(super) fn layout_data<T>(shape: &[usize], order: FlagOrder, f: impl Fn(&[usize]) -> T) -> Vec<T> {
    let size = shape.iter().product();
    (0..size)
        .map(|mut off| {
            let mut idx = vec![0usize; shape.len()];
            match order {
                RowMajor => (0..shape.len()).rev().for_each(|d| {
                    idx[d] = off % shape[d];
                    off /= shape[d];
                }),
                ColMajor => (0..shape.len()).for_each(|d| {
                    idx[d] = off % shape[d];
                    off /= shape[d];
                }),
            }
            f(&idx)
        })
        .collect()
}

/// Unravel a row-major flat batch index.
fn unravel(mut k: usize, batch: &[usize]) -> Vec<isize> {
    let mut idx = vec![0isize; batch.len()];
    for dim in (0..batch.len()).rev() {
        idx[dim] = (k % batch[dim]) as isize;
        k /= batch[dim];
    }
    idx
}

type EigVecs<D> = Option<Tensor<c64, DeviceBLAS, D>>;

/// The slot `idx` (batch multi-index, at most two dims) of a stacked result
/// (copied). Under a column-major device the batch dims are the last axes.
pub(super) fn batch_slot<R: DataAPI<Data = Vec<c64>>>(
    t: &TensorAny<R, c64, DeviceBLAS, IxD>,
    idx: &[isize],
) -> Tensor<c64, DeviceBLAS, IxD> {
    let view = t.view();
    let v = view.to_dyn();
    // the leading full-range indices are spelled out: a single `..` covers one axis
    match (t.device().default_order(), t.shape().len() - idx.len(), idx) {
        (RowMajor, _, []) => v.into_owned(),
        (RowMajor, _, [i]) => v.i(*i).into_owned(),
        (RowMajor, _, [i, j]) => v.i((*i, *j)).into_owned(),
        (ColMajor, 1, [i]) => v.i((.., *i)).into_owned(),
        (ColMajor, 1, [i, j]) => v.i((.., *i, *j)).into_owned(),
        (ColMajor, 2, [i]) => v.i((.., .., *i)).into_owned(),
        (ColMajor, 2, [i, j]) => v.i((.., .., *i, *j)).into_owned(),
        _ => unreachable!("batch layout not supported by the test helper"),
    }
}

/// The block of a stacked result for the slots listed in `batch`, paired with
/// the 2-D call on each slot matrix: the spectra agree as sorted multisets,
/// and `|v|` in spectrum order agrees for every requested block.
pub(super) fn assert_parity(
    batch: &[usize],
    slots_a: &[Vec<c64>],
    slots_b: &[Vec<c64>],
    w: &Tensor<c64, DeviceBLAS, IxD>,
    vl: EigVecs<IxD>,
    vr: EigVecs<IxD>,
) {
    let order = w.device().default_order();
    let n = w.shape()[if order == RowMajor { w.shape().len() - 1 } else { 0 }];
    assert_eq!(w.shape(), &stack_shape(order, batch, &[n])[..]);
    for (k, slot_a) in slots_a.iter().enumerate() {
        let idx = unravel(k, batch);
        let device = w.device();
        let a = rt::asarray((slot_a.clone(), [n, n].c(), device)).into_dim::<Ix2>();
        let r = if slots_b.is_empty() {
            rt::linalg::eig(a.view())
        } else {
            // a single batch dim of b pairs with the last batch axis (row-major
            // semantics) or with the first one (column-major semantics)
            let kb =
                if order == RowMajor { k % batch[batch.len() - 1] } else { k / batch[1..].iter().product::<usize>() };
            let b = rt::asarray((slots_b[kb].clone(), [n, n].c(), device)).into_dim::<Ix2>();
            rt::linalg::eig((a.view(), b.view(), EigSide::Both))
        };
        let w_slot = batch_slot(w, &idx).into_dim::<Ix1>();
        assert_same_spectrum(w_slot.raw(), &sorted_spectrum(&r.eigenvalues), &format!("slot {k}"));
        let (order1, order2) = (canonical_order(&w_slot), canonical_order(&r.eigenvalues));
        for (v, v2) in [(vl.as_ref(), r.left.as_ref()), (vr.as_ref(), r.right.as_ref())] {
            if let (Some(v), Some(v2)) = (v, v2) {
                assert_eq!(v.shape(), &stack_shape(order, batch, &[n, n])[..]);
                let v_slot = batch_slot(v, &idx).into_dim::<Ix2>();
                assert!(
                    (fingerprint_abs_reordered(&v_slot, &order1) - fingerprint_abs_reordered(v2, &order2)).abs() < 1e-8,
                    "slot {k}"
                );
            }
        }
    }
}

#[test]
#[allow(clippy::type_complexity)]
fn test_eig() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let n = 1024usize;

    // default: right eigenvectors
    let r = rt::linalg::eig(a.view());
    assert!(r.left.is_none() && r.right.is_some());
    assert_eq!(r.eigenvalues.shape(), &[n]);
    assert_eq!(r.right.as_ref().unwrap().shape(), &[n, n]);
    let order = canonical_order(&r.eigenvalues);
    assert!(
        (fingerprint(&rt::asarray((sorted_spectrum(&r.eigenvalues), &device)))
            - c64::new(-53.51484095091083, -871.7616347408099))
        .norm()
            < 1e-8
    );
    assert!((fingerprint_abs_reordered(r.right.as_ref().unwrap(), &order) - 4.932401633755345).abs() < 1e-8);
    // unit 2-norm columns (GEEV normalization)
    for j in [0usize, 1, 500, 1023] {
        let norm: f64 = (0..n).map(|i| r.right.as_ref().unwrap()[[i, j]].norm_sqr()).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    // a one-sided result destructures into (w, v)
    let (w, _v): (Tensor<c64, DeviceBLAS, Ix1>, Tensor<c64, DeviceBLAS, Ix2>) = rt::linalg::eig(a.view()).into();
    assert!(
        (fingerprint(&rt::asarray((sorted_spectrum(&w), &device))) - c64::new(-53.51484095091083, -871.7616347408099))
            .norm()
            < 1e-8
    );

    // left only
    let r = rt::linalg::eig((a.view(), EigSide::Left));
    assert!(r.left.is_some() && r.right.is_none());
    assert!(
        (fingerprint_abs_reordered(r.left.as_ref().unwrap(), &canonical_order(&r.eigenvalues)) - -5.456470679201179)
            .abs()
            < 1e-8
    );

    // both sides
    let r = rt::linalg::eig((a.view(), EigSide::Both));
    assert!(r.left.is_some() && r.right.is_some());
    let order = canonical_order(&r.eigenvalues);
    assert!((fingerprint_abs_reordered(r.left.as_ref().unwrap(), &order) - -5.456470679201179).abs() < 1e-8);
    assert!((fingerprint_abs_reordered(r.right.as_ref().unwrap(), &order) - 4.932401633755345).abs() < 1e-8);
    let (_w, _vl, _vr): (Tensor<c64, DeviceBLAS, Ix1>, Tensor<c64, DeviceBLAS, Ix2>, Tensor<c64, DeviceBLAS, Ix2>) =
        r.into();

    // string and flag sides
    let r = rt::linalg::eig((a.view(), "left"));
    assert!(r.left.is_some() && r.right.is_none());
    let r = rt::linalg::eig((a.view(), "both"));
    assert!(r.left.is_some() && r.right.is_some());
    let r = rt::linalg::eig((a.view(), "none"));
    assert!(r.left.is_none() && r.right.is_none());
    let r = rt::linalg::eig((a.view(), FlagSide::L));
    assert!(r.left.is_some() && r.right.is_none());
    let r = rt::linalg::eig((a.view(), FlagSide::R));
    assert!(r.left.is_none() && r.right.is_some());
    let r = rt::linalg::eig((a.view(), EigSide::None));
    assert!(r.left.is_none() && r.right.is_none());
    assert!(
        (fingerprint(&rt::asarray((sorted_spectrum(&r.eigenvalues), &device)))
            - c64::new(-53.51484095091083, -871.7616347408099))
        .norm()
            < 1e-8
    );

    // a mutable input is consumed as scratch
    let mut a2 = a.clone();
    let r = rt::linalg::eig(a2.view_mut());
    assert!(!rt::allclose(a2.view(), a.view(), None));
    assert!(
        (fingerprint_abs_reordered(r.right.as_ref().unwrap(), &canonical_order(&r.eigenvalues)) - 4.932401633755345)
            .abs()
            < 1e-8
    );

    // invalid side string
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = rt::linalg::eig((a.view(), "diagonal"));
    }));
    assert!(result.is_err());
}

#[test]
fn test_eig_homogeneous() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let one = c64::new(1.0, 0.0);

    // (alpha, beta) stacked as (2, n) with the default right eigenvectors
    let r = rt::linalg::eig((a.view(), Homogeneous));
    assert_eq!(r.eigenvalues.shape(), &[2, 1024]);
    assert!(r.left.is_none() && r.right.is_some());
    assert_eq!(r.right.as_ref().unwrap().shape(), &[1024, 1024]);
    let ws = sorted_spectrum(&r.eigenvalues.i((0, ..)).into_dim::<Ix1>());
    assert!(
        (fingerprint(&rt::asarray((ws, &device))) - c64::new(-53.51484095091083, -871.7616347408099)).norm() < 1e-8
    );
    assert!((0..1024).all(|j| r.eigenvalues[[1, j]] == one));
    assert!((fingerprint_homogeneous(&r.eigenvalues) - -0.6191999034499596).abs() < 1e-8);

    // the eigenvector block follows the side marker as usual
    let r = rt::linalg::eig((a.view(), EigSide::None, Homogeneous));
    assert!(r.left.is_none() && r.right.is_none());
    let r = rt::linalg::eig((a.view(), "both", Homogeneous));
    assert_eq!(r.eigenvalues.shape(), &[2, 1024]);
    assert!(r.left.is_some() && r.right.is_some());
    let r = rt::linalg::eig((a.view(), FlagSide::L, Homogeneous));
    assert!(r.left.is_some() && r.right.is_none());

    // eigvals: the same homogeneous spectrum
    let w = rt::linalg::eigvals(a.view());
    let w2 = rt::linalg::eigvals((a.view(), Homogeneous));
    assert_eq!(w.shape(), &[1024]);
    assert_eq!(w2.shape(), &[2, 1024]);
    assert!(rt::allclose(w2.i((0, ..)).into_dim::<Ix1>(), &w, None));
    assert!((0..1024).all(|j| w2[[1, j]] == one));
    assert!((fingerprint_homogeneous(&w2) - -0.6191999034499596).abs() < 1e-8);
}

#[test]
fn test_eig_generalized() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    // default: right eigenvectors, rescaled to unit 2-norm as SciPy does
    let r = rt::linalg::eig((a.view(), b.view()));
    assert!(r.left.is_none() && r.right.is_some());
    let (fp_re, fp_im) = fingerprint_spectrum(r.eigenvalues.raw());
    assert!((fp_re - -27.59161731816745).abs() < 1e-8 && (fp_im - -19.504398285092513).abs() < 1e-8);
    assert!((fingerprint_abs_columns(r.right.as_ref().unwrap()) - -1.263975164568733).abs() < 1e-8);
    for j in [0usize, 1, 1000] {
        let norm: f64 = (0..1024).map(|i| r.right.as_ref().unwrap()[[i, j]].norm_sqr()).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    // both sides; copies of a and b are consumed as scratch
    let (mut a2, mut b2) = (a.clone(), b.clone());
    let r = rt::linalg::eig((a2.view_mut(), b2.view_mut(), EigSide::Both));
    assert!(!rt::allclose(a2.view(), a.view(), None) && !rt::allclose(b2.view(), b.view(), None));
    assert!((fingerprint_abs_columns(r.left.as_ref().unwrap()) - -1.0729412595886967).abs() < 1e-8);
    assert!((fingerprint_abs_columns(r.right.as_ref().unwrap()) - -1.263975164568733).abs() < 1e-8);
    let r = rt::linalg::eig((a.view(), b.view(), "left"));
    assert!(r.left.is_some() && r.right.is_none());
    assert!((fingerprint_abs_columns(r.left.as_ref().unwrap()) - -1.0729412595886967).abs() < 1e-8);

    // homogeneous: rows (alpha, beta), alpha / beta is the spectrum
    let r = rt::linalg::eig((a.view(), b.view(), Homogeneous));
    assert_eq!(r.eigenvalues.shape(), &[2, 1024]);
    let w = rt::linalg::eigvals((a.view(), b.view()));
    let mut lambda: Vec<c64> = (0..1024).map(|j| r.eigenvalues[[0, j]] / r.eigenvalues[[1, j]]).collect();
    sort_in_place(&mut lambda);
    let mut ws = sorted_spectrum(&w);
    sort_in_place(&mut ws);
    let scale: f64 = ws.iter().map(|z| 1.0 + z.norm()).sum();
    for (x, y) in lambda.iter().zip(ws.iter()) {
        assert!((*x - *y).norm() < 1e-12 * scale);
    }
    // eigvals shares the homogeneous form
    let w2 = rt::linalg::eigvals((a.view(), b.view(), Homogeneous));
    assert_eq!(w2.shape(), &[2, 1024]);
    assert!(rt::allclose(&w2, &r.eigenvalues, None));
    assert!((fingerprint_homogeneous(&w2) - -1.3522939501084459).abs() < 1e-8);
}

/// The docstring example: a 90-degree rotation has the eigenvalues `±i`.
#[test]
fn test_eig_docstring() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((
        vec![c64::new(0.0, 0.0), c64::new(-1.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
        [2, 2].c(),
        &device,
    ))
    .into_dim::<Ix2>();
    let r = rt::linalg::eig(a.view());
    assert_same_spectrum(r.eigenvalues.raw(), &[c64::new(0.0, 1.0), c64::new(0.0, -1.0)], "rotation");
    // v(0) = (1, -i) / sqrt(2) or (1, +i) / sqrt(2) for w = ∓i
    for j in 0..2 {
        let (v0, v1) = (r.right.as_ref().unwrap()[[0, j]], r.right.as_ref().unwrap()[[1, j]]);
        assert!((v0.norm() - 2.0f64.sqrt() / 2.0).abs() < 1e-12 && v0.im == 0.0);
        assert!((v1 - c64::new(0.0, -r.eigenvalues[[j]].im) * v0.norm()).norm() < 1e-12);
    }
}

/// SciPy's rules for a vanishing `beta` of a singular pencil.
#[test]
fn test_eigvals_generalized_degenerate() {
    let device = DeviceBLAS::default();
    let diag = |d: [c64; 3]| {
        rt::asarray((
            vec![
                d[0],
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                d[1],
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                d[2],
            ],
            [3, 3].c(),
            &device,
        ))
        .into_dim::<Ix2>()
    };
    let (zero, one) = (c64::new(0.0, 0.0), c64::new(1.0, 0.0));

    // alpha != 0, beta = 0: inf + 0j
    let w = rt::linalg::eigvals((
        diag([c64::new(2.0, 0.0), c64::new(3.0, 0.0), one]).view(),
        diag([one, zero, one]).view(),
    ));
    assert_eq!(w.shape(), &[3]);
    let finite: Vec<c64> = w.raw().iter().copied().filter(|z| z.re.is_finite()).collect();
    let mut finite = finite;
    sort_in_place(&mut finite);
    assert_eq!(finite, vec![one, c64::new(2.0, 0.0)]);
    assert!(w.raw().iter().filter(|z| z.re.is_infinite()).all(|z| z.im == 0.0));

    // alpha = beta = 0 with a non-real alpha present: nan + nan j
    let w = rt::linalg::eigvals((diag([c64::new(2.0, 1.0), zero, one]).view(), diag([one, zero, one]).view()));
    assert!(w.raw().iter().any(|z| z.re.is_nan() && z.im.is_nan()));
}

#[test]
fn test_eig_nd() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let n = 5usize;
        let batch = [2usize, 3];
        let slots: Vec<Vec<c64>> = (0..6).map(|k| get_vec::<c64>('b')[k * n * n..(k + 1) * n * n].to_vec()).collect();
        let shape = |batch: &[usize], inner: &[usize]| stack_shape(order, batch, inner);
        let stack = || {
            let (ib, im) = if order == RowMajor { (0, 2) } else { (2, 0) };
            layout_data(&shape(&batch, &[n, n]), order, |idx| {
                slots[idx[ib] * 3 + idx[ib + 1]][idx[im] * n + idx[im + 1]]
            })
        };
        let a = rt::asarray((stack(), tag(order, &shape(&batch, &[n, n])), &device));

        // eigenvalues (batch, n); per-slot parity with the 2-D call
        let r = rt::linalg::eig(a.view());
        assert_eq!(r.eigenvalues.shape(), &shape(&batch, &[n])[..]);
        assert_eq!(r.right.as_ref().unwrap().shape(), &shape(&batch, &[n, n])[..]);
        assert_parity(&batch, &slots, &[], &r.eigenvalues, None, r.right);

        // both sides
        let r = rt::linalg::eig((a.view(), EigSide::Both));
        assert_parity(&batch, &slots, &[], &r.eigenvalues, r.left, r.right);

        // homogeneous: the spectrum is (batch, 2, n), alpha over beta
        let r = rt::linalg::eig((a.view(), Homogeneous));
        assert_eq!(r.eigenvalues.shape(), &shape(&batch, &[2, n])[..]);
        let w2 = rt::linalg::eigvals(a.view());
        assert_eq!(w2.shape(), &shape(&batch, &[n])[..]);
        let (alpha, beta) = if order == RowMajor {
            (r.eigenvalues.i((.., .., 0, ..)).into_owned(), r.eigenvalues.i((.., .., 1, ..)).into_owned())
        } else {
            (r.eigenvalues.i((0, ..)).into_owned(), r.eigenvalues.i((1, ..)).into_owned())
        };
        let (alpha, beta) = (alpha.into_dim::<Ix3>(), beta.into_dim::<Ix3>());
        assert!(rt::allclose(&alpha, &w2, None));
        assert!(beta.raw().iter().all(|&z| z == c64::new(1.0, 0.0)));

        // a non-contiguous batch slice keeps the slots j in 1..3
        let a_s = if order == RowMajor {
            a.i((.., 1.., .., ..)).into_dim::<Ix4>()
        } else {
            a.i((.., .., .., 1..)).into_dim::<Ix4>()
        };
        let r = rt::linalg::eig(a_s.view());
        let slots_sub: Vec<Vec<c64>> = [1usize, 2, 4, 5].iter().map(|&k| slots[k].clone()).collect();
        assert_parity(&[2, 2], &slots_sub, &[], &r.eigenvalues.into_dyn(), None, r.right.map(|v| v.into_dyn()));

        // an in-order mutable input is destroyed in place
        let mut a = rt::asarray((stack(), tag(order, &shape(&batch, &[n, n])), &device));
        let snapshot = a.clone();
        let r = rt::linalg::eig(a.view_mut());
        assert!(!rt::allclose(a.view(), snapshot.view(), None));
        assert_parity(&batch, &slots, &[], &r.eigenvalues, None, r.right);

        // zero-size matrix and empty batch: zero-filled outputs
        let a0: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[3], &[0, 0]), &device));
        let r = rt::linalg::eig((a0.view(), EigSide::Both));
        assert_eq!(r.eigenvalues.shape(), &shape(&[3], &[0])[..]);
        assert_eq!(r.left.as_ref().unwrap().shape(), &shape(&[3], &[0, 0])[..]);
        assert_eq!(r.right.as_ref().unwrap().shape(), &shape(&[3], &[0, 0])[..]);
        let a0: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[0], &[n, n]), &device));
        let r = rt::linalg::eig(a0.view());
        assert_eq!(r.eigenvalues.shape(), &shape(&[0], &[n])[..]);
        assert_eq!(r.right.as_ref().unwrap().shape(), &shape(&[0], &[n, n])[..]);

        // non-square matrices are rejected
        let a_bad: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[2], &[3, 4]), &device));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = rt::linalg::eig(a_bad.view());
        }));
        assert!(result.is_err());
    }
}

#[test]
fn test_eig_nd_generalized() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let n = 4usize;
        let batch = [2usize, 3];
        let shape = |batch: &[usize], inner: &[usize]| stack_shape(order, batch, inner);
        let slot = |data: char, k: usize| -> Vec<c64> { get_vec::<c64>(data)[k * n * n..(k + 1) * n * n].to_vec() };
        let slots_a: Vec<Vec<c64>> = (0..6).map(|k| slot('b', k)).collect();
        // b carries one batch dim: pairing with the last axis of `batch` under a
        // row-major device, with the first one under a column-major device
        let slots_b: Vec<Vec<c64>> = (0..if order == RowMajor { 3 } else { 2 }).map(|k| slot('a', k)).collect();
        let nb = slots_b.len();
        // the physical index: `nb` batch dims, then (row-major) or before
        // (column-major) the matrix axes
        let data = |slots: &[Vec<c64>], batch: &[usize]| {
            let (nb, nd) = (batch.len(), batch.len() + 2);
            let (ib, im) = if order == RowMajor { (0, nb) } else { (nd - nb, 0) };
            layout_data(&shape(batch, &[n, n]), order, |idx| {
                let bidx = (0..nb).fold(0usize, |acc, d| acc * batch[d] + idx[ib + d]);
                slots[bidx % slots.len()][idx[im] * n + idx[im + 1]]
            })
        };
        let (data_a, data_b) = (data(&slots_a, &batch), data(&slots_b, &[nb]));
        let a = rt::asarray((data_a.clone(), tag(order, &shape(&batch, &[n, n])), &device));
        let b = rt::asarray((data_b.clone(), tag(order, &shape(&[nb], &[n, n])), &device));

        // the batch dims of a and b broadcast: b pairs with every a row
        let r = rt::linalg::eig((a.view(), b.view(), EigSide::Both));
        assert_eq!(r.eigenvalues.shape(), &shape(&batch, &[n])[..]);
        assert_eq!(r.left.as_ref().unwrap().shape(), &shape(&batch, &[n, n])[..]);
        assert_parity(&batch, &slots_a, &slots_b, &r.eigenvalues, r.left, r.right);
        let r = rt::linalg::eig((a.view(), b.view()));
        assert_parity(&batch, &slots_a, &slots_b, &r.eigenvalues, None, r.right);

        // homogeneous
        let r = rt::linalg::eig((a.view(), b.view(), Homogeneous));
        assert_eq!(r.eigenvalues.shape(), &shape(&batch, &[2, n])[..]);
        let w = rt::linalg::eigvals((a.view(), b.view()));
        assert_eq!(w.shape(), &shape(&batch, &[n])[..]);

        // in place: a is destroyed (used once per slice), the broadcast b is not
        let mut a = rt::asarray((data_a.clone(), tag(order, &shape(&batch, &[n, n])), &device));
        let mut b = rt::asarray((data_b.clone(), tag(order, &shape(&[nb], &[n, n])), &device));
        let (snap_a, snap_b) = (a.clone(), b.clone());
        let r = rt::linalg::eig((a.view_mut(), b.view_mut()));
        assert!(!rt::allclose(a.view(), snap_a.view(), None));
        assert!(rt::allclose(b.view(), snap_b.view(), None));
        assert_parity(&batch, &slots_a, &slots_b, &r.eigenvalues, None, r.right);

        // equal batch shapes: both are destroyed
        let a = rt::asarray((data_a.clone(), tag(order, &shape(&batch, &[n, n])), &device));
        let (ib, im) = if order == RowMajor { (0, 2) } else { (2, 0) };
        let pick = |idx: &[usize]| if order == RowMajor { idx[ib + 1] } else { idx[ib] };
        let data_b_full =
            layout_data(&shape(&batch, &[n, n]), order, |idx| slots_b[pick(idx)][idx[im] * n + idx[im + 1]]);
        let mut b_full = rt::asarray((data_b_full, tag(order, &shape(&batch, &[n, n])), &device));
        let (snap_a, snap_b) = (a.clone(), b_full.clone());
        let r = rt::linalg::eig((&a, b_full.view_mut()));
        assert!(rt::allclose(a.view(), snap_a.view(), None));
        assert!(!rt::allclose(b_full.view(), snap_b.view(), None));
        assert_parity(&batch, &slots_a, &slots_b, &r.eigenvalues, None, r.right);

        // mixed operand kinds
        let r = rt::linalg::eig((a.clone(), &b));
        assert_parity(&batch, &slots_a, &slots_b, &r.eigenvalues, None, r.right);
        let r = rt::linalg::eig((a.view(), b.view(), "left"));
        assert_parity(&batch, &slots_a, &slots_b, &r.eigenvalues, r.left, None);
        let r = rt::linalg::eig((&a, &b, "both"));
        assert_parity(&batch, &slots_a, &slots_b, &r.eigenvalues, r.left, r.right);

        // zero-size
        let a0: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[0], &[n, n]), &device));
        let b0: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[0], &[n, n]), &device));
        let r = rt::linalg::eig((a0.view(), b0.view()));
        assert_eq!(r.eigenvalues.shape(), &shape(&[0], &[n])[..]);
        let a0: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[3], &[0, 0]), &device));
        let b0: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[3], &[0, 0]), &device));
        let r = rt::linalg::eig((a0.view(), b0.view(), EigSide::Both));
        assert_eq!(r.eigenvalues.shape(), &shape(&[3], &[0])[..]);
        assert_eq!(r.left.as_ref().unwrap().shape(), &shape(&[3], &[0, 0])[..]);

        // mismatched matrix shapes are rejected
        let a_bad: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[3], &[n, n + 1]), &device));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = rt::linalg::eig((a_bad.view(), b.view()));
        }));
        assert!(result.is_err());
    }
}
