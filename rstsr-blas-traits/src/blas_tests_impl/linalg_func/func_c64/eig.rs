use super::*;
use rstsr::prelude::linalg::{EigResult, EigSide, Homogeneous};

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
///
/// Summed directly instead of through `fingerprint`, which builds its tensor on
/// the device and would therefore follow the device default order.
pub(super) fn fingerprint_abs_reordered<R: DataAPI<Data = Vec<c64>>>(
    v: &TensorAny<R, c64, DeviceBLAS, Ix2>,
    order: &[usize],
) -> f64 {
    let n = v.ncol();
    let mut acc = 0.0;
    for i in 0..v.nrow() {
        for (k, &j) in order.iter().enumerate() {
            acc += v.i((i, j)).to_scalar().norm() * ((i * n + k) as f64).cos();
        }
    }
    acc
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

/// The spectra agree as multisets (each expected value consumes its nearest
/// actual value, so a roundoff difference in the sort key does not matter).
pub(super) fn assert_same_spectrum(ws: &[c64], expected: &[c64], what: &str) {
    assert_same_spectrum_within(ws, expected, 1e-9, what);
}

/// As [`assert_same_spectrum`], with the tolerance `tol * sum(1 + |expected|)`.
pub(super) fn assert_same_spectrum_within(ws: &[c64], expected: &[c64], tol: f64, what: &str) {
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
        assert!((a - *e).norm() < tol * scale, "{what}: {ws:?} vs {expected:?}");
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
type Spec = Tensor<c64, DeviceBLAS, Ix1>;
type Block = Tensor<c64, DeviceBLAS, Ix2>;

/// The residual identities of one stack slot against its input data, column
/// `j` taking the slot's own `w[j]` (the per-slice LAPACK order): right
/// vectors `A v(j) = w(j) v(j)` (`A v(j) = w(j) B v(j)`), left vectors
/// `Aᴴ u(j) = conj(w(j)) u(j)` (`Aᴴ u(j) = conj(w(j)) Bᴴ u(j)`).
fn assert_slot_relation<R>(a: &[c64], b: &[c64], v: &TensorAny<R, c64, DeviceBLAS, Ix2>, w: &[c64], left: bool)
where
    R: DataAPI<Data = Vec<c64>>,
{
    let n = w.len();
    let lam_max = w.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    let scale: f64 = a.iter().map(|x| 1.0 + x.norm()).sum::<f64>() + lam_max * b.iter().map(|x| x.norm()).sum::<f64>();
    for (j, &wj) in w.iter().enumerate() {
        let lam = if left { wj.conj() } else { wj };
        for i in 0..n {
            let (mut av, mut bv) = (c64::new(0.0, 0.0), c64::new(0.0, 0.0));
            for k in 0..n {
                let bk_of = |r: usize, c: usize| if b.is_empty() { c64::new(0.0, 0.0) } else { b[r * n + c] };
                let (ak, bk) =
                    if left { (a[k * n + i].conj(), bk_of(k, i).conj()) } else { (a[i * n + k], bk_of(i, k)) };
                let vkj = v.i((k, j)).to_scalar();
                av += ak * vkj;
                bv += bk * vkj;
            }
            let residual = if b.is_empty() { av - lam * v.i((i, j)).to_scalar() } else { av - lam * bv };
            assert!(residual.norm() < 1e-10 * scale, "residual of column {j} (left = {left}): {}", residual.norm());
        }
    }
}

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
        // a single batch dim of b pairs with the last batch axis (row-major
        // semantics) or with the first one (column-major semantics)
        let b_slot: &[c64] = if slots_b.is_empty() {
            &[]
        } else if order == RowMajor {
            &slots_b[k % batch[batch.len() - 1]]
        } else {
            &slots_b[k / batch[1..].iter().product::<usize>()]
        };
        let r = if slots_b.is_empty() {
            rt::linalg::eig(a.view())
        } else {
            let b = rt::asarray((b_slot.to_vec(), [n, n].c(), device)).into_dim::<Ix2>();
            rt::linalg::eig((a.view(), b.view(), EigSide::Both))
        };
        let w_slot = batch_slot(w, &idx).into_dim::<Ix1>();
        assert_same_spectrum(w_slot.raw(), &sorted_spectrum(&r.eigenvalues), &format!("slot {k}"));
        let (order1, order2) = (canonical_order(&w_slot), canonical_order(&r.eigenvalues));
        for (left, v, v2) in [(true, vl.as_ref(), r.left.as_ref()), (false, vr.as_ref(), r.right.as_ref())] {
            if let (Some(v), Some(v2)) = (v, v2) {
                assert_eq!(v.shape(), &stack_shape(order, batch, &[n, n])[..]);
                let v_slot = batch_slot(v, &idx).into_dim::<Ix2>();
                assert!(
                    (fingerprint_abs_reordered(&v_slot, &order1) - fingerprint_abs_reordered(v2, &order2)).abs() < 1e-8,
                    "slot {k}"
                );
                assert_slot_relation(slot_a, b_slot, &v_slot, w_slot.raw(), left);
            }
        }
    }
}

/// SciPy fingerprints of one full-size solve with both sides: the sorted
/// spectrum, `|vl|` / `|vr|` in spectrum order, and GEEV's unit-norm columns.
#[test]
fn test_eig() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let r = rt::linalg::eig((a.view(), EigSide::Both));
    let (vl, vr) = (r.left.as_ref().unwrap(), r.right.as_ref().unwrap());
    let order = canonical_order(&r.eigenvalues);
    assert!(
        (fingerprint(&rt::asarray((sorted_spectrum(&r.eigenvalues), &device)))
            - c64::new(-53.51484095091083, -871.7616347408099))
        .norm()
            < 1e-8
    );
    let (fp_vl, fp_vr) = (fingerprint_abs_reordered(vl, &order), fingerprint_abs_reordered(vr, &order));
    assert!((fp_vl - -5.456470679201179).abs() < 1e-8, "eig 1024: fp_vl = {fp_vl:.17}");
    assert!((fp_vr - 4.932401633755345).abs() < 1e-8, "eig 1024: fp_vr = {fp_vr:.17}");
    for j in [order[0], order[1], order[511]] {
        let nvl: f64 = (0..1024).map(|i| vl[[i, j]].norm_sqr()).sum::<f64>();
        let nvr: f64 = (0..1024).map(|i| vr[[i, j]].norm_sqr()).sum::<f64>();
        assert!((nvl - 1.0).abs() < 1e-12, "eig 1024: |vl| column {j}: {}", nvl.sqrt());
        assert!((nvr - 1.0).abs() < 1e-12, "eig 1024: |vr| column {j}: {}", nvr.sqrt());
    }
}

/// SciPy fingerprints of a 256 x 256 generalized solve (the leading blocks
/// of the manifest arrays; a full-size QZ solve is far costlier than GEEV)
/// with both sides: the sorted spectrum and `|vl|` / `|vr|` (order-invariant),
/// the vectors rescaled to unit 2-norm as SciPy does.
#[test]
fn test_eig_generalized() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a')[..256 * 256].to_vec(), [256, 256].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((get_vec::<c64>('b')[..256 * 256].to_vec(), [256, 256].c(), &device)).into_dim::<Ix2>();
    let r = rt::linalg::eig((a.view(), b.view(), EigSide::Both));
    let (vl, vr) = (r.left.as_ref().unwrap(), r.right.as_ref().unwrap());
    let (fp_re, fp_im) = fingerprint_spectrum(r.eigenvalues.raw());
    assert!((fp_re - -38.302519430567614).abs() < 1e-8 && (fp_im - -73.72687717058412).abs() < 1e-8);
    assert!((fingerprint_abs_columns(vl) - -1.4726483066580205).abs() < 1e-8);
    assert!((fingerprint_abs_columns(vr) - -1.7966116235924798).abs() < 1e-8);
    for j in [0usize, 1, 255] {
        let norm: f64 = (0..256).map(|i| vr[[i, j]].norm_sqr()).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }
}

/// Small operands of the overload and parameter tests: 4x4 blocks of the
/// manifest data, with a regular `b` for the generalized problem.
pub(super) fn small_operands(device: &DeviceBLAS) -> (Tensor<c64, DeviceBLAS, Ix2>, Tensor<c64, DeviceBLAS, Ix2>) {
    let a = rt::asarray((get_vec::<c64>('b')[..16].to_vec(), [4, 4].c(), device)).into_dim::<Ix2>();
    let b = rt::asarray((get_vec::<c64>('a')[..16].to_vec(), [4, 4].c(), device)).into_dim::<Ix2>();
    (a, b)
}

type EigOut = EigResult<Spec, Option<Block>, Option<Block>>;
type EigOutH = EigResult<Block, Option<Block>, Option<Block>>;

/// `r` carries exactly the requested blocks, which equal the reference's
/// (any spectrum type `W`, plain or homogeneous).
fn assert_blocks<W>(r: &EigResult<W, Option<Block>, Option<Block>>, left: bool, right: bool, re: &EigOut, what: &str) {
    assert_eq!((r.left.is_some(), r.right.is_some()), (left, right), "{what}: requested blocks");
    if left {
        assert!(rt::allclose(r.left.as_ref().unwrap(), re.left.as_ref().unwrap(), None), "{what}: left vectors");
    }
    if right {
        assert!(rt::allclose(r.right.as_ref().unwrap(), re.right.as_ref().unwrap(), None), "{what}: right vectors");
    }
}

/// A plain form: the reference spectrum (up to roundoff) and blocks.
fn assert_form(r: &EigOut, left: bool, right: bool, re: &EigOut, what: &str) {
    assert_blocks(r, left, right, re, what);
    assert_same_spectrum_within(r.eigenvalues.raw(), re.eigenvalues.raw(), 1e-12, what);
}

/// The homogeneous form of the standard problem: `alpha` is the spectrum and
/// `beta` is exactly `1`.
fn assert_form_h(r: &EigOutH, left: bool, right: bool, re: &EigOut, what: &str) {
    assert_blocks(r, left, right, re, what);
    assert_eq!(r.eigenvalues.shape(), &[2, 4], "{what}: shape");
    assert!(rt::allclose(r.eigenvalues.i((0, ..)).into_dim::<Ix1>(), &re.eigenvalues, None), "{what}: alpha");
    assert!(r.eigenvalues.raw()[4..].iter().all(|&x| x == c64::new(1.0, 0.0)), "{what}: beta");
}

/// The homogeneous form of the generalized problem: `alpha / beta` is the
/// reference spectrum.
fn assert_form_gh(r: &EigOutH, left: bool, right: bool, re: &EigOut, what: &str) {
    assert_blocks(r, left, right, re, what);
    assert_eq!(r.eigenvalues.shape(), &[2, 4], "{what}: shape");
    let lambda: Vec<c64> = (0..4).map(|j| r.eigenvalues[[0, j]] / r.eigenvalues[[1, j]]).collect();
    assert_same_spectrum_within(&lambda, re.eigenvalues.raw(), 1e-12, what);
}

/// The overload matrix and the side / homogeneous parameters on a small
/// matrix: every form gives the spectrum and the requested blocks of one
/// `Both` call, for every operand kind (mutable operands are reset before
/// each form, since they are consumed as scratch).
#[test]
fn test_eig_overloads() {
    let device = DeviceBLAS::default();
    let (a, b) = small_operands(&device);
    let sa = rt::linalg::eig((a.view(), EigSide::Both));
    let ga = rt::linalg::eig((a.view(), b.view(), EigSide::Both));
    let (mut am, mut bm) = (a.clone(), b.clone());

    // the default form takes `b` only in the generalized problem
    macro_rules! default_call {
        ($a:expr) => {
            rt::linalg::eig($a)
        };
        ($a:expr, $b:expr) => {
            rt::linalg::eig(($a, $b))
        };
    }
    macro_rules! forms {
        ($re:ident, $homogeneous:ident, $what:expr, $a:expr $(, $b:expr)?) => {{
            let f = |r: EigOut, left: bool, right: bool, what: &str| {
                assert_form(&r, left, right, &$re, &format!("{}: {}", $what, what))
            };
            let h = |r: EigOutH, left: bool, right: bool, what: &str| {
                $homogeneous(&r, left, right, &$re, &format!("{}: {}", $what, what))
            };
            f(default_call!($a $(, $b)?), false, true, "default");
            f(rt::linalg::eig(($a $(, $b)?, EigSide::Left)), true, false, "EigSide::Left");
            f(rt::linalg::eig(($a $(, $b)?, EigSide::Right)), false, true, "EigSide::Right");
            f(rt::linalg::eig(($a $(, $b)?, EigSide::Both)), true, true, "EigSide::Both");
            f(rt::linalg::eig(($a $(, $b)?, EigSide::None)), false, false, "EigSide::None");
            f(rt::linalg::eig(($a $(, $b)?, "left")), true, false, "\"left\"");
            f(rt::linalg::eig(($a $(, $b)?, "right")), false, true, "\"right\"");
            f(rt::linalg::eig(($a $(, $b)?, "both")), true, true, "\"both\"");
            f(rt::linalg::eig(($a $(, $b)?, "none")), false, false, "\"none\"");
            f(rt::linalg::eig(($a $(, $b)?, FlagSide::L)), true, false, "FlagSide::L");
            f(rt::linalg::eig(($a $(, $b)?, FlagSide::R)), false, true, "FlagSide::R");
            h(rt::linalg::eig(($a $(, $b)?, Homogeneous)), false, true, "Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, EigSide::Left, Homogeneous)), true, false, "EigSide::Left, Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, EigSide::Right, Homogeneous)), false, true, "EigSide::Right, Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, EigSide::Both, Homogeneous)), true, true, "EigSide::Both, Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, EigSide::None, Homogeneous)), false, false, "EigSide::None, Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, FlagSide::L, Homogeneous)), true, false, "FlagSide::L, Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, FlagSide::R, Homogeneous)), false, true, "FlagSide::R, Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, "left", Homogeneous)), true, false, "\"left\", Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, "right", Homogeneous)), false, true, "\"right\", Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, "both", Homogeneous)), true, true, "\"both\", Homogeneous");
            h(rt::linalg::eig(($a $(, $b)?, "none", Homogeneous)), false, false, "\"none\", Homogeneous");
        }};
    }
    macro_rules! with_b_kinds {
        ($what:expr, $a:expr) => {
            forms!(ga, assert_form_gh, concat!($what, ", owned b"), $a, b.clone());
            forms!(ga, assert_form_gh, concat!($what, ", ref b"), $a, &b);
            forms!(ga, assert_form_gh, concat!($what, ", view b"), $a, b.view());
            forms!(ga, assert_form_gh, concat!($what, ", mut b"), $a, {
                bm.clone_from(&b);
                bm.view_mut()
            });
        };
    }

    forms!(sa, assert_form_h, "owned a", a.clone());
    forms!(sa, assert_form_h, "ref a", &a);
    forms!(sa, assert_form_h, "view a", a.view());
    forms!(sa, assert_form_h, "mut a", {
        am.clone_from(&a);
        am.view_mut()
    });
    with_b_kinds!("owned a", a.clone());
    with_b_kinds!("ref a", &a);
    with_b_kinds!("view a", a.view());
    with_b_kinds!("mut a", {
        am.clone_from(&a);
        am.view_mut()
    });

    // a mutable input is consumed as scratch
    am.clone_from(&a);
    let r = rt::linalg::eig(am.view_mut());
    assert!(!rt::allclose(am.view(), a.view(), None), "standard: the input must be destroyed");
    assert_form(&r, false, true, &sa, "standard, in place");
    am.clone_from(&a);
    bm.clone_from(&b);
    let r = rt::linalg::eig((am.view_mut(), bm.view_mut()));
    assert!(!rt::allclose(am.view(), a.view(), None), "generalized: a must be destroyed");
    assert!(!rt::allclose(bm.view(), b.view(), None), "generalized: b must be destroyed");
    assert_form(&r, false, true, &ga, "generalized, in place");

    // an unknown side string panics
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = rt::linalg::eig((a.view(), "diagonal"));
    }));
    assert!(result.is_err());
}

/// The homogeneous spectrum `(alpha, beta)`: `eig` and `eigvals` agree on it
/// for the standard and the generalized problem.
#[test]
fn test_eig_homogeneous() {
    let device = DeviceBLAS::default();
    let (a, b) = small_operands(&device);
    let r = rt::linalg::eig((a.view(), Homogeneous));
    assert_eq!(r.eigenvalues.shape(), &[2, 4]);
    assert!(rt::allclose(rt::linalg::eigvals((a.view(), Homogeneous)), &r.eigenvalues, None));
    let r = rt::linalg::eig((a.view(), b.view(), Homogeneous));
    assert_eq!(r.eigenvalues.shape(), &[2, 4]);
    assert!(rt::allclose(rt::linalg::eigvals((a.view(), b.view(), Homogeneous)), &r.eigenvalues, None));
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

        // non-square matrices and inputs of fewer than two dims are rejected
        let a_bad: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[2], &[3, 4]), &device));
        assert!(rt::linalg::eig_f(a_bad.view()).is_err());
        let a1: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[2], &[]), &device));
        let err = rt::linalg::eig_f(a1.view()).err().unwrap();
        assert!(err.to_string().contains("InvalidLayout"), "{err}");
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

        // mismatched matrix shapes are rejected: a square `a` of another size
        // than `b`, a non-square `a`, a non-square `b`, and a `b` of one dim
        let a_bad: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[3], &[n + 1, n + 1]), &device));
        let err = rt::linalg::eig_f((a_bad.view(), b.view())).err().unwrap();
        assert!(err.to_string().contains("InvalidLayout"), "{err}");
        let a_bad: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[3], &[n, n + 1]), &device));
        assert!(rt::linalg::eig_f((a_bad.view(), b.view())).is_err());
        let b_bad: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[nb], &[n, n + 1]), &device));
        assert!(rt::linalg::eig_f((a.view(), b_bad.view())).is_err());
        let b1: Tensor<c64, DeviceBLAS, IxD> = rt::zeros((shape(&[], &[n]), &device));
        assert!(rt::linalg::eig_f((a.view(), b1.view())).is_err());
    }
}

/// A `Both` result destructures into `(w, vl, vr)` and a one-sided result
/// into `(w, v)`.
#[test]
fn test_eig_result_conversions() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('b')[..16].to_vec(), [4, 4].c(), &device)).into_dim::<Ix2>();
    let (w, vl, vr): (Spec, Block, Block) = rt::linalg::eig((a.view(), EigSide::Both)).into();
    assert_eq!((w.shape().to_vec(), vl.shape().to_vec(), vr.shape().to_vec()), (vec![4], vec![4, 4], vec![4, 4]));
    for side in [EigSide::Left, EigSide::Right] {
        let (w, v): (Spec, Block) = rt::linalg::eig((a.view(), side)).into();
        assert_eq!((w.shape().to_vec(), v.shape().to_vec()), (vec![4], vec![4, 4]));
    }
}

#[test]
#[should_panic(expected = "EigResult: both eigenvector sides are present")]
fn test_eig_result_pair_rejects_both() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('b')[..16].to_vec(), [4, 4].c(), &device)).into_dim::<Ix2>();
    let _: (Spec, Block) = rt::linalg::eig((a.view(), EigSide::Both)).into();
}

#[test]
#[should_panic(expected = "EigResult: no eigenvector side is present")]
fn test_eig_result_pair_rejects_none() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('b')[..16].to_vec(), [4, 4].c(), &device)).into_dim::<Ix2>();
    let _: (Spec, Block) = rt::linalg::eig((a.view(), EigSide::None)).into();
}

#[test]
#[should_panic(expected = "EigResult: only the left eigenvectors are present")]
fn test_eig_result_triple_rejects_left() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('b')[..16].to_vec(), [4, 4].c(), &device)).into_dim::<Ix2>();
    let r = rt::linalg::eig((a.view(), EigSide::Left));
    let _: (Spec, Block, Block) = r.into();
}

#[test]
#[should_panic(expected = "EigResult: only the right eigenvectors are present")]
fn test_eig_result_triple_rejects_right() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('b')[..16].to_vec(), [4, 4].c(), &device)).into_dim::<Ix2>();
    let r = rt::linalg::eig((a.view(), EigSide::Right));
    let _: (Spec, Block, Block) = r.into();
}

#[test]
#[should_panic(expected = "EigResult: no eigenvector side is present")]
fn test_eig_result_triple_rejects_none() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('b')[..16].to_vec(), [4, 4].c(), &device)).into_dim::<Ix2>();
    let r = rt::linalg::eig((a.view(), EigSide::None));
    let _: (Spec, Block, Block) = r.into();
}

/// The static-dim overloads type the result by `a`: a `b` with more batch
/// axes than `a` is rejected there, while the dynamic-dim call broadcasts.
#[test]
fn test_eig_static_dim_b_with_more_batch_axes() {
    let device = DeviceBLAS::default();
    let a: Block = rt::asarray((get_vec::<c64>('b')[..16].to_vec(), [4, 4].c(), &device)).into_dim::<Ix2>();
    let mut b_data = vec![c64::new(0.0, 0.0); 32];
    for k in 0..2 {
        for i in 0..4 {
            b_data[k * 16 + i * 4 + i] = c64::new((k + 1) as f64, 0.0);
        }
    }
    let b: Tensor<c64, DeviceBLAS, Ix3> = rt::asarray((b_data, [2usize, 4, 4].c(), &device)).into_dim::<Ix3>();
    type Pair = Result<Spec, rt::Error>;
    let forms: Vec<rt::Error> = vec![
        rt::linalg::eigvals_f((a.view(), b.view())).err().unwrap(),
        rt::linalg::eigvals_f((a.view(), b.view(), Homogeneous)).err().unwrap(),
        rt::linalg::eig_f((a.view(), b.view(), EigSide::Both)).err().unwrap(),
        rt::linalg::eig_f((a.view(), b.view(), "left", Homogeneous)).err().unwrap(),
    ];
    assert!(forms.iter().all(|e| e.to_string().contains("more batch axes")), "{forms:?}");
    // equal ranks keep the static result type; the dynamic-dim call broadcasts
    let w: Pair = rt::linalg::eigvals_f((a.view(), b.i(0).into_dim::<Ix2>().view()));
    assert_eq!(w.unwrap().shape(), &[4]);
    let w = rt::linalg::eigvals((a.view().to_dyn(), b.view().to_dyn()));
    assert_eq!(w.shape(), &[2, 4]);
    // the slots pair `a` with `I` and with `2 I`: the spectrum of `a` and its half
    let w1 = rt::linalg::eigvals(a.view());
    let slot = |k: usize| (0..4).map(|j| w[[k, j]]).collect::<Vec<c64>>();
    assert_same_spectrum(&slot(0), w1.raw(), "a against I");
    let half: Vec<c64> = w1.raw().iter().map(|z| z / c64::new(2.0, 0.0)).collect();
    assert_same_spectrum(&slot(1), &half, "a against 2 I");
}
