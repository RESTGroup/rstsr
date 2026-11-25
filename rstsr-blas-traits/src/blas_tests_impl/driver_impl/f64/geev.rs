use super::*;
use duplicate::duplicate_item;
use rstsr_blas_traits::lapack_eig::*;

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

const ATOL: f64 = 1e-10;

/// Logical `(m, n)` matrix laid out per `case` (see `svd_layout.rs`), for
/// either dtype. The returned owned tensor keeps the case's layout (padded
/// leading dimension or strides), so a `view_mut()` hands that layout to the
/// driver in place.
pub(super) fn make_case<T>(v: &[T], case: &str, m: usize, n: usize, device: &DeviceBLAS) -> Tensor<T, DeviceBLAS, Ix2>
where
    T: rstsr_blas_traits::prelude_dev::BlasFloat + num::complex::ComplexFloat<Real = f64>,
{
    let a = match case {
        "f-contig" => rt::asarray((v, [m, n].f(), device)).into_dim::<Ix2>().into_owned(),
        "f-padded" => {
            let buf = rt::asarray((v, [m + 4, n + 4].f(), device)).into_dim::<Ix2>();
            buf.i((0..m, 0..n)).into_dim::<Ix2>().into_owned_keep_layout()
        },
        "c-contig" => rt::asarray((v, [m, n].c(), device)).into_dim::<Ix2>().into_owned(),
        "c-padded" => {
            let buf = rt::asarray((v, [m + 4, n + 4].c(), device)).into_dim::<Ix2>();
            buf.i((0..m, 0..n)).into_dim::<Ix2>().into_owned_keep_layout()
        },
        "non-contig" => {
            let buf = rt::asarray((v, [2 * m + 4, 2 * n + 4].c(), device)).into_dim::<Ix2>();
            let strided = buf.i((slice!(None, None, 2), slice!(None, None, 2))).into_dim::<Ix2>();
            strided.i((0..m, 0..n)).into_dim::<Ix2>().into_owned_keep_layout()
        },
        _ => panic!("unknown layout case {case}"),
    };
    assert_eq!(a.shape(), &[m, n], "layout case {case}");
    match case {
        "f-padded" => assert_eq!(a.layout().stride().to_vec(), vec![1, (m + 4) as isize]),
        "c-padded" => assert_eq!(a.layout().stride().to_vec(), vec![(n + 4) as isize, 1]),
        "non-contig" => assert_eq!(a.layout().stride().to_vec(), vec![2 * (2 * n + 4) as isize, 2]),
        _ => (),
    }
    a
}

/// A driver call that copies its input left it untouched.
pub(super) fn assert_untouched<T>(what: &str, before: &[T], t: &Tensor<T, DeviceBLAS, Ix2>)
where
    T: PartialEq + std::fmt::Debug,
{
    assert_eq!(before, t.raw(), "{what}: the input must not be written through");
}

/// An in-place driver call changed the visible block of `t` and wrote
/// nothing in the rest of its buffer (padding included).
pub(super) fn assert_destroyed_in_place<T>(what: &str, before: &[T], t: &Tensor<T, DeviceBLAS, Ix2>)
where
    T: PartialEq + std::fmt::Debug,
{
    let layout = t.layout();
    let ((m, n), (off, rs, cs)) = (
        (layout.shape()[0], layout.shape()[1]),
        (layout.offset(), layout.stride()[0] as usize, layout.stride()[1] as usize),
    );
    let block: Vec<usize> = (0..m).flat_map(|i| (0..n).map(move |j| off + i * rs + j * cs)).collect();
    let after = t.raw();
    assert!(block.iter().any(|&k| before[k] != after[k]), "{what}: the visible block must be destroyed in place");
    assert!(
        (0..after.len()).all(|k| block.contains(&k) || before[k] == after[k]),
        "{what}: the buffer outside the visible block must stay intact"
    );
}

/// Real test data with room for every layout case of an `m x n` block:
/// `(2m + 4) * (2n + 4)` elements of the `b` manifest array.
pub(super) fn data_f64(m: usize, n: usize) -> Vec<f64> {
    get_vec::<f64>('b')[..(2 * m + 4) * (2 * n + 4)].to_vec()
}

/// Complex test data of the same size (the real data interleaved as real and
/// imaginary parts).
pub(super) fn data_c64(m: usize, n: usize) -> Vec<c64> {
    get_vec::<f64>('b')[..(2 * m + 4) * (2 * n + 4) * 2].chunks(2).map(|p| c64::new(p[0], p[1])).collect()
}

/// Spectrum of a real matrix with a real eigenvalue `2` and the conjugate
/// pair `3 ± 4i` (companion form, columns ordered as LAPACK returns).
fn companion_pair() -> Vec<f64> {
    // A = [[2, 0, 0], [0, 3, -4], [0, 4, 3]]: eigenvalues 2, 3 + 4i, 3 - 4i
    vec![2.0, 0.0, 0.0, 0.0, 3.0, -4.0, 0.0, 4.0, 3.0]
}

/// Check the eigenvector relation of every column by direct arithmetic:
/// right vectors `A v(j) = w(j) v(j)`, left vectors
/// `Aᴴ u(j) = conj(w(j)) u(j)`.
pub(super) fn assert_eigen_relation<R, T>(
    a: &TensorAny<R, T, DeviceBLAS, Ix2>,
    v: &Tensor<c64, DeviceBLAS, Ix2>,
    w: &Tensor<c64, DeviceBLAS, Ix1>,
    left: bool,
) where
    R: DataAPI<Data = Vec<T>>,
    T: num::complex::ComplexFloat<Real = f64>,
{
    let to_c = |x: T| c64::new(x.re(), x.im());
    let n = a.nrow();
    let mut scale = 0.0;
    for i in 0..n {
        for k in 0..n {
            scale += to_c(a.i((i, k)).to_scalar()).norm();
        }
    }
    for j in 0..n {
        let lam = if left { w[[j]].conj() } else { w[[j]] };
        for i in 0..n {
            let mut av = c64::new(0.0, 0.0);
            for k in 0..n {
                let cak = to_c(if left { a.i((k, i)).to_scalar() } else { a.i((i, k)).to_scalar() });
                av += (if left { cak.conj() } else { cak }) * v[[k, j]];
            }
            let diff = (av - lam * v[[i, j]]).norm();
            assert!(
                diff < ATOL * scale,
                "eigenvector {j} (left = {left}) does not satisfy its relation: residual {diff}"
            );
        }
    }
}

/// GEEV's normalization: unit 2-norm columns, largest component real.
pub(super) fn assert_geev_normalized(v: &Tensor<c64, DeviceBLAS, Ix2>) {
    for j in 0..v.ncol() {
        let norm: f64 = (0..v.nrow()).map(|i| v[[i, j]].norm_sqr()).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12, "column {j} is not of unit 2-norm: {norm}");
        let (imax, mmax) =
            (0..v.nrow()).map(|i| (i, v[[i, j]].norm())).fold((0, 0.0), |acc, x| if x.1 > acc.1 { x } else { acc });
        assert!(mmax > 0.0 && v[[imax, j]].im.abs() < 1e-12, "largest component of column {j} is not real");
    }
}

pub(super) fn sorted_spectrum(w: &Tensor<c64, DeviceBLAS, Ix1>) -> Vec<c64> {
    let mut ws: Vec<c64> = (0..w.shape()[0]).map(|i| w[[i]]).collect();
    ws.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap().then(a.im.partial_cmp(&b.im).unwrap()));
    ws
}

/// Canonical order of an unordered spectrum: lexicographic by real then
/// imaginary part (as `np.argsort`).
pub(super) fn canonical_order(w: &Tensor<c64, DeviceBLAS, Ix1>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..w.shape()[0]).collect();
    order.sort_by(|&i, &j| {
        let (wi, wj) = (w[[i]], w[[j]]);
        wi.re.partial_cmp(&wj.re).unwrap().then(wi.im.partial_cmp(&wj.im).unwrap())
    });
    order
}

/// Order-invariant eigenvector-block fingerprint: every column is rescaled to
/// unit 2-norm (the raw GGEV scaling differs between LAPACK builds), then the
/// per-column fingerprints of `|v|` are sorted.
pub(super) fn fingerprint_abs_columns(v: &Tensor<c64, DeviceBLAS, Ix2>) -> f64 {
    let (m, n) = (v.nrow(), v.ncol());
    let mut fps: Vec<f64> = (0..n)
        .map(|j| {
            let norm = (0..m).map(|i| v[[i, j]].norm_sqr()).sum::<f64>().sqrt();
            let col: Vec<f64> = (0..m).map(|i| v[[i, j]].norm() / norm).collect();
            fingerprint(&rt::asarray((col, v.device())))
        })
        .collect();
    fps.sort_by(|x, y| x.partial_cmp(y).unwrap());
    fingerprint(&rt::asarray((fps, v.device())))
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

/// Fingerprint of `|v|` with the columns reordered by `order` (phase- and
/// sign-invariant, as the eigenvector order is backend-dependent).
pub(super) fn fingerprint_abs_reordered(v: &Tensor<c64, DeviceBLAS, Ix2>, order: &[usize]) -> f64 {
    let (m, n) = (v.nrow(), v.ncol());
    let mut va = Vec::with_capacity(m * n);
    for i in 0..m {
        for &j in order {
            va.push(v[[i, j]].norm());
        }
    }
    fingerprint(&rt::asarray((va, [m, n].c(), v.device())))
}

/// Compare two spectra as multisets (eigenvalues are unordered).
pub(super) fn assert_same_spectrum(w: &[c64], expected: &[c64], what: &str) {
    assert_eq!(w.len(), expected.len(), "{what}: spectrum length");
    let mut w = w.to_vec();
    w.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap().then(a.im.partial_cmp(&b.im).unwrap()));
    let mut expected = expected.to_vec();
    expected.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap().then(a.im.partial_cmp(&b.im).unwrap()));
    let scale: f64 = expected.iter().map(|x| 1.0 + x.norm()).sum();
    for (a, b) in w.iter().zip(expected.iter()) {
        assert!((*a - *b).norm() < ATOL * scale, "{what}: eigenvalues {w:?} vs {expected:?}");
    }
}

#[test]
fn test_dgeev() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // default: right eigenvectors only
    let (w, vl, vr) = DGEEV::default().a(a.view()).build().unwrap().run().unwrap();
    assert!(vl.is_none());
    let vr = vr.unwrap();
    assert_eq!(w.shape(), &[1024]);
    assert_eq!(vr.shape(), &[1024, 1024]);
    // numpy: the sorted spectrum and the reordered |v| (phase-invariant)
    let order = canonical_order(&w);
    let ws = sorted_spectrum(&w);
    assert!(
        (fingerprint(&rt::asarray((ws, &device))) - c64::new(-36.380652131982586, -116.87071686473277)).norm() < 1e-8
    );
    let fp_vr = fingerprint_abs_reordered(&vr, &order);
    assert!((fp_vr - -12.000277853410061).abs() < 1e-8, "dgeev 1024: fp_vr = {fp_vr:.17}");

    // the conjugate pairs are consecutive (positive imaginary part first)
    let mut j = 0;
    while j < 1024 {
        if w[[j]].im == 0.0 {
            j += 1;
        } else {
            assert!(w[[j]].im > 0.0 && w[[j + 1]] == w[[j]].conj(), "pair at {j}");
            j += 2;
        }
    }
    assert_geev_normalized(&vr);
}

#[test]
fn test_dgeev_relations() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('b')[..36].to_vec(), [6, 6], &device)).into_dim::<Ix2>();

    for (left, right) in [(false, true), (true, false), (true, true), (false, false)] {
        let (w, vl, vr) = DGEEV::default().a(a.view()).left(left).right(right).build().unwrap().run().unwrap();
        assert_eq!(left, vl.is_some());
        assert_eq!(right, vr.is_some());
        if let Some(vr) = &vr {
            assert_eigen_relation(&a, vr, &w, false);
            assert_geev_normalized(vr);
        }
        if let Some(vl) = &vl {
            assert_eigen_relation(&a, vl, &w, true);
            assert_geev_normalized(vl);
        }
    }
}

/// A real matrix with a real eigenvalue and a conjugate pair: the packed
/// real `VR` columns are expanded into `v(j) = x ± i y`.
#[test]
fn test_dgeev_conjugate_pair() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((companion_pair(), [3, 3].c(), &device)).into_dim::<Ix2>();
    let (w, _, vr) = DGEEV::default().a(a.view()).build().unwrap().run().unwrap();
    let vr = vr.unwrap();
    assert_same_spectrum(
        &sorted_spectrum(&w),
        &[c64::new(2.0, 0.0), c64::new(3.0, 4.0), c64::new(3.0, -4.0)],
        "companion pair",
    );
    assert_eigen_relation(&a, &vr, &w, false);

    // all-real spectrum: diag(1, 2, 3)
    let a = rt::asarray((vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], [3, 3].c(), &device)).into_dim::<Ix2>();
    let (w, vl, vr) = DGEEV::default().a(a.view()).left(true).build().unwrap().run().unwrap();
    assert_same_spectrum(
        &sorted_spectrum(&w),
        &[c64::new(1.0, 0.0), c64::new(2.0, 0.0), c64::new(3.0, 0.0)],
        "diagonal",
    );
    for j in 0..3 {
        for i in 0..3 {
            assert_eq!(vr.as_ref().unwrap()[[i, j]].im, 0.0);
            assert_eq!(vl.as_ref().unwrap()[[i, j]].im, 0.0);
        }
    }
}

#[duplicate_item(
    name                       case          driver  data_fn     ;
    [test_dgeev_layout_f_contig   ] ["f-contig"] [DGEEV] [data_f64];
    [test_dgeev_layout_f_padded   ] ["f-padded"] [DGEEV] [data_f64];
    [test_dgeev_layout_c_contig   ] ["c-contig"] [DGEEV] [data_f64];
    [test_dgeev_layout_c_padded   ] ["c-padded"] [DGEEV] [data_f64];
    [test_dgeev_layout_non_contig ] ["non-contig"] [DGEEV] [data_f64];
    [test_zgeev_layout_f_contig   ] ["f-contig"] [ZGEEV] [data_c64];
    [test_zgeev_layout_f_padded   ] ["f-padded"] [ZGEEV] [data_c64];
    [test_zgeev_layout_c_contig   ] ["c-contig"] [ZGEEV] [data_c64];
    [test_zgeev_layout_c_padded   ] ["c-padded"] [ZGEEV] [data_c64];
    [test_zgeev_layout_non_contig ] ["non-contig"] [ZGEEV] [data_c64];
)]
#[test]
fn name() {
    let data = data_fn(6, 6);
    let mut device = DeviceBLAS::default();
    for order in [RowMajor, ColMajor] {
        device.set_default_order(order);
        for n in [1usize, 2, 3, 5, 6] {
            let what = format!("{} {:?} n = {}", case, order, n);
            // an immutable view is copied by the driver: the input stays intact
            let a = make_case(&data, case, n, n, &device);
            let snapshot = a.raw().clone();
            let (w, vl, vr) = driver::default().a(a.view()).left(true).build().unwrap().run().unwrap();
            assert_untouched(&what, &snapshot, &a);
            let (vr, vl) = (vr.unwrap(), vl.unwrap());
            assert_eigen_relation(&a, &vr, &w, false);
            assert_eigen_relation(&a, &vl, &w, true);
            // a mutable view is destroyed in place when the layout is in order
            // for the call (a `1 x 1` block always is); LAPACK leaves `n <= 2`
            // intact
            let mut a_mut = make_case(&data, case, n, n, &device);
            let snapshot = a_mut.raw().clone();
            let a_ref: TensorReference<'_, _, DeviceBLAS, Ix2> = a_mut.view_mut().into();
            let (w, vl, vr) = driver::default().a(a_ref).left(true).build().unwrap().run().unwrap();
            let (vr, vl) = (vr.unwrap(), vl.unwrap());
            assert_eigen_relation(&a, &vr, &w, false);
            assert_eigen_relation(&a, &vl, &w, true);
            match (a_mut.c_prefer() || a_mut.f_prefer(), n >= 3) {
                (false, _) => assert_untouched(&what, &snapshot, &a_mut),
                (true, true) => assert_destroyed_in_place(&what, &snapshot, &a_mut),
                (true, false) => (),
            }
        }
    }
}

#[test]
fn test_zgeev() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let (w, vl, vr) = ZGEEV::default().a(a.view()).left(true).build().unwrap().run().unwrap();
    assert!(vl.is_some() && vr.is_some());
    // numpy: sorted spectrum, |vl| and |vr| columns reordered by the sorted order
    let order = canonical_order(&w);
    let ws = sorted_spectrum(&w);
    assert!(
        (fingerprint(&rt::asarray((ws, &device))) - c64::new(-53.51484095091083, -871.7616347408099)).norm() < 1e-8
    );
    {
        // temporary diagnostic: libtest prints this only when the test fails
        let vl = vl.as_ref().unwrap();
        let vr = vr.as_ref().unwrap();
        let id: Vec<usize> = (0..1024).collect();
        let resid = |v: &Tensor<c64, DeviceBLAS, Ix2>, jj: usize, left: bool| -> f64 {
            let lam = if left { w[[jj]].conj() } else { w[[jj]] };
            let (mut num, mut den) = (0.0, 0.0);
            for i in 0..1024 {
                let mut s = c64::new(0.0, 0.0);
                for k in 0..1024 {
                    s += if left { a[[k, i]].conj() * v[[k, jj]] } else { a[[i, k]] * v[[k, jj]] };
                }
                num += (s - lam * v[[i, jj]]).norm_sqr();
                den += v[[i, jj]].norm_sqr();
            }
            (num / den).sqrt()
        };
        let cnorm = |v: &Tensor<c64, DeviceBLAS, Ix2>, j: usize| -> f64 {
            (0..1024).map(|i| v[[i, j]].norm_sqr()).sum::<f64>()
        };
        eprintln!(
            "EIGDIAG default_order={:?} c_prefer={} f_prefer={} order0={} order1={} w0={:?} w1={:?}",
            device.default_order(),
            a.c_prefer(),
            a.f_prefer(),
            order[0],
            order[1],
            w[[0]],
            w[[1]],
        );
        eprintln!(
            "EIGDIAG fp_vl={:.17} fp_vr={:.17} fp_vl_id={:.17} fp_vlT={:.17} fp_vr_id={:.17}",
            fingerprint_abs_reordered(vl, &order),
            fingerprint_abs_reordered(vr, &order),
            fingerprint_abs_reordered(vl, &id),
            fingerprint_abs_reordered(&vl.t().to_owned(), &order),
            fingerprint_abs_reordered(vr, &id),
        );
        eprintln!(
            "EIGDIAG resid_vl={:.3e} resid_vl500={:.3e} resid_vr={:.3e} norm_vl0={:.17} norm_vl1={:.17} norm_vr0={:.17} vl00=({:.6},{:.6}) vl10=({:.6},{:.6})",
            resid(vl, order[0], true),
            resid(vl, order[500], true),
            resid(vr, order[0], false),
            cnorm(vl, order[0]),
            cnorm(vl, order[1]),
            cnorm(vr, order[0]),
            vl[[0, order[0]]].re,
            vl[[0, order[0]]].im,
            vl[[1, order[0]]].re,
            vl[[1, order[0]]].im,
        );
    }
    let (fp_vl, fp_vr) = (
        fingerprint_abs_reordered(vl.as_ref().unwrap(), &order),
        fingerprint_abs_reordered(vr.as_ref().unwrap(), &order),
    );
    assert!((fp_vl - -5.456470679201179).abs() < 1e-8, "zgeev 1024: fp_vl = {fp_vl:.17}");
    assert!((fp_vr - 4.932401633755345).abs() < 1e-8, "zgeev 1024: fp_vr = {fp_vr:.17}");
}

#[test]
fn test_dgeev_in_place() {
    let mut device = DeviceBLAS::default();
    for order in [RowMajor, ColMajor] {
        device.set_default_order(order);
        // an owned in-order mutable input is written through (destroyed)
        let mut a: Tensor<f64, DeviceBLAS, Ix2> =
            rt::asarray((get_vec::<f64>('b')[..36].to_vec(), [6, 6], &device)).into_dim::<Ix2>();
        let snapshot = a.clone();
        {
            let a_ref: TensorReference<'_, f64, DeviceBLAS, Ix2> = a.view_mut().into();
            DGEEV::default().a(a_ref).build().unwrap().run().unwrap();
        }
        assert!(!rt::allclose(a.view(), snapshot.view(), None), "owned in-order input must be destroyed in place");

        // a view is never written through
        let a = rt::asarray((get_vec::<f64>('b')[..36].to_vec(), [6, 6], &device)).into_dim::<Ix2>();
        let snapshot = a.clone();
        let (w, _, _) = DGEEV::default().a(a.view()).build().unwrap().run().unwrap();
        assert!(rt::allclose(&a, &snapshot, None));
        assert!(w.shape() == &[6]);
    }
}

#[test]
fn test_geev_zero_size() {
    let device = DeviceBLAS::default();
    for (m, n) in [(0usize, 0usize)] {
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([m, n], &device)).into_dim::<Ix2>();
        for (left, right) in [(false, true), (true, false), (true, true), (false, false)] {
            let (w, vl, vr) = DGEEV::default().a(a.view()).left(left).right(right).build().unwrap().run().unwrap();
            assert_eq!(w.shape(), &[0]);
            assert_eq!(vl.is_some(), left);
            assert_eq!(vr.is_some(), right);
            if let Some(v) = vl {
                assert_eq!(v.shape(), &[0, 0]);
            }
            if let Some(v) = vr {
                assert_eq!(v.shape(), &[0, 0]);
            }
        }
        let a: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros(([m, n], &device)).into_dim::<Ix2>();
        let (w, _, vr) = ZGEEV::default().a(a.view()).build().unwrap().run().unwrap();
        assert_eq!(w.shape(), &[0]);
        assert_eq!(vr.unwrap().shape(), &[0, 0]);
    }
}

#[test]
fn test_geev_rejects_non_square() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('b')[..30].to_vec(), [5, 6].c(), &device)).into_dim::<Ix2>();
    assert!(DGEEV::default().a(a.view()).build().unwrap().run().is_err());
    let a = rt::asarray((get_vec::<c64>('b')[..30].to_vec(), [5, 6].c(), &device)).into_dim::<Ix2>();
    assert!(ZGEEV::default().a(a.view()).build().unwrap().run().is_err());
}
