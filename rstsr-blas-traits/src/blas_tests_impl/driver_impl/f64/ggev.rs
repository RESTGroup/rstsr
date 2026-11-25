use super::*;
use duplicate::duplicate_item;
use rstsr_blas_traits::lapack_eig::*;

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

const ATOL: f64 = 1e-10;

/// Check the generalized eigenvector relation of every column by direct
/// arithmetic: right vectors `A v(j) = w(j) B v(j)`, left vectors
/// `Aᴴ u(j) = conj(w(j)) Bᴴ u(j)`, with `w = alpha / beta`.
fn assert_ggev_relation<RA, RB, T>(
    a: &TensorAny<RA, T, DeviceBLAS, Ix2>,
    b: &TensorAny<RB, T, DeviceBLAS, Ix2>,
    v: &Tensor<c64, DeviceBLAS, Ix2>,
    alpha: &Tensor<c64, DeviceBLAS, Ix1>,
    beta: &Tensor<T, DeviceBLAS, Ix1>,
    left: bool,
) where
    RA: DataAPI<Data = Vec<T>>,
    RB: DataAPI<Data = Vec<T>>,
    T: num::complex::ComplexFloat<Real = f64> + num::Zero,
{
    let to_c = |x: T| c64::new(x.re(), x.im());
    let n = a.nrow();
    let mut scale = 0.0;
    for i in 0..n {
        for k in 0..n {
            scale +=
                to_c(a.i((i, k)).to_scalar()).norm() + to_c(b.i((i, k)).to_scalar()).norm() * (1.0 + alpha[[k]].norm());
        }
    }
    for j in 0..n {
        assert!(beta[[j]] != T::zero(), "test data gives a regular pencil");
        let lam = if left { (alpha[[j]] / to_c(beta[[j]])).conj() } else { alpha[[j]] / to_c(beta[[j]]) };
        for i in 0..n {
            let (mut av, mut bv) = (c64::new(0.0, 0.0), c64::new(0.0, 0.0));
            for k in 0..n {
                let (cak, cbk) = if left {
                    (to_c(a.i((k, i)).to_scalar()), to_c(b.i((k, i)).to_scalar()))
                } else {
                    (to_c(a.i((i, k)).to_scalar()), to_c(b.i((i, k)).to_scalar()))
                };
                let (cak, cbk) = if left { (cak.conj(), cbk.conj()) } else { (cak, cbk) };
                let vk = v[[k, j]];
                av += cak * vk;
                bv += cbk * vk;
            }
            let diff = (av - lam * bv).norm();
            assert!(diff < ATOL * scale, "eigenvector {j} (left = {left}): residual {diff}");
        }
    }
}

/// GGEV's scaling: the largest component of every column has
/// `abs(re) + abs(im) = 1`.
fn assert_ggev_scaled(v: &Tensor<c64, DeviceBLAS, Ix2>) {
    for j in 0..v.ncol() {
        let mmax = (0..v.nrow()).map(|i| v[[i, j]].re.abs() + v[[i, j]].im.abs()).fold(0.0, f64::max);
        assert!(
            (mmax - 1.0).abs() < 1e-12,
            "column {j} is not scaled to a largest component of abs(re) + abs(im) = 1: {mmax}"
        );
    }
}

/// `b`-side data of `n x n` elements: a well-conditioned positive-definite
/// matrix built as `B = Mᵀ M + n I` from the real test data.
fn spd_b(n: usize) -> Vec<f64> {
    let raw = &get_vec::<f64>('a')[..n * n];
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut x = 0.0;
            for k in 0..n {
                x += raw[k * n + i] * raw[k * n + j];
            }
            b[i * n + j] = x + if i == j { n as f64 } else { 0.0 };
        }
    }
    b
}

/// Second-operand data of the GGEV layout cases, with room for every layout
/// case of an `m x n` block (the `a` manifest arrays; `data_f64` / `data_c64`
/// in `geev.rs` supply the first operand).
fn data_f64_second(m: usize, n: usize) -> Vec<f64> {
    get_vec::<f64>('a')[..(2 * m + 4) * (2 * n + 4)].to_vec()
}

fn data_c64_second(m: usize, n: usize) -> Vec<c64> {
    get_vec::<c64>('a')[..(2 * m + 4) * (2 * n + 4)].to_vec()
}

/// `b`-side complex data of `n x n` elements with a dominant diagonal
/// (`B = data + n I`).
fn diag_dominant_c64(n: usize) -> Vec<c64> {
    let mut b: Vec<c64> = get_vec::<c64>('b')[..n * n].to_vec();
    for i in 0..n {
        b[i * n + i] += c64::new(n as f64, 0.0);
    }
    b
}

#[test]
fn test_dggev() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [256, 256].c(), &device)).into_dim::<Ix2>();
    let b_vec = spd_b(256);
    let b = rt::asarray((&b_vec[..], [256, 256].c(), &device)).into_dim::<Ix2>();

    // default: right eigenvectors only, LAPACK's (alpha, beta) and scaling
    let (alpha, beta, vl, vr) = DGGEV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    assert!(vl.is_none());
    let vr = vr.unwrap();
    assert_eq!(alpha.shape(), &[256]);
    assert_eq!(beta.shape(), &[256]);
    assert_eq!(vr.shape(), &[256, 256]);
    assert!(beta.raw().iter().all(|&x| x > 0.0));
    assert_ggev_scaled(&vr);
    // SciPy: the sorted spectrum alpha / beta and |vr| (the LAPACK scaling is
    // kept at the driver tier). The order inside a conjugate pair is not
    // defined (real parts may differ in the last bit), so the fingerprints
    // are order-invariant; the raw GGEV scaling differs between LAPACK builds
    let w = rt::asarray(((0..256).map(|j| alpha[[j]] / beta[[j]]).collect::<Vec<_>>(), &device)).into_dim::<Ix1>();
    let ws = super::geev::sorted_spectrum(&w);
    let fp = |v: Vec<f64>| fingerprint(&rt::asarray((v, &device)));
    assert!((fp(ws.iter().map(|z| z.re).collect()) - -0.03604323300216897).abs() < 1e-8);
    let mut im: Vec<f64> = ws.iter().map(|z| z.im.abs()).collect();
    im.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!((fp(im) - -0.026149942284387048).abs() < 1e-8);
    assert!((super::geev::fingerprint_abs_columns(&vr) - -1.225589423351893).abs() < 1e-8);
}

#[test]
fn test_dggev_relations() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('b')[..36].to_vec(), [6, 6], &device)).into_dim::<Ix2>();
    let b = rt::asarray((spd_b(6), [6, 6], &device)).into_dim::<Ix2>();
    for (left, right) in [(false, true), (true, false), (true, true), (false, false)] {
        let (alpha, beta, vl, vr) =
            DGGEV::default().a(a.view()).b(b.view()).left(left).right(right).build().unwrap().run().unwrap();
        assert_eq!(left, vl.is_some());
        assert_eq!(right, vr.is_some());
        if let Some(vr) = &vr {
            assert_ggev_relation(&a, &b, vr, &alpha, &beta, false);
            assert_ggev_scaled(vr);
        }
        if let Some(vl) = &vl {
            assert_ggev_relation(&a, &b, vl, &alpha, &beta, true);
            assert_ggev_scaled(vl);
        }
    }
}

/// `A v = λ B v` with a diagonal pair has the exact eigenvalues `a(j) / b(j)`.
#[test]
fn test_dggev_diagonal() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], [3, 3].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((vec![2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0], [3, 3].c(), &device)).into_dim::<Ix2>();
    let (alpha, beta, vl, vr) = DGGEV::default().a(a.view()).b(b.view()).left(true).build().unwrap().run().unwrap();
    let mut ws: Vec<c64> = (0..3).map(|j| alpha[[j]] / beta[[j]]).collect();
    ws.sort_by(|x, y| x.re.partial_cmp(&y.re).unwrap().then(x.im.partial_cmp(&y.im).unwrap()));
    for (x, y) in ws.iter().zip([0.5, 0.75, 2.0]) {
        assert!((*x - c64::new(y, 0.0)).norm() < 1e-12);
    }
    assert_ggev_relation(&a, &b, vr.as_ref().unwrap(), &alpha, &beta, false);
    assert_ggev_relation(&a, &b, vl.as_ref().unwrap(), &alpha, &beta, true);
}

/// A singular `B` gives `beta = 0` (the eigenvalue is infinite): `B = 0`
/// makes every `beta` vanish while `alpha` is non-zero.
#[test]
fn test_dggev_singular_b() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], [3, 3].c(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([3, 3], &device)).into_dim::<Ix2>();
    let (alpha, beta, _, _) = DGGEV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    assert!(beta.raw().iter().all(|&x| x == 0.0));
    assert!(alpha.raw().iter().all(|x| x.norm() > 0.0));
}

#[duplicate_item(
    name                       case          driver  data_a     data_b       ;
    [test_dggev_layout_f_contig   ] ["f-contig"] [DGGEV] [super::geev::data_f64] [data_f64_second];
    [test_dggev_layout_f_padded   ] ["f-padded"] [DGGEV] [super::geev::data_f64] [data_f64_second];
    [test_dggev_layout_c_contig   ] ["c-contig"] [DGGEV] [super::geev::data_f64] [data_f64_second];
    [test_dggev_layout_c_padded   ] ["c-padded"] [DGGEV] [super::geev::data_f64] [data_f64_second];
    [test_dggev_layout_non_contig ] ["non-contig"] [DGGEV] [super::geev::data_f64] [data_f64_second];
    [test_zggev_layout_f_contig   ] ["f-contig"] [ZGGEV] [super::geev::data_c64] [data_c64_second];
    [test_zggev_layout_f_padded   ] ["f-padded"] [ZGGEV] [super::geev::data_c64] [data_c64_second];
    [test_zggev_layout_c_contig   ] ["c-contig"] [ZGGEV] [super::geev::data_c64] [data_c64_second];
    [test_zggev_layout_c_padded   ] ["c-padded"] [ZGGEV] [super::geev::data_c64] [data_c64_second];
    [test_zggev_layout_non_contig ] ["non-contig"] [ZGGEV] [super::geev::data_c64] [data_c64_second];
)]
#[test]
fn name() {
    use super::geev::{assert_destroyed_in_place, assert_untouched, make_case};
    let data = data_a(6, 6);
    let bdata = data_b(6, 6);
    let mut device = DeviceBLAS::default();
    for order in [RowMajor, ColMajor] {
        device.set_default_order(order);
        for n in [1usize, 2, 3, 5, 6] {
            let what = format!("{} {:?} n = {}", case, order, n);
            // immutable views are copied by the driver: the inputs stay intact
            let a = make_case(&data, case, n, n, &device);
            let b = make_case(&bdata, case, n, n, &device);
            let (a_snap, b_snap) = (a.raw().clone(), b.raw().clone());
            let (alpha, beta, vl, vr) =
                driver::default().a(a.view()).b(b.view()).left(true).build().unwrap().run().unwrap();
            assert_untouched(&what, &a_snap, &a);
            assert_untouched(&what, &b_snap, &b);
            let (vr, vl) = (vr.unwrap(), vl.unwrap());
            assert_ggev_relation(&a, &b, &vr, &alpha, &beta, false);
            assert_ggev_relation(&a, &b, &vl, &alpha, &beta, true);
            assert_ggev_scaled(&vr);
            // the layout cases of a and b are independent, also when the two
            // operands prefer different orders
            for (case_a, case_b) in [("f-padded", "non-contig"), ("f-padded", "c-padded"), ("c-padded", "f-padded")] {
                let what = format!("{what} mix {case_a} x {case_b}");
                let a_mix = make_case(&data, case_a, n, n, &device);
                let b_mix = make_case(&bdata, case_b, n, n, &device);
                let (a_snap, b_snap) = (a_mix.raw().clone(), b_mix.raw().clone());
                let (alpha, beta, vl, vr) =
                    driver::default().a(a_mix.view()).b(b_mix.view()).left(true).build().unwrap().run().unwrap();
                assert_untouched(&what, &a_snap, &a_mix);
                assert_untouched(&what, &b_snap, &b_mix);
                assert_ggev_relation(&a_mix, &b_mix, vr.as_ref().unwrap(), &alpha, &beta, false);
                assert_ggev_relation(&a_mix, &b_mix, vl.as_ref().unwrap(), &alpha, &beta, true);
                // a mutable operand of a conflicting-order pencil is consumed
                // only if its own order is the call order
                let mut a_mix = a_mix;
                let mut b_mix = b_mix;
                let a_ref: TensorReference<'_, _, DeviceBLAS, Ix2> = a_mix.view_mut().into();
                let b_ref: TensorReference<'_, _, DeviceBLAS, Ix2> = b_mix.view_mut().into();
                driver::default().a(a_ref).b(b_ref).left(true).build().unwrap().run().unwrap();
                let destroyed = [("a", &a_snap, &a_mix), ("b", &b_snap, &b_mix)].map(|(operand, before, t)| {
                    let what = format!("{what} {operand}");
                    let block_changed = before.iter().zip(t.raw()).any(|(x, y)| x != y);
                    if block_changed {
                        assert_destroyed_in_place(&what, before, t);
                    } else {
                        assert_untouched(&what, before, t);
                    }
                    block_changed
                });
                assert!(n < 3 || destroyed.iter().any(|&d| d), "{what}: one operand must be consumed");
            }
            // mutable views are destroyed in place when their layout is in
            // order for the call (a `1 x 1` block always is); LAPACK leaves
            // `n <= 2` intact
            let mut a_mut = make_case(&data, case, n, n, &device);
            let mut b_mut = make_case(&bdata, case, n, n, &device);
            let (a_snap, b_snap) = (a_mut.raw().clone(), b_mut.raw().clone());
            let a_ref: TensorReference<'_, _, DeviceBLAS, Ix2> = a_mut.view_mut().into();
            let b_ref: TensorReference<'_, _, DeviceBLAS, Ix2> = b_mut.view_mut().into();
            let (alpha, beta, vl, vr) = driver::default().a(a_ref).b(b_ref).left(true).build().unwrap().run().unwrap();
            assert_ggev_relation(&a, &b, vr.as_ref().unwrap(), &alpha, &beta, false);
            assert_ggev_relation(&a, &b, vl.as_ref().unwrap(), &alpha, &beta, true);
            for (operand, before, t) in [("a", &a_snap, &a_mut), ("b", &b_snap, &b_mut)] {
                let what = format!("{what} {operand}");
                match (t.c_prefer() || t.f_prefer(), n >= 3) {
                    (false, _) => assert_untouched(&what, before, t),
                    (true, true) => assert_destroyed_in_place(&what, before, t),
                    (true, false) => (),
                }
            }
        }
    }
}

/// Non-square or mismatched operands are rejected before the driver call.
#[test]
fn test_ggev_rejects_shapes() {
    let device = DeviceBLAS::default();
    // (shape of a, shape of b): non-square a, non-square b, mismatched
    for ([ma, na], [mb, nb]) in [([5, 6], [6, 6]), ([5, 5], [6, 5]), ([5, 5], [6, 6])] {
        let (av, bv) = (get_vec::<f64>('a')[..ma * na].to_vec(), get_vec::<f64>('b')[..mb * nb].to_vec());
        let a = rt::asarray((av, [ma, na].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((bv, [mb, nb].c(), &device)).into_dim::<Ix2>();
        assert!(DGGEV::default().a(a.view()).b(b.view()).build().unwrap().run().is_err());
        let (av, bv) = (get_vec::<c64>('a')[..ma * na].to_vec(), get_vec::<c64>('b')[..mb * nb].to_vec());
        let a = rt::asarray((av, [ma, na].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((bv, [mb, nb].c(), &device)).into_dim::<Ix2>();
        assert!(ZGGEV::default().a(a.view()).b(b.view()).build().unwrap().run().is_err());
    }
}

/// An empty pencil gives empty `(alpha, beta)` and eigenvector blocks.
#[test]
fn test_ggev_zero_size() {
    let device = DeviceBLAS::default();
    let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([0, 0], &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([0, 0], &device)).into_dim::<Ix2>();
    for (left, right) in [(false, true), (true, false), (true, true), (false, false)] {
        let (alpha, beta, vl, vr) =
            DGGEV::default().a(a.view()).b(b.view()).left(left).right(right).build().unwrap().run().unwrap();
        assert_eq!(alpha.shape(), &[0]);
        assert_eq!(beta.shape(), &[0]);
        assert_eq!(vl.is_some(), left);
        assert_eq!(vr.is_some(), right);
        if let Some(v) = vl {
            assert_eq!(v.shape(), &[0, 0]);
        }
        if let Some(v) = vr {
            assert_eq!(v.shape(), &[0, 0]);
        }
    }
    let a: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros(([0, 0], &device)).into_dim::<Ix2>();
    let b: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros(([0, 0], &device)).into_dim::<Ix2>();
    let (alpha, beta, _, vr) = ZGGEV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    assert_eq!(alpha.shape(), &[0]);
    assert_eq!(beta.shape(), &[0]);
    assert_eq!(vr.unwrap().shape(), &[0, 0]);
}

#[test]
fn test_zggev() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a'), [256, 256].c(), &device)).into_dim::<Ix2>();
    let b_vec = diag_dominant_c64(256);
    let b = rt::asarray((&b_vec[..], [256, 256].c(), &device)).into_dim::<Ix2>();
    let (alpha, beta, vl, vr) = ZGGEV::default().a(a.view()).b(b.view()).left(true).build().unwrap().run().unwrap();
    assert!(vl.is_some() && vr.is_some());
    // LAPACK's `beta` of the complex routine is complex
    assert!(beta.raw().iter().all(|&x| x.norm() > 0.0));
    // scipy: the sorted spectrum alpha / beta and the unit-norm |vl| / |vr|
    // (fingerprinted order-invariant; the raw LAPACK scaling is kept in the
    // outputs but differs between LAPACK builds)
    let w = rt::asarray(((0..256).map(|j| alpha[[j]] / beta[[j]]).collect::<Vec<_>>(), &device)).into_dim::<Ix1>();
    let (fp_re, fp_im) = super::geev::fingerprint_spectrum(w.raw());
    assert!((fp_re - -0.11836546248737323).abs() < 1e-8 && (fp_im - -0.11210131721328423).abs() < 1e-8);
    assert!((super::geev::fingerprint_abs_columns(vl.as_ref().unwrap()) - -1.3881039015638714).abs() < 1e-8);
    assert!((super::geev::fingerprint_abs_columns(vr.as_ref().unwrap()) - -1.9454332085201405).abs() < 1e-8);
}
