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
    // scipy: the sorted spectrum alpha / beta and |vr| (the LAPACK scaling is
    // kept at the driver tier). The real parts of a conjugate pair may differ
    // in the last bit, so the order of the two members is not defined:
    // fingerprint the real parts, the |imaginary parts| and the unit-norm
    // columns of |vr| one by one (all order-invariant; the raw GGEV scaling
    // differs between LAPACK builds)
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
    name                       case          ;
    [test_dggev_layout_f_contig   ] ["f-contig"];
    [test_dggev_layout_f_padded   ] ["f-padded"];
    [test_dggev_layout_c_contig   ] ["c-contig"];
    [test_dggev_layout_c_padded   ] ["c-padded"];
    [test_dggev_layout_non_contig ] ["non-contig"];
)]
#[test]
fn name() {
    let data = super::geev::data_f64(6, 6);
    let bdata = super::geev::data_f64_b(6, 6);
    for n in [1usize, 2, 3, 5, 6] {
        // the layout cases of a and b are independent
        let a = super::geev::make_case(&data, case, n, n);
        let b = super::geev::make_case(&bdata, case, n, n);
        let (alpha, beta, vl, vr) = DGGEV::default().a(a.view()).b(b.view()).left(true).build().unwrap().run().unwrap();
        assert_ggev_relation(&a, &b, vr.as_ref().unwrap(), &alpha, &beta, false);
        assert_ggev_relation(&a, &b, vl.as_ref().unwrap(), &alpha, &beta, true);
        let a = super::geev::make_case(&data, "f-padded", n, n);
        let b = super::geev::make_case(&bdata, "non-contig", n, n);
        let (alpha, beta, vl, vr) = DGGEV::default().a(a.view()).b(b.view()).left(true).build().unwrap().run().unwrap();
        assert_ggev_relation(&a, &b, vr.as_ref().unwrap(), &alpha, &beta, false);
        assert_ggev_relation(&a, &b, vl.as_ref().unwrap(), &alpha, &beta, true);
    }
}

#[duplicate_item(
    name                       case          ;
    [test_zggev_layout_f_contig   ] ["f-contig"];
    [test_zggev_layout_f_padded   ] ["f-padded"];
    [test_zggev_layout_c_contig   ] ["c-contig"];
    [test_zggev_layout_c_padded   ] ["c-padded"];
    [test_zggev_layout_non_contig ] ["non-contig"];
)]
#[test]
fn name() {
    let data = super::geev::data_c64(6, 6);
    let bdata = super::geev::data_c64_b(6, 6);
    for n in [1usize, 2, 3, 5, 6] {
        let a = super::geev::make_case(&data, case, n, n);
        let b = super::geev::make_case(&bdata, case, n, n);
        let (alpha, beta, vl, vr) = ZGGEV::default().a(a.view()).b(b.view()).left(true).build().unwrap().run().unwrap();
        assert_ggev_relation(&a, &b, vr.as_ref().unwrap(), &alpha, &beta, false);
        assert_ggev_relation(&a, &b, vl.as_ref().unwrap(), &alpha, &beta, true);
        assert_ggev_scaled(vr.as_ref().unwrap());
    }
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
