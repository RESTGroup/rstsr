//! Runnable counterpart of the linear-algebra cheat sheet
//! (`rstsr/src/docs/cheatsheet_linalg.md`), backed by the faer device.
//! Requires the `linalg` feature. Assertions use invariants (residuals and
//! round-trips) rather than magic constants, so they do not drift with faer.

// the three-way SVD tuple annotation is legible as-is
#![allow(clippy::type_complexity)]

use rstsr::prelude::*;

fn device() -> DeviceFaer {
    let mut device = DeviceFaer::default();
    device.set_default_order(RowMajor);
    device
}

/// A symmetric 2x2 matrix `[[2, 1], [1, 2]]`.
fn sym(device: &DeviceFaer) -> Tensor<f64, DeviceFaer, IxD> {
    rt::tensor_from_nested!([[2.0, 1.0], [1.0, 2.0]], device)
}

/// The identity matrix of order `n`.
fn eye(n: usize, device: &DeviceFaer) -> Tensor<f64, DeviceFaer, IxD> {
    rt::eye((n, device))
}

#[test]
fn eigh_standard() {
    let device = device();
    let a = sym(&device);
    let (w, v): (Tensor<f64, DeviceFaer, _>, Tensor<f64, DeviceFaer, _>) = rt::linalg::eigh(&a).into();

    // A V = V diag(w)  and  Vᵀ V = I
    assert!(rt::allclose(&(&a % &v), &(&v % &rt::diag(&w)), None));
    assert!(rt::allclose(&v.t() % &v, eye(2, &device), None));
}

#[test]
fn eigvalsh_standard() {
    let device = device();
    let a = sym(&device);
    // eigenvalues are ascending: [1, 3]
    let w = rt::linalg::eigvalsh(&a);
    assert!(rt::allclose(&w, &rt::tensor_from_nested!([1.0, 3.0], &device), None));
}

/// A 3x3 matrix with the real eigenvalue `1` and the conjugate pair `2 ± i`
/// (companion form).
fn companion(device: &DeviceFaer) -> Tensor<f64, DeviceFaer, IxD> {
    rt::tensor_from_nested!([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0], [0.0, 1.0, 2.0]], device)
}

/// The spectrum given as its real and imaginary parts equals `expected` (pairs
/// sorted by real then imaginary part, as the parity script prints them).
fn assert_spectrum(re: &Tensor<f64, DeviceFaer, IxD>, im: &Tensor<f64, DeviceFaer, IxD>, expected: &[(f64, f64)]) {
    let mut ws: Vec<(f64, f64)> = re.to_vec().into_iter().zip(im.to_vec()).collect();
    ws.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
    assert_eq!(ws.len(), expected.len());
    for ((re, im), (ere, eim)) in ws.into_iter().zip(expected) {
        assert!((re - ere).abs() < 1e-12 && (im - eim).abs() < 1e-12, "{re} + {im}i vs {ere} + {eim}i");
    }
}

#[test]
fn eig_standard() {
    let device = device();
    let a = companion(&device);
    let (w, v): (Tensor<_, DeviceFaer, _>, Tensor<_, DeviceFaer, _>) = rt::linalg::eig(&a).into();
    assert_eq!(w.shape(), &[3]);
    assert_eq!(v.shape(), &[3, 3]);
    assert_spectrum(&w.map(|z| z.re), &w.map(|z| z.im), &[(1.0, 0.0), (2.0, -1.0), (2.0, 1.0)]);
    // A V = V diag(w) with V = X + iY and w = lambda + i mu, in real arithmetic:
    // A X = X diag(lambda) - Y diag(mu)  and  A Y = X diag(mu) + Y diag(lambda)
    let (x, y) = (v.map(|z| z.re), v.map(|z| z.im));
    let (lam, mu) = (w.map(|z| z.re), w.map(|z| z.im));
    assert!(rt::allclose(&(&a % &x), &(&x % &rt::diag(&lam)) - &(&y % &rt::diag(&mu)), None));
    assert!(rt::allclose(&(&a % &y), &(&x % &rt::diag(&mu)) + &(&y % &rt::diag(&lam)), None));
}

#[test]
fn eigvals_standard() {
    let device = device();
    let a = companion(&device);
    let w = rt::linalg::eigvals(&a);
    assert_eq!(w.shape(), &[3]);
    assert_spectrum(&w.map(|z| z.re), &w.map(|z| z.im), &[(1.0, 0.0), (2.0, -1.0), (2.0, 1.0)]);
}

#[test]
fn eigh_generalized() {
    let device = device();
    let a = sym(&device);
    let b = eye(2, &device);
    let (w, v): (Tensor<f64, DeviceFaer, _>, Tensor<f64, DeviceFaer, _>) =
        rt::linalg::eigh((&a, &b, Lower, 1_u8 as i32)).into();
    assert!(rt::allclose(&(&a % &v), &(&v % &rt::diag(&w)), None));
}

#[test]
fn cholesky_lower() {
    let device = device();
    let a = sym(&device);
    let l = rt::linalg::cholesky(&a);
    assert!(rt::allclose(&(&l % &l.t()), &a, None));
}

#[test]
fn qr_reduced() {
    let device = device();
    let a = rt::tensor_from_nested!([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
    let (q, r): (Tensor<f64, DeviceFaer, _>, Tensor<f64, DeviceFaer, _>) = rt::linalg::qr(&a).into();
    assert!(rt::allclose(&(&q % &r), &a, None));
    assert!(rt::allclose(&q.t() % &q, eye(2, &device), None));
}

#[test]
fn svd_reconstruct() {
    let device = device();
    let a = rt::tensor_from_nested!([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
    let (u, s, vt): (Tensor<f64, DeviceFaer, _>, Tensor<f64, DeviceFaer, _>, Tensor<f64, DeviceFaer, _>) =
        rt::linalg::svd((&a, false)).into();
    assert!(rt::allclose(&(&u % &rt::diag(&s) % &vt), &a, None));
}

#[test]
fn solve_general_residual() {
    let device = device();
    let a = rt::tensor_from_nested!([[3.0, 1.0], [1.0, 2.0]], &device);
    let b = rt::tensor_from_nested!([9.0, 8.0], &device);
    let x = rt::linalg::solve_general((&a, &b));
    assert!(rt::allclose(&(&a % &x), &b, None));
}

#[test]
fn solve_triangular_residual() {
    let device = device();
    let a = rt::tensor_from_nested!([[2.0, 0.0], [1.0, 3.0]], &device); // lower triangular
    let b = rt::tensor_from_nested!([4.0, 7.0], &device);
    let x = rt::linalg::solve_triangular((&a, &b, Lower));
    assert!(rt::allclose(&(&a % &x), &b, None));
}

#[test]
fn inv_and_pinv() {
    let device = device();
    let a = rt::tensor_from_nested!([[2.0, 1.0], [1.0, 3.0]], &device);
    let ai = rt::linalg::inv(&a);
    let prod: Tensor<f64, DeviceFaer, _> = &a % &ai;
    assert!(rt::allclose(prod, eye(2, &device), None));

    let p: Tensor<f64, DeviceFaer, _> = rt::linalg::pinv(&a).pinv;
    let apa: Tensor<f64, DeviceFaer, _> = &a % &p % &a;
    assert!(rt::allclose(&apa, &a, None));
}

#[test]
fn det_and_slogdet() {
    let device = device();
    let a = rt::tensor_from_nested!([[2.0, 1.0], [1.0, 3.0]], &device);
    let det: f64 = rt::linalg::det(&a).to_scalar();
    let (sign, logabsdet) = rt::linalg::slogdet(&a).into();
    let sign_v: f64 = sign.to_scalar();
    let logabsdet_v: f64 = logabsdet.to_scalar();
    let recovered = sign_v * logabsdet_v.exp();
    assert!((det - recovered).abs() < 1e-10);
}

#[test]
fn matrix_power_rank_norm() {
    let device = device();
    let a = rt::tensor_from_nested!([[1.0, 2.0], [3.0, 4.0]], &device);
    let a2 = rt::linalg::matrix_power((&a, 2));
    assert!(rt::allclose(&a2, &(&a % &a), None));

    let singular = rt::tensor_from_nested!([[1.0, 2.0], [2.0, 4.0]], &device);
    let rank: usize = rt::linalg::matrix_rank(&singular).to_scalar();
    assert_eq!(rank, 1);

    // Frobenius norm of [[1, 2], [3, 4]] is sqrt(30)
    let fro: f64 = rt::linalg::matrix_norm(&a).to_scalar();
    assert!((fro - 30.0_f64.sqrt()).abs() < 1e-12);

    let v = rt::tensor_from_nested!([3.0, 4.0], &device);
    let vnorm: f64 = rt::linalg::vector_norm(&v).to_scalar();
    assert!((vnorm - 5.0).abs() < 1e-12);
}
