use super::*;
use duplicate::duplicate_item;
use rstsr_blas_traits::lapack_qr::*;

const ATOL: f64 = 1e-10;

/// Upper triangle of the factor (the R factor), as an owned `(k, n)` tensor.
pub(super) fn upper_part(h: &Tensor<f64, DeviceBLAS, Ix2>, k: usize) -> Tensor<f64, DeviceBLAS, Ix2> {
    let mut r: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([k, h.ncol()], h.device())).into_dim::<Ix2>();
    for i in 0..k {
        r.i_mut((i, i..h.ncol())).assign_f(h.view().i((i, i..h.ncol()))).unwrap();
    }
    r
}

/// R with a real-positive diagonal (QR is unique up to a sign per column).
pub(super) fn canonical_r(r: &Tensor<f64, DeviceBLAS, Ix2>) -> Tensor<f64, DeviceBLAS, Ix2> {
    let mut out = r.clone();
    for i in 0..r.nrow() {
        let s = if r[[i, i]] < 0.0 { -1.0 } else { 1.0 };
        out.i_mut((i, i..r.ncol())).assign_f(r.view().i((i, i..r.ncol())) * s).unwrap();
    }
    out
}

/// Logical `(m, n)` matrix laid out per `case` (see `svd_layout.rs`).
pub(super) fn make_case_f64(device: &DeviceBLAS, case: &str, m: usize, n: usize) -> Tensor<f64, DeviceBLAS, Ix2> {
    let v = &get_vec::<f64>('b')[..(m + 4) * (n + 4)];
    match case {
        "f-contig" => rt::asarray((v, [m, n].f(), device)).into_dim::<Ix2>().into_owned(),
        "f-padded" => {
            let buf = rt::asarray((v, [m + 4, n + 4].f(), device)).into_dim::<Ix2>().into_owned();
            buf.i((0..m, 0..n)).into_dim::<Ix2>().into_owned()
        },
        "c-contig" => rt::asarray((v, [m, n].c(), device)).into_dim::<Ix2>().into_owned(),
        "c-padded" => {
            let buf = rt::asarray((v, [m + 4, n + 4].c(), device)).into_dim::<Ix2>().into_owned();
            buf.i((0..m, 0..n)).into_dim::<Ix2>().into_owned()
        },
        "non-contig" => {
            let buf = rt::asarray((v, [m + 4, n + 4].c(), device)).into_dim::<Ix2>().into_owned();
            let strided = buf.i((slice!(None, None, 2), slice!(None, None, 2))).into_dim::<Ix2>().into_owned();
            strided.i((0..m, 0..n)).into_dim::<Ix2>().into_owned()
        },
        _ => panic!("unknown layout case {case}"),
    }
}

/// Factor `a`, regenerate the thin Q, and check `Q R = A` and `Qᵀ Q = I`.
pub(super) fn assert_qr_pair_correct(device: &DeviceBLAS, a: &Tensor<f64, DeviceBLAS, Ix2>) {
    let (m, n) = (a.nrow(), a.ncol());
    let k = m.min(n);
    let (h, tau) = DGEQRF::default().a(a.view()).build().unwrap().run().unwrap();
    let h = h.into_owned();
    assert_eq!(h.shape(), &[m, n]);
    assert_eq!(tau.shape(), &[k]);
    if k == 0 {
        return;
    }
    // ORGQR accepts m x k input; the wide case needs the leading k columns only
    let a_q = h.view().i((.., 0..k)).into_dim::<Ix2>().into_owned();
    let q = DORGQR::default().a(a_q.view()).tau(tau.view()).build().unwrap().run().unwrap().into_owned();
    assert_eq!(q.shape(), &[m, k]);

    let r = upper_part(&h, k);
    let diff = (q.view() % r.view()) - a.view();
    let scale: f64 = (a.view().abs() + 1.0).sum_all();
    assert!((diff.abs()).sum_all() < ATOL * scale, "Q R != A ({m}x{n})");

    let qt = q.view().t().into_owned();
    let g = qt.view() % q.view();
    let eye: Tensor<f64, DeviceBLAS, _> = rt::eye((k, device));
    assert!(((g.abs() - eye.view()).abs()).sum_all() < ATOL * (k * k) as f64, "Q not orthonormal ({m}x{n})");
}

#[test]
fn test_dgeqrf() {
    let device = DeviceBLAS::default();
    let a_vec = &get_vec::<f64>('a')[..1024 * 512];
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // R (sign-canonicalized) matches numpy's
    let (h, tau) = DGEQRF::default().a(a.view()).build().unwrap().run().unwrap();
    assert_eq!(tau.shape(), &[512]);
    let r = canonical_r(&upper_part(&h.into_owned(), 512));
    assert!((fingerprint(&r) - -245.99004994559718).abs() < 1e-8);
}

#[duplicate_item(
    name                       case          ;
    [test_dgeqrf_layout_f_contig   ] ["f-contig"];
    [test_dgeqrf_layout_f_padded   ] ["f-padded"];
    [test_dgeqrf_layout_c_contig   ] ["c-contig"];
    [test_dgeqrf_layout_c_padded   ] ["c-padded"];
    [test_dgeqrf_layout_non_contig ] ["non-contig"];
)]
#[test]
fn name() {
    let device = DeviceBLAS::default();
    for (m, n) in [(5usize, 8usize), (8, 5), (6, 6), (1, 7), (7, 1)] {
        let a = make_case_f64(&device, case, m, n);
        assert_qr_pair_correct(&device, &a);
    }
}

#[test]
fn test_dgeqrf_in_place() {
    let mut device = DeviceBLAS::default();
    for order in [RowMajor, ColMajor] {
        device.set_default_order(order);
        // owned, in the device default order: written through, buffer kept
        let mut a: Tensor<f64, DeviceBLAS, Ix2> =
            rt::asarray((get_vec::<f64>('b')[..6 * 4].to_vec(), [6, 4], &device)).into_dim::<Ix2>();
        let snapshot = a.clone();
        let ptr = a.view().as_ptr();
        let a_ref: TensorReference<'_, f64, DeviceBLAS, Ix2> = a.view_mut().into();
        let (h, _) = DGEQRF::default().a(a_ref).build().unwrap().run().unwrap();
        assert_eq!(h.view().as_ptr(), ptr, "owned in-order input must be factored in place");
        assert!(!rt::allclose(h.view(), snapshot.view(), None), "factorization must write through");

        // a view is never written through
        let a = rt::asarray((get_vec::<f64>('b')[..6 * 4].to_vec(), [6, 4], &device)).into_dim::<Ix2>();
        let snapshot = a.clone();
        let ptr = a.view().as_ptr();
        let (h, _) = DGEQRF::default().a(a.view()).build().unwrap().run().unwrap();
        assert_ne!(h.view().as_ptr(), ptr);
        assert!(rt::allclose(&a, &snapshot, None));
    }
}

#[test]
fn test_dgeqrf_zero_size() {
    let device = DeviceBLAS::default();
    for (m, n) in [(0usize, 4usize), (4, 0), (0, 0)] {
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([m, n], &device)).into_dim::<Ix2>();
        let (h, tau) = DGEQRF::default().a(a.view()).build().unwrap().run().unwrap();
        assert_eq!(h.view().shape(), &[m, n]);
        assert_eq!(tau.shape(), &[0]);
    }
}
