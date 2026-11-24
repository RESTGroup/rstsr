use super::*;
use duplicate::duplicate_item;
use rstsr_blas_traits::lapack_qr::*;

#[test]
fn test_dorgqr() {
    let device = DeviceBLAS::default();
    let a_vec = &get_vec::<f64>('a')[..1024 * 512];
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // reduced Q (k = n = 512)
    let (h, tau) = DGEQRF::default().a(a.view()).build().unwrap().run().unwrap();
    let q = DORGQR::default().a(h.view()).tau(tau.view()).build().unwrap().run().unwrap();
    let q = q.into_owned();
    assert_eq!(q.shape(), &[1024, 512]);
    assert!((fingerprint(&q.abs()) - -3.816695392252389).abs() < 1e-8);

    // k < n on the square wide form: only the leading k columns are read
    let a_all = &get_vec::<f64>('a')[..512 * 512];
    let a_sq = rt::asarray((a_all, [512, 512].c(), &device)).into_dim::<Ix2>();
    let (h, tau) = DGEQRF::default().a(a_sq.view()).build().unwrap().run().unwrap();
    let q_all = DORGQR::default().a(h.view()).tau(tau.view()).build().unwrap().run().unwrap().into_owned();
    let (h, tau) = DGEQRF::default().a(a_sq.view()).build().unwrap().run().unwrap();
    let q_k = DORGQR::default().a(h.view()).tau(tau.view()).k(300usize).build().unwrap().run().unwrap().into_owned();
    assert_eq!(q_k.shape(), &[512, 512]);
    let lead = q_k.view().i((.., 0..300)).into_dim::<Ix2>().into_owned();
    let lead_ref = q_all.view().i((.., 0..300)).into_dim::<Ix2>().into_owned();
    assert!(rt::allclose(lead.view().abs(), lead_ref.view().abs(), Some(1e-12)));
}

#[duplicate_item(
    name                       case          ;
    [test_dorgqr_layout_f_contig   ] ["f-contig"];
    [test_dorgqr_layout_f_padded   ] ["f-padded"];
    [test_dorgqr_layout_c_contig   ] ["c-contig"];
    [test_dorgqr_layout_c_padded   ] ["c-padded"];
    [test_dorgqr_layout_non_contig ] ["non-contig"];
)]
#[test]
fn name() {
    let device = DeviceBLAS::default();
    for (m, n) in [(5usize, 8usize), (8, 5), (6, 6), (1, 7), (7, 1)] {
        let a = super::geqrf::make_case_f64(&device, case, m, n);
        super::geqrf::assert_qr_pair_correct(&device, &a);
    }
}

#[test]
fn test_dorgqr_rejects_wide_input() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('b')[..5 * 8].to_vec(), [5, 8].c(), &device)).into_dim::<Ix2>();
    let (h, tau) = DGEQRF::default().a(a.view()).build().unwrap().run().unwrap();
    // LAPACK requires m >= n >= k
    assert!(DORGQR::default().a(h.view()).tau(tau.view()).build().unwrap().run().is_err());
    let a_q = h.view().i((.., 0..5)).into_dim::<Ix2>().into_owned();
    let q = DORGQR::default().a(a_q.view()).tau(tau.view()).build().unwrap().run().unwrap();
    assert_eq!(q.view().shape(), &[5, 5]);
}

#[test]
fn test_dorgqr_zero_size() {
    let device = DeviceBLAS::default();
    for (m, n) in [(0usize, 4usize), (4, 0), (0, 0)] {
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([m, n], &device)).into_dim::<Ix2>();
        let (h, tau) = DGEQRF::default().a(a.view()).build().unwrap().run().unwrap();
        let k = m.min(n);
        let a_q = h.view().i((.., 0..k)).into_dim::<Ix2>().into_owned();
        let q = DORGQR::default().a(a_q.view()).tau(tau.view()).build().unwrap().run().unwrap();
        assert_eq!(q.view().shape(), &[m, k]);
    }
}
