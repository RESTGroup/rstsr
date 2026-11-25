use rstsr::prelude::*;
use rstsr_blas_traits::prelude::*;

fn device() -> DeviceOpenBLAS {
    DeviceOpenBLAS::default()
}

/* #region GESV / SYSV */

fn lu_of_gesv(a: &Tensor<f64, DeviceOpenBLAS, Ix2>, nrhs: usize) -> Tensor<f64, DeviceOpenBLAS, Ix2> {
    let device = device();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([a.nrow(), nrhs].f(), &device)).into_dim::<Ix2>();
    let (lu, _, _) = DGESV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    lu.into_owned()
}

fn lu_of_sysv(a: &Tensor<f64, DeviceOpenBLAS, Ix2>, nrhs: usize) -> Tensor<f64, DeviceOpenBLAS, Ix2> {
    let device = device();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([a.nrow(), nrhs].f(), &device)).into_dim::<Ix2>();
    let (lu, _, _) = DSYSV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    lu.into_owned()
}

/// OpenBLAS's optimized `dgesv` returned before the factorization when
/// `nrhs == 0` up to 0.3.31; 0.3.32 restored the reference behavior.
fn openblas_factors_zero_rhs() -> bool {
    let config = rstsr_openblas::openblas_config();
    let mut version = config.split_whitespace().nth(1).unwrap_or("").split('.').map(|x| x.parse::<u32>().unwrap_or(0));
    match (version.next(), version.next(), version.next()) {
        (Some(major), Some(minor), Some(patch)) => (major, minor, patch) >= (0, 3, 32),
        _ => true,
    }
}

#[test]
fn gesv_empty_n() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();

    let (a, ipiv, b) = DGESV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let (a, b) = (a.into_owned(), b.into_owned());
    assert_eq!(a.shape(), &[0, 0]);
    assert_eq!(b.shape(), &[0, 0]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn gesv_empty_n_nonempty_rhs() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 3].f(), &device())).into_dim::<Ix2>();

    let (a, ipiv, b) = DGESV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let (a, b) = (a.into_owned(), b.into_owned());
    assert_eq!(a.shape(), &[0, 0]);
    assert_eq!(b.shape(), &[0, 3]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn gesv_zero_rhs_still_factors_a() {
    if !openblas_factors_zero_rhs() {
        eprintln!("skipped: this OpenBLAS does not factor A when NRHS == 0");
        return;
    }
    let device = device();
    let a = rt::tensor_from_nested!([[4.0f64, 3.0], [6.0, 3.0]], &device).into_dim::<Ix2>();

    // nrhs == 0 still factors A; the factors must match the nrhs == 1 call
    let lu_empty = lu_of_gesv(&a, 0);
    let lu_full = lu_of_gesv(&a, 1);
    assert!(rt::allclose(&lu_empty, &lu_full, None), "lu_empty = {lu_empty:?}, lu_full = {lu_full:?}");
}

#[test]
fn gesv_zero_rhs_singular_a_errors() {
    if !openblas_factors_zero_rhs() {
        eprintln!("skipped: this OpenBLAS does not factor A when NRHS == 0");
        return;
    }
    let device = device();
    let a = rt::tensor_from_nested!([[1.0f64, 2.0], [2.0, 4.0]], &device).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([2usize, 0].f(), &device)).into_dim::<Ix2>();

    assert!(DGESV::default().a(a.view()).b(b.view()).build().unwrap().run().is_err());
}

#[test]
fn sysv_empty_n() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();

    let (a, ipiv, b) = DSYSV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let (a, b) = (a.into_owned(), b.into_owned());
    assert_eq!(a.shape(), &[0, 0]);
    assert_eq!(b.shape(), &[0, 0]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn sysv_zero_rhs_still_factors_a() {
    let device = device();
    let a = rt::tensor_from_nested!([[4.0f64, 3.0], [3.0, 3.0]], &device).into_dim::<Ix2>();

    let lu_empty = lu_of_sysv(&a, 0);
    let lu_full = lu_of_sysv(&a, 1);
    assert!(rt::allclose(&lu_empty, &lu_full, None), "lu_empty = {lu_empty:?}, lu_full = {lu_full:?}");
}

#[test]
fn sysv_zero_rhs_singular_a_errors() {
    let device = device();
    let a = rt::tensor_from_nested!([[1.0f64, 2.0], [2.0, 4.0]], &device).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([2usize, 0].f(), &device)).into_dim::<Ix2>();

    assert!(DSYSV::default().a(a.view()).b(b.view()).build().unwrap().run().is_err());
}

/* #endregion */

/* #region GETRF / GETRI / POTRF */

#[test]
fn getrf_empty_rows() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 3].f(), &device())).into_dim::<Ix2>();
    let (a, ipiv) = DGETRF::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(a.shape(), &[0, 3]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn getrf_empty_cols() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([3usize, 0].f(), &device())).into_dim::<Ix2>();
    let (a, ipiv) = DGETRF::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(a.shape(), &[3, 0]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn getri_empty() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let ipiv: Tensor<blas_int, DeviceOpenBLAS, Ix1> = rt::zeros(([0usize].f(), &device)).into_dim::<Ix1>();

    let a = DGETRI::default().a(a.view()).ipiv(ipiv.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(a.shape(), &[0, 0]);
}

#[test]
fn potrf_empty() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let a = DPOTRF::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(a.shape(), &[0, 0]);
}

/* #endregion */

/* #region EIGH */

#[test]
fn syev_empty() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let (w, a) = DSYEV::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(w.shape(), &[0]);
    assert_eq!(a.shape(), &[0, 0]);
}

#[test]
fn syevd_empty() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let (w, a) = DSYEVD::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(w.shape(), &[0]);
    assert_eq!(a.shape(), &[0, 0]);
}

#[test]
fn sygv_empty() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let (w, a) = DSYGV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(w.shape(), &[0]);
    assert_eq!(a.shape(), &[0, 0]);
}

#[test]
fn sygvd_empty() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let (w, a) = DSYGVD::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(w.shape(), &[0]);
    assert_eq!(a.shape(), &[0, 0]);
}

/* #endregion */

/* #region SVD */

/// Empty operand with degenerate strides `[1, 1]`: the leading dimension must
/// not be derived from the layout before the empty shortcut.
fn degenerate_empty(device: &DeviceOpenBLAS) -> Tensor<f64, DeviceOpenBLAS, Ix2> {
    let buf: Vec<f64> = Vec::new();
    rt::asarray((buf, [5usize, 0].f(), device)).into_reverse_axes().into_dim::<Ix2>()
}

#[test]
fn gesvd_empty_degenerate_strides() {
    let device = device();
    let a = degenerate_empty(&device);
    assert_eq!(a.shape(), &[0, 5]);

    let (s, u, vt, superb) = DGESVD::default().a(a.view()).build().unwrap().run().unwrap();
    assert_eq!(s.shape(), &[0]);
    assert_eq!(u.unwrap().shape(), &[0, 0]);
    assert_eq!(vt.unwrap().shape(), &[5, 5]);
    assert_eq!(superb.shape(), &[0]);
}

#[test]
fn gesdd_empty_degenerate_strides() {
    let device = device();
    let a = degenerate_empty(&device);
    assert_eq!(a.shape(), &[0, 5]);

    let (s, u, vt) = DGESDD::default().a(a.view()).build().unwrap().run().unwrap();
    assert_eq!(s.shape(), &[0]);
    assert_eq!(u.unwrap().shape(), &[0, 0]);
    assert_eq!(vt.unwrap().shape(), &[5, 5]);
}

/* #endregion */

/* #region GELSD / lstsq */

#[test]
fn lstsq_empty_rows() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 5].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix1> = rt::zeros(([0usize].f(), &device)).into_dim::<Ix1>();
    let result = rt::linalg::lstsq((a.view(), b.view()));
    assert_eq!(result.x.shape(), &[5]);
    assert_eq!(result.s.shape(), &[0]);
    assert!(result.rank.shape().is_empty());
    assert_eq!(result.residuals.shape(), &[0]);
    assert_eq!(result.rank.raw()[0], 0);
    assert!(result.x.raw().iter().all(|x| *x == 0.0));
}

#[test]
fn lstsq_empty_cols() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([5usize, 0].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix1> =
        rt::asarray((vec![1.0, 2.0, 3.0, 4.0, 5.0], [5].c(), &device)).into_dim::<Ix1>();
    let result = rt::linalg::lstsq((a.view(), b.view()));
    assert_eq!(result.x.shape(), &[0]);
    assert_eq!(result.s.shape(), &[0]);
    assert!(result.residuals.shape().is_empty());
    assert_eq!(result.rank.raw()[0], 0);
    assert_eq!(result.residuals.raw()[0], 55.0);

    // the residual of every right-hand side is synthesized without a driver call
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> =
        rt::asarray((vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [5usize, 2].f(), &device))
            .into_dim::<Ix2>();
    let result = rt::linalg::lstsq((a.view(), b.view()));
    assert_eq!(result.x.shape(), &[0, 2]);
    assert_eq!(result.residuals.shape(), &[2]);
    assert_eq!(result.residuals.raw()[0], 55.0);
    assert_eq!(result.residuals.raw()[1], 330.0);
}

#[test]
fn lstsq_zero_rhs() {
    let device = device();
    // a Vandermonde matrix of six distinct rows has full column rank
    let a_data: Vec<f64> = (0..6).flat_map(|i| (0..4).map(move |j| (i as f64 + 1.0).powi(j))).collect();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::asarray((a_data, [6usize, 4].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([6usize, 0].f(), &device)).into_dim::<Ix2>();
    let result = rt::linalg::lstsq((a.view(), b.view()));
    assert_eq!(result.x.shape(), &[4, 0]);
    assert_eq!(result.s.shape(), &[4]);
    assert_eq!(result.residuals.shape(), &[0]);
    assert_eq!(result.rank.raw()[0], 4);
}

#[test]
fn lstsq_empty_batch() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix3> = rt::zeros(([0usize, 4, 5], &device)).into_dim::<Ix3>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix3> = rt::zeros(([0usize, 4, 2], &device)).into_dim::<Ix3>();
    let result = rt::linalg::lstsq((a.view(), b.view()));
    assert_eq!(result.x.shape(), &[0, 5, 2]);
    assert_eq!(result.s.shape(), &[0, 4]);
    assert_eq!(result.rank.shape(), &[0]);
    assert_eq!(result.residuals.shape(), &[0, 0]);
}

/* #endregion */
