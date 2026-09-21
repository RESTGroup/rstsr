//! SVD on an empty matrix: `superb` is sized `min(m, n) - 1`, which used to
//! underflow to `usize::MAX` for a zero-sized dimension and surface as a bogus
//! allocation failure instead of an (empty) result.

use rstsr::prelude::*;
use rstsr_blas_traits::prelude::*;

#[test]
fn gesvd_empty_matrix() {
    let device = DeviceOpenBLAS::default();
    let a: Tensor<f64, _, Ix2> = rt::asarray((vec![], [0, 3], &device)).into_dim::<Ix2>();

    let (s, u, vt, _superb) =
        GESVD::default().a(a.view()).full_matrices(false).compute_uv(false).build().unwrap().run().unwrap();

    assert_eq!(s.size(), 0);
    assert!(u.is_none());
    assert!(vt.is_none());
}

#[test]
fn gesvd_empty_matrix_with_uv() {
    let device = DeviceOpenBLAS::default();
    let a: Tensor<f64, _, Ix2> = rt::asarray((vec![], [0, 3], &device)).into_dim::<Ix2>();

    // jobz 'S': U is [m, min(m, n)], VT is [min(m, n), n]
    let (s, u, vt, _superb) =
        GESVD::default().a(a.view()).full_matrices(false).compute_uv(true).build().unwrap().run().unwrap();

    assert_eq!(s.size(), 0);
    let (u, vt) = (u.unwrap(), vt.unwrap());
    assert_eq!(*u.shape(), [0, 0]);
    assert_eq!(*vt.shape(), [0, 3]);
}

#[test]
fn gesdd_empty_matrix_with_uv() {
    let device = DeviceOpenBLAS::default();
    let a: Tensor<f64, _, Ix2> = rt::asarray((vec![], [0, 3], &device)).into_dim::<Ix2>();

    let (s, u, vt) = GESDD::default().a(a.view()).full_matrices(false).compute_uv(true).build().unwrap().run().unwrap();

    assert_eq!(s.size(), 0);
    let (u, vt) = (u.unwrap(), vt.unwrap());
    assert_eq!(*u.shape(), [0, 0]);
    assert_eq!(*vt.shape(), [0, 3]);
}
