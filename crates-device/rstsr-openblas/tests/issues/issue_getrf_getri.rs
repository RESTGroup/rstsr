//! GETRF/GETRI contracts: pivot sizing on wide matrices, LU values, and
//! rejection of a pivot vector shorter than the matrix order in GETRI.

use rstsr::prelude::*;
use rstsr_blas_traits::prelude::*;

#[test]
fn getrf_wide_matrix_values() {
    let device = DeviceOpenBLAS::default();
    let mut a: Tensor<f64, _, Ix2> =
        rt::tensor_from_nested!([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device).into_dim::<Ix2>();

    let (a_lu, ipiv) = GETRF::default().a(a.view_mut()).build().unwrap().run().unwrap();
    let a_lu = a_lu.into_owned();

    assert_eq!(*a_lu.shape(), [2, 4]);
    // LAPACK xGETRF defines ipiv over min(m, n) entries
    assert_eq!(ipiv.size(), 2);
    // partial pivoting swaps rows 0 and 1 (|5| > |1); 0-based pivots
    let piv = ipiv.raw().to_vec();
    assert_eq!(piv, vec![1, 1]);

    // PA = LU with L = [[1, 0], [0.2, 1]] and U = [[5, 6, 7, 8], [0, 0.8, 1.6, 2.4]]
    let expect_lu = rt::tensor_from_nested!([[5.0, 6.0, 7.0, 8.0], [0.2, 0.8, 1.6, 2.4]], &device);
    assert!(rt::allclose(&a_lu, &expect_lu, None), "a_lu = {a_lu:?}");
}

#[test]
fn getri_rejects_short_ipiv() {
    let device = DeviceOpenBLAS::default();
    let mut a: Tensor<f64, _, Ix2> = rt::tensor_from_nested!([[2.0, 0.0], [0.0, 4.0]], &device).into_dim::<Ix2>();
    // pivot vector shorter than n = 2: LAPACK xGETRI reads ipiv[0..n]
    let ipiv_full = rt::asarray((vec![1 as blas_int, 1, 1], &device)).into_dim::<Ix1>();
    let ipiv = ipiv_full.i(0..1).into_dim::<Ix1>();

    let result = GETRI::default().a(a.view_mut()).ipiv(ipiv.view()).build().unwrap().run();
    assert!(result.is_err(), "GETRI with a 1-element ipiv on a 2x2 matrix must error");
}
