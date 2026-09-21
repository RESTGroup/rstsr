//! Issue `blas3_view_layout`: the GEMM/SYHEMM/TRSM builders passed
//! allocation-base pointers (`raw().as_ptr()`, layout offset ignored) and
//! hard-coded `ldc`/`ldb = m` to the drivers, so any f-prefer view with a
//! non-zero offset or a padded leading dimension read the wrong elements
//! and wrote outside the view (into unrelated parent elements).
//!
//! Fixed by offset-aware `as_ptr`/`as_mut_ptr` and the output's real
//! `ld_col()`. SYHEMM shares the wrapper structure but has no driver
//! implementation yet, so it is covered by the same fix without a test.
//!
//! Run notes: on the unfixed wrappers, `gemm_operand_view_with_offset`
//! returns `[[0, 3], [1, 4]]` instead of `[[1, 4], [2, 5]]` (reads the
//! window anchored at element 0), and `gemm_output_view_with_offset_and_
//! padded_ld` leaves stale parent values in `c` while overwriting the
//! parent's column 0, which the assertions catch exactly (integer data,
//! no tolerance).

use rstsr::prelude::*;
use rstsr_blas_traits::prelude::*;

#[test]
fn gemm_operand_view_with_offset() {
    let device = DeviceOpenBLAS::default();
    // f-contiguous parent P = [[0, 3], [1, 4], [2, 5]] (into_contig forces the
    // f-prefer path regardless of the device default order / col_major feature)
    let p =
        rt::tensor_from_nested!([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]], &device).into_dim::<Ix2>().into_contig(ColMajor);
    // row slice: shape [2, 2], strides [1, 3], offset 1 — an f-prefer view
    let a = p.i(1..3).into_dim::<Ix2>();
    assert!(a.f_prefer());
    let b = rt::tensor_from_nested!([[1.0, 0.0], [0.0, 1.0]], &device).into_dim::<Ix2>().into_contig(ColMajor);

    let c = GEMM::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap().into_owned();

    // a @ I = [[1, 4], [2, 5]]
    let expect = rt::tensor_from_nested!([[1.0, 4.0], [2.0, 5.0]], &device);
    assert!(rt::allclose(&c, &expect, None), "c = {c:?}");
}

#[test]
fn gemm_output_view_with_offset_and_padded_ld() {
    let device = DeviceOpenBLAS::default();
    // f-contiguous parent Q, Q[i, j] = 100 + i + 6 j
    let mut q = rt::tensor_from_nested!(
        [
            [100.0, 106.0, 112.0, 118.0],
            [101.0, 107.0, 113.0, 119.0],
            [102.0, 108.0, 114.0, 120.0],
            [103.0, 109.0, 115.0, 121.0],
            [104.0, 110.0, 116.0, 122.0],
            [105.0, 111.0, 117.0, 123.0]
        ],
        &device
    )
    .into_dim::<Ix2>()
    .into_contig(ColMajor);
    // c view: rows 0..4 of columns 1..4 — shape [4, 3], strides [1, 6], offset 6
    let c_view = q.slice_mut((0..4, 1..4)).into_dim::<Ix2>();
    assert!(c_view.f_prefer());
    assert_ne!(c_view.ld_col().unwrap(), 4); // leading dimension is padded, not m

    let a = rt::tensor_from_nested!([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0]], &device)
        .into_dim::<Ix2>()
        .into_contig(ColMajor);
    let b =
        rt::tensor_from_nested!([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], &device).into_dim::<Ix2>().into_contig(ColMajor);

    let c = GEMM::default().a(a.view()).b(b.view()).c(c_view).build().unwrap().run().unwrap().into_owned();

    // c must equal a @ b
    let expect_c =
        rt::tensor_from_nested!([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 3.0, 3.0]], &device);
    assert!(rt::allclose(&c, &expect_c, None), "c = {c:?}");
    // elements of the parent outside the view must be untouched: the expected
    // final state of the whole parent has only the c slots overwritten
    let expect_q = rt::tensor_from_nested!(
        [
            [100.0, 1.0, 0.0, 0.0],
            [101.0, 0.0, 1.0, 1.0],
            [102.0, 1.0, 1.0, 1.0],
            [103.0, 2.0, 3.0, 3.0],
            [104.0, 110.0, 116.0, 122.0],
            [105.0, 111.0, 117.0, 123.0]
        ],
        &device
    );
    assert!(rt::allclose(&q, &expect_q, None), "q = {q:?}");
}

#[test]
fn trsm_output_view_with_offset_and_padded_ld() {
    let device = DeviceOpenBLAS::default();
    // f-contiguous parent G; the b view is rows 0..2 of columns 2..4:
    // shape [2, 2], strides [1, 3], offset 6, leading dimension 3 != m
    let mut g =
        rt::tensor_from_nested!([[10.0, 13.0, 1.0, 2.0], [11.0, 14.0, 3.0, 4.0], [12.0, 15.0, 16.0, 18.0]], &device)
            .into_dim::<Ix2>()
            .into_contig(ColMajor);
    let b_view = g.slice_mut((0..2, 2..4)).into_dim::<Ix2>();
    assert!(b_view.f_prefer());
    assert_ne!(b_view.ld_col().unwrap(), 2);

    // solve L X = B with L = [[2, 0], [1, 4]] (lower), B = [[1, 2], [3, 4]]
    let a_tri = rt::tensor_from_nested!([[2.0, 0.0], [1.0, 4.0]], &device).into_dim::<Ix2>().into_contig(ColMajor);

    TRSM::default().a(a_tri.view()).b(b_view).uplo(FlagUpLo::L).build().unwrap().run().unwrap();

    // X = L^-1 B = [[0.5, 1], [0.625, 0.75]]; parent slots outside the view unchanged
    let expect =
        rt::tensor_from_nested!([[10.0, 13.0, 0.5, 1.0], [11.0, 14.0, 0.625, 0.75], [12.0, 15.0, 16.0, 18.0]], &device);
    assert!(rt::allclose(&g, &expect, None), "g = {g:?}");
}
