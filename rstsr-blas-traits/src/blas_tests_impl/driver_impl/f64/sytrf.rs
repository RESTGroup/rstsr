//! Layout-matrix, singularity and zero-size tests for the SYTRF/HETRF driver
//! (skill `lapacke-translation` §4).
//!
//! Self-validating (no pinned fingerprints): every layout case is compared
//! with the f-contig run of the same shape and `uplo` (factored triangle and
//! pivots identical), and the permutation-free block-determinant identity
//! `det(A) = prod(det(D-blocks))` is checked on the reference run.

use super::*;
use rstsr_blas_traits::blas_int;
use rstsr_blas_traits::lapack_solve::*;

const ATOL: f64 = 1e-12;
const CASES: [&str; 6] = ["f-contig", "f-padded", "c-contig", "c-padded", "non-contig", "transposed"];
const NS: [usize; 5] = [1, 2, 3, 5, 6];

/// Buffer of a layout case holding the row-major matrix `a` (`n x n`); padding
/// and the skipped entries are `NaN`, so a wrong leading dimension shows up.
fn layout_buffer(device: &DeviceBLAS, case: &str, a: &[f64], n: usize) -> Tensor<f64, DeviceBLAS, Ix2> {
    let at = |v: Vec<f64>, rows: usize, cols: usize, order: FlagOrder| -> Tensor<f64, DeviceBLAS, Ix2> {
        match order {
            RowMajor => rt::asarray((v, [rows, cols].c(), device)).into_dim::<Ix2>(),
            ColMajor => rt::asarray((v, [rows, cols].f(), device)).into_dim::<Ix2>(),
        }
    };
    // the logical matrix as a Fortran- or C-ordered vector of shape (rows, cols)
    let packed = |rows: usize, cols: usize, order: FlagOrder| -> Vec<f64> {
        let mut v = vec![f64::NAN; rows * cols];
        for i in 0..n {
            for j in 0..n {
                v[match order {
                    ColMajor => i + j * rows,
                    RowMajor => i * cols + j,
                }] = a[i * n + j];
            }
        }
        v
    };
    match case {
        "c-contig" => at(a.to_vec(), n, n, RowMajor),
        "f-contig" => at(packed(n, n, ColMajor), n, n, ColMajor),
        // padded columns (Fortran order) or padded rows (C order)
        "f-padded" => at(packed(n + 4, n + 4, ColMajor), n + 4, n + 4, ColMajor),
        "c-padded" => at(packed(n + 4, n + 4, RowMajor), n + 4, n + 4, RowMajor),
        // every second entry along both axes of a buffer of double size
        "non-contig" => {
            let mut v = vec![f64::NAN; 4 * n * n];
            for i in 0..n {
                for j in 0..n {
                    v[(2 * i) * (2 * n) + 2 * j] = a[i * n + j];
                }
            }
            at(v, 2 * n, 2 * n, RowMajor)
        },
        // the matrix stored as its transpose (a column-major (N, N) buffer has
        // the memory of a row-major (N, N) one)
        "transposed" => at(a.to_vec(), n, n, ColMajor),
        _ => panic!("unknown layout case {case}"),
    }
}

/// The logical `n x n` matrix as an immutable view of a buffer of [`layout_buffer`].
fn layout_view<'a>(
    buf: &'a Tensor<f64, DeviceBLAS, Ix2>,
    case: &str,
    n: usize,
) -> TensorView<'a, f64, DeviceBLAS, Ix2> {
    match case {
        "f-contig" | "c-contig" => buf.view(),
        "f-padded" | "c-padded" => buf.i((0..n, 0..n)).into_dim::<Ix2>(),
        "non-contig" => buf.i((slice!(0, 2 * n, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>(),
        "transposed" => buf.t(),
        _ => panic!("unknown layout case {case}"),
    }
}

/// The same view, mutable (the transposed case as a reversed-axes view).
fn layout_view_mut<'a>(
    buf: &'a mut Tensor<f64, DeviceBLAS, Ix2>,
    case: &str,
    n: usize,
) -> TensorMut<'a, f64, DeviceBLAS, Ix2> {
    match case {
        "f-contig" | "c-contig" => buf.view_mut().into_dim::<Ix2>(),
        "f-padded" | "c-padded" => buf.i_mut((0..n, 0..n)).into_dim::<Ix2>(),
        "non-contig" => buf.i_mut((slice!(0, 2 * n, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>(),
        "transposed" => buf.view_mut().into_reverse_axes().into_dim::<Ix2>(),
        _ => panic!("unknown layout case {case}"),
    }
}

/// Assert that the `NaN` entries of the fresh layout buffer (padding and
/// skipped entries) are still `NaN` in `buf`.
fn check_padding(buf: &Tensor<f64, DeviceBLAS, Ix2>, device: &DeviceBLAS, case: &str, a: &[f64], n: usize, at: &str) {
    let fresh = layout_buffer(device, case, a, n);
    for (idx, (&v, &f)) in buf.raw().iter().zip(fresh.raw().iter()).enumerate() {
        if f.is_nan() {
            assert!(v.is_nan(), "padding touched at {idx}, {at}");
        }
    }
}

/// Symmetric matrix of shape (`n`, `n`) from the manifest data (both
/// triangles hold the same values, so lower and upper runs read the same
/// matrix).
fn sym_source(src: &[f64], n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = src[i * n + j] + src[j * n + i];
        }
    }
    a
}

/// Block sizes of the factorization decoded from LAPACK's `ipiv`: `((start,
/// size), ...)` covering all `n` rows in order.
fn block_starts(ipiv: &[blas_int], lower: bool) -> Vec<(usize, usize)> {
    let n = ipiv.len();
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < n {
        let two = if lower { ipiv[i] < 0 } else { i + 1 < n && ipiv[i] < 0 };
        if two {
            blocks.push((i, 2));
            i += 2;
        } else {
            blocks.push((i, 1));
            i += 1;
        }
    }
    blocks
}

/// The packed factor of one triangle (`Lower`: `j <= i`, `Upper`: `j >= i`)
/// and the pivots, for the layout-parity comparison.
fn packed_triangle(lu: &TensorView<'_, f64, DeviceBLAS, Ix2>, lower: bool) -> Vec<f64> {
    let n = lu.shape()[0];
    let mut v = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let inside = if lower { j <= i } else { j >= i };
            if inside {
                v.push(lu[[i, j]]);
            }
        }
    }
    v
}

/// `det(A) = prod(det(D-blocks))` on the packed factor: permutation-free,
/// so the identity holds for the compact `?SYTRF` output directly. The
/// 2x2 block `[[d11, b], [b, d22]]` is symmetric (real case), and only the
/// selected triangle holds the factor's `b` — the other triangle of the
/// buffer still holds the input, moved around by the interchanges.
fn check_block_determinant(
    lu: &TensorView<'_, f64, DeviceBLAS, Ix2>,
    ipiv: &[blas_int],
    a: &[f64],
    lower: bool,
    at: &str,
) {
    let n = lu.shape()[0];
    let mut det_prod = 1.0;
    for (k, size) in block_starts(ipiv, lower) {
        det_prod *= match size {
            1 => lu[[k, k]],
            2 => {
                let b = if lower { lu[[k + 1, k]] } else { lu[[k, k + 1]] };
                lu[[k, k]] * lu[[k + 1, k + 1]] - b * b
            },
            _ => unreachable!(),
        };
    }
    let mat = rt::asarray((a.to_vec(), [n, n].c(), lu.device())).into_dim::<Ix2>();
    let det: f64 = rt::linalg::det(mat.view()).to_scalar();
    assert!(
        (det_prod - det).abs() <= 1e-10 * det.abs().max(1.0),
        "block determinants {det_prod} != det(A) {det}, {at}"
    );
}

#[test]
fn test_dsytrf_layouts() {
    let src = get_vec::<f64>('b');
    for n in NS {
        let a = sym_source(&src, n);
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            for lower in [true, false] {
                let uplo = if lower { Lower } else { Upper };
                let at_uplo = format!("{n}, {order:?}, {}", if lower { "lower" } else { "upper" });
                let mut reference: Option<(Vec<f64>, Vec<blas_int>)> = None;
                for case in CASES {
                    let at = format!("{at_uplo}, {case}");
                    let buf = layout_buffer(&device, case, &a, n);
                    let buf_ptr = buf.raw().as_ptr();
                    let view = layout_view(&buf, case, n);
                    assert_eq!(view.shape(), &[n, n], "view shape, {at}");
                    // the view really is the logical matrix
                    for i in 0..n {
                        for j in 0..n {
                            assert_eq!(view[[i, j]], a[i * n + j], "view value ({i}, {j}), {at}");
                        }
                    }
                    let (lu, ipiv, info) =
                        DSYTRF::default().a(view).uplo(uplo).build().unwrap().run_with_info().unwrap();
                    assert_eq!(info, 0, "info, {at}");
                    // a read-only view is factored into a fresh buffer
                    assert_ne!(lu.view().raw().as_ptr(), buf_ptr, "fresh buffer, {at}");
                    let packed = packed_triangle(&lu.view(), lower);
                    let ipiv: Vec<blas_int> = ipiv.to_vec();
                    match &reference {
                        None => {
                            reference = Some((packed.clone(), ipiv.clone()));
                            check_block_determinant(&lu.view(), &ipiv, &a, lower, &at);
                        },
                        Some((r_packed, r_ipiv)) => {
                            assert_eq!(ipiv, *r_ipiv, "pivots, {at}");
                            assert!(
                                packed.iter().zip(r_packed.iter()).all(|(x, y)| (x - y).abs() < ATOL),
                                "factored triangle, {at}"
                            );
                        },
                    }
                    // a mutable view is written through: the returned factor is
                    // the input buffer itself (in place for an in-order layout,
                    // written back from a work buffer otherwise)
                    let mut buf = buf;
                    let view = layout_view_mut(&mut buf, case, n);
                    let (lu, ipiv_mut, info) =
                        DSYTRF::default().a(view).uplo(uplo).build().unwrap().run_with_info().unwrap();
                    assert_eq!(info, 0, "info, mutable {at}");
                    assert_eq!(ipiv_mut.to_vec(), reference.as_ref().unwrap().1, "pivots, mutable {at}");
                    assert_eq!(lu.view().raw().as_ptr(), buf_ptr, "buffer identity, mutable {at}");
                    drop(lu);
                    let visible = layout_view(&buf, case, n);
                    let packed_ref = &reference.as_ref().unwrap().0;
                    for (idx, &v) in packed_triangle(&visible, lower).iter().enumerate() {
                        assert!((v - packed_ref[idx]).abs() < ATOL, "in-place factor {idx}, {at}");
                    }
                    // the padding is not touched
                    check_padding(&buf, &device, case, &a, n, &at);
                }
            }
        }
    }
}

/// The builder default `uplo` follows the device order (`RowMajor => Lower`,
/// `ColMajor => Upper`).
#[test]
fn test_dsytrf_default_uplo() {
    let src = get_vec::<f64>('b');
    let n = 5;
    let a = sym_source(&src, n);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let buf = layout_buffer(&device, "c-contig", &a, n);
        let (_, piv_default, _) = DSYTRF::default().a(buf.view()).build().unwrap().run_with_info().unwrap();
        let lower_default = match order {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        let (_, piv_explicit, _) =
            DSYTRF::default().a(buf.view()).uplo(lower_default).build().unwrap().run_with_info().unwrap();
        assert_eq!(piv_default.to_vec(), piv_explicit.to_vec(), "default uplo, {order:?}");
    }
}

#[test]
fn test_dsytrf_singular() {
    // rank-2 symmetric: D has an exactly-zero pivot
    let a = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0];
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        for lower in [true, false] {
            let uplo = if lower { Lower } else { Upper };
            let at = format!("{order:?}, {}", if lower { "lower" } else { "upper" });
            let buf = layout_buffer(&device, "c-contig", &a, 3);
            // run: the singularity is an error
            let err = DSYTRF::default().a(buf.view()).uplo(uplo).build().unwrap().run().err().expect("run must error");
            assert!(format!("{err}").contains("info = "), "run error message, {at}");
            // run_with_info: complete factors plus the index of the zero pivot
            let (lu, ipiv, info) = DSYTRF::default().a(buf.view()).uplo(uplo).build().unwrap().run_with_info().unwrap();
            assert!(info > 0 && info <= 3, "info = {info}, {at}");
            assert!(lu.view().raw().iter().all(|v| v.is_finite()), "factors finite, {at}");
            // the block decode covers all rows
            let covered: usize = block_starts(&ipiv.to_vec(), lower).iter().map(|&(_, s)| s).sum();
            assert_eq!(covered, 3, "blocks cover all rows, {at}");
        }
    }
}

#[test]
fn test_dsytrf_zero_size() {
    let device = DeviceBLAS::default();
    let a = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
    let (lu, ipiv, info) = DSYTRF::default().a(a.view()).build().unwrap().run_with_info().unwrap();
    assert_eq!(info, 0, "info");
    assert_eq!(lu.view().shape(), &[0, 0], "lu shape");
    assert_eq!(ipiv.shape(), &[0], "ipiv shape");
}
