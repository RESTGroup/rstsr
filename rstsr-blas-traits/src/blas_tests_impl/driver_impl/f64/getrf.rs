//! Layout-matrix, singularity and zero-size tests for the GETRF driver
//! (skill `lapacke-translation` §4).
//!
//! Self-validating (reconstruction `P A = L U` from the packed factors and the
//! pivots) so every layout case is checked against identical semantics without
//! pinned fingerprints.

use super::*;
use rstsr_blas_traits::blas_int;
use rstsr_blas_traits::lapack_solve::*;

const ATOL: f64 = 1e-12;
const CASES: [&str; 6] = ["f-contig", "f-padded", "c-contig", "c-padded", "non-contig", "transposed"];
const SHAPES: [(usize, usize); 6] = [(1, 1), (6, 6), (8, 5), (5, 8), (4, 1), (1, 4)];

/// Buffer of a layout case holding the row-major matrix `a` (`m x n`); padding
/// and the skipped entries are `NaN`, so a wrong leading dimension shows up.
fn layout_buffer(device: &DeviceBLAS, case: &str, a: &[f64], m: usize, n: usize) -> Tensor<f64, DeviceBLAS, Ix2> {
    let at = |v: Vec<f64>, rows: usize, cols: usize, order: FlagOrder| -> Tensor<f64, DeviceBLAS, Ix2> {
        match order {
            RowMajor => rt::asarray((v, [rows, cols].c(), device)).into_dim::<Ix2>(),
            ColMajor => rt::asarray((v, [rows, cols].f(), device)).into_dim::<Ix2>(),
        }
    };
    // the logical matrix as a Fortran- or C-ordered vector of shape (rows, cols)
    let packed = |rows: usize, cols: usize, order: FlagOrder| -> Vec<f64> {
        let mut v = vec![f64::NAN; rows * cols];
        for i in 0..m {
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
        "c-contig" => at(a.to_vec(), m, n, RowMajor),
        "f-contig" => at(packed(m, n, ColMajor), m, n, ColMajor),
        // padded columns (Fortran order) or padded rows (C order)
        "f-padded" => at(packed(m + 4, n + 4, ColMajor), m + 4, n + 4, ColMajor),
        "c-padded" => at(packed(m + 4, n + 4, RowMajor), m + 4, n + 4, RowMajor),
        // every second entry along both axes of a buffer of double size
        "non-contig" => {
            let mut v = vec![f64::NAN; 4 * m * n];
            for i in 0..m {
                for j in 0..n {
                    v[(2 * i) * (2 * n) + 2 * j] = a[i * n + j];
                }
            }
            at(v, 2 * m, 2 * n, RowMajor)
        },
        // the matrix stored as its transpose (a column-major (N, M) buffer has
        // the memory of a row-major (M, N) one)
        "transposed" => at(a.to_vec(), n, m, ColMajor),
        _ => panic!("unknown layout case {case}"),
    }
}

/// The logical `m x n` matrix as an immutable view of a buffer of [`layout_buffer`].
fn layout_view<'a>(
    buf: &'a Tensor<f64, DeviceBLAS, Ix2>,
    case: &str,
    m: usize,
    n: usize,
) -> TensorView<'a, f64, DeviceBLAS, Ix2> {
    match case {
        "f-contig" | "c-contig" => buf.view(),
        "f-padded" | "c-padded" => buf.i((0..m, 0..n)).into_dim::<Ix2>(),
        "non-contig" => buf.i((slice!(0, 2 * m, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>(),
        "transposed" => buf.t(),
        _ => panic!("unknown layout case {case}"),
    }
}

/// The same view, mutable (the transposed case as a reversed-axes view).
fn layout_view_mut<'a>(
    buf: &'a mut Tensor<f64, DeviceBLAS, Ix2>,
    case: &str,
    m: usize,
    n: usize,
) -> TensorMut<'a, f64, DeviceBLAS, Ix2> {
    match case {
        "f-contig" | "c-contig" => buf.view_mut().into_dim::<Ix2>(),
        "f-padded" | "c-padded" => buf.i_mut((0..m, 0..n)).into_dim::<Ix2>(),
        "non-contig" => buf.i_mut((slice!(0, 2 * m, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>(),
        "transposed" => buf.view_mut().into_reverse_axes().into_dim::<Ix2>(),
        _ => panic!("unknown layout case {case}"),
    }
}

/// Assert that the `NaN` entries of the fresh layout buffer (padding and
/// skipped entries) are still `NaN` in `buf`.
fn check_padding(
    buf: &Tensor<f64, DeviceBLAS, Ix2>,
    device: &DeviceBLAS,
    case: &str,
    a: &[f64],
    m: usize,
    n: usize,
    at: &str,
) {
    let fresh = layout_buffer(device, case, a, m, n);
    for (idx, (&v, &f)) in buf.raw().iter().zip(fresh.raw().iter()).enumerate() {
        if f.is_nan() {
            assert!(v.is_nan(), "padding touched at {idx}, {at}");
        }
    }
}

/// Assert `lu` and `ipiv` are the LU factors of the row-major matrix `a`
/// (`L` strictly below the diagonal with a unit diagonal, `U` on and above it,
/// 0-based LAPACK pivots), by reconstructing `P A = L U`.
fn check_factors(a: &[f64], m: usize, n: usize, lu: TensorView<'_, f64, DeviceBLAS, Ix2>, ipiv: &[blas_int], at: &str) {
    let k = m.min(n);
    assert_eq!(lu.shape(), &[m, n], "lu shape, {at}");
    assert_eq!(ipiv.len(), k, "ipiv len, {at}");
    // LAPACK pivots: row i is swapped with row ipiv[i] >= i
    assert!(ipiv.iter().enumerate().all(|(i, &p)| i as blas_int <= p && p < m as blas_int), "pivot range, {at}");
    // permuted matrix P A: apply the swaps in order
    let mut pa: Vec<Vec<f64>> = (0..m).map(|i| (0..n).map(|j| a[i * n + j]).collect()).collect();
    for (i, &p) in ipiv.iter().enumerate() {
        pa.swap(i, p as usize);
    }
    for i in 0..m {
        for j in 0..n {
            let mut product = 0.0;
            for l in 0..k {
                let l_il = if l < i {
                    lu[[i, l]]
                } else if l == i {
                    1.0
                } else {
                    0.0
                };
                let u_lj = if l <= j { lu[[l, j]] } else { 0.0 };
                product += l_il * u_lj;
            }
            assert!((pa[i][j] - product).abs() < ATOL, "P A != L U at ({i}, {j}), {at}");
        }
    }
}

/// Singular matrices with the 1-based index of their first zero pivot.
fn singular_cases() -> Vec<(Vec<f64>, usize, usize, blas_int)> {
    vec![
        (vec![0.0; 9], 3, 3, 1),
        (vec![1.0, 2.0, 2.0, 4.0], 2, 2, 2),
        (vec![0.0, 1.0, 0.0, 2.0], 2, 2, 1),
        (vec![1.0, 2.0, 0.0, 0.0, 1.0, 2.0], 3, 2, 2),
    ]
}

#[test]
fn test_dgetrf_layouts() {
    let src = get_vec::<f64>('b');
    for (m, n) in SHAPES {
        // the same logical matrix for every case and device order
        let a: Vec<f64> = src[..m * n].to_vec();
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let mut reference: Option<(Vec<f64>, Vec<blas_int>)> = None;
            for case in CASES {
                let at = format!("{m}x{n}, {order:?}, {case}");
                let buf = layout_buffer(&device, case, &a, m, n);
                let buf_ptr = buf.raw().as_ptr();
                let view = layout_view(&buf, case, m, n);
                assert_eq!(view.shape(), &[m, n], "view shape, {at}");
                // the view really is the logical matrix
                for i in 0..m {
                    for j in 0..n {
                        assert_eq!(view[[i, j]], a[i * n + j], "view value ({i}, {j}), {at}");
                    }
                }
                let (lu, ipiv, info) = DGETRF::default().a(view).build().unwrap().run_with_info().unwrap();
                assert_eq!(info, 0, "info, {at}");
                // a read-only view is factored into a fresh buffer
                assert_ne!(lu.view().raw().as_ptr(), buf_ptr, "fresh buffer, {at}");
                let lu = lu.into_owned();
                let ipiv: Vec<blas_int> = ipiv.to_vec();
                check_factors(&a, m, n, lu.view(), &ipiv, &at);
                let ipiv_ref = ipiv.clone();
                // every layout gives the same factorization
                let mut packed: Vec<f64> = Vec::with_capacity(m * n);
                for i in 0..m {
                    for j in 0..n {
                        packed.push(lu[[i, j]]);
                    }
                }
                match &reference {
                    None => reference = Some((packed.clone(), ipiv_ref.clone())),
                    Some((r_lu, r_ipiv)) => {
                        assert_eq!(ipiv, *r_ipiv, "pivots, {at}");
                        assert!(packed.iter().zip(r_lu.iter()).all(|(x, y)| (x - y).abs() < ATOL), "factors, {at}");
                    },
                }
                // a mutable view is written through: the returned factors are
                // the input buffer itself (in place for an in-order layout,
                // written back from a work buffer otherwise)
                let mut buf = buf;
                let view = layout_view_mut(&mut buf, case, m, n);
                let (lu, ipiv_mut, info) = DGETRF::default().a(view).build().unwrap().run_with_info().unwrap();
                assert_eq!(info, 0, "info, mutable {at}");
                assert_eq!(ipiv_mut.to_vec(), ipiv_ref, "pivots, mutable {at}");
                assert_eq!(lu.view().raw().as_ptr(), buf_ptr, "buffer identity, mutable {at}");
                drop(lu);
                let visible = layout_view(&buf, case, m, n);
                for i in 0..m {
                    for j in 0..n {
                        let got = visible[[i, j]];
                        assert!((got - packed[i * n + j]).abs() < ATOL, "in-place factors ({i}, {j}), {at}");
                    }
                }
                // the padding is not touched
                check_padding(&buf, &device, case, &a, m, n, &at);
            }
        }
    }
}

/// The row-major bridge (transpose in, factor, transpose back) must leave
/// exactly the factors a direct column-major call computes: bitwise, pivots
/// included, also for a singular matrix whose `info > 0` factors are written
/// back.
#[test]
fn test_dgetrf_bridge_parity() {
    let src = get_vec::<f64>('b');
    let mut cases: Vec<(Vec<f64>, usize, usize, blas_int)> =
        SHAPES.iter().map(|&(m, n)| (src[..m * n].to_vec(), m, n, 0)).collect();
    cases.extend(singular_cases());
    for (a, m, n, expected) in cases {
        let at = format!("{m}x{n}");
        // the row-major in-place call, on a row-major device
        let mut device_c = DeviceBLAS::default();
        device_c.set_default_order(RowMajor);
        let mut buf_c = rt::asarray((a.clone(), [m, n].c(), &device_c)).into_dim::<Ix2>();
        // the direct column-major call, on a column-major device
        let mut device_f = DeviceBLAS::default();
        device_f.set_default_order(ColMajor);
        let mut fortran = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                fortran[i + j * m] = a[i * n + j];
            }
        }
        let mut buf_f = rt::asarray((fortran, [m, n].f(), &device_f)).into_dim::<Ix2>();

        let (lu_c, ipiv_c, info_c) = DGETRF::default().a(buf_c.view_mut()).build().unwrap().run_with_info().unwrap();
        let (lu_f, ipiv_f, info_f) = DGETRF::default().a(buf_f.view_mut()).build().unwrap().run_with_info().unwrap();
        assert_eq!((info_c, info_f), (expected, expected), "info, {at}");
        drop((lu_c, lu_f));
        assert_eq!(ipiv_c.to_vec(), ipiv_f.to_vec(), "pivots, {at}");
        // the destroyed a of the bridged call is the packed factor of the direct call
        for i in 0..m {
            for j in 0..n {
                assert_eq!(buf_c[[i, j]].to_bits(), buf_f[[i, j]].to_bits(), "packed factor ({i}, {j}), {at}");
            }
        }
    }
}

#[test]
fn test_dgetrf_singular() {
    for (a, m, n, expected) in singular_cases() {
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            for case in CASES {
                let at = format!("{m}x{n}, {order:?}, {case}");
                let mut buf = layout_buffer(&device, case, &a, m, n);
                // run: the singularity is an error
                let err = DGETRF::default()
                    .a(layout_view(&buf, case, m, n))
                    .build()
                    .unwrap()
                    .run()
                    .err()
                    .expect("run must report the zero pivot");
                assert!(format!("{err}").contains(&format!("info = {expected}")), "run error, {at}");
                // run_with_info: complete factors plus the 1-based index of the zero pivot
                let (lu, ipiv, info) =
                    DGETRF::default().a(layout_view(&buf, case, m, n)).build().unwrap().run_with_info().unwrap();
                assert_eq!(info, expected, "info, {at}");
                check_factors(&a, m, n, lu.view(), &ipiv.to_vec(), &at);
                drop(lu);
                // a mutable view: the factors are written back for info > 0
                // (through the row-major bridge too), the padding is not touched
                let (lu, ipiv, info) = DGETRF::default()
                    .a(layout_view_mut(&mut buf, case, m, n))
                    .build()
                    .unwrap()
                    .run_with_info()
                    .unwrap();
                assert_eq!(info, expected, "info, mutable {at}");
                check_factors(&a, m, n, lu.view(), &ipiv.to_vec(), &format!("mutable {at}"));
                drop(lu);
                check_padding(&buf, &device, case, &a, m, n, &at);
            }
        }
    }
}

#[test]
fn test_dgetrf_zero_size() {
    let device = DeviceBLAS::default();
    for (m, n) in [(0usize, 0usize), (0, 3), (3, 0)] {
        let a = rt::zeros_f(([m, n].c(), &device)).unwrap().into_dim::<Ix2>();
        let (lu, ipiv, info) = DGETRF::default().a(a.view()).build().unwrap().run_with_info().unwrap();
        assert_eq!(info, 0, "info, {m}x{n}");
        assert_eq!(lu.view().shape(), &[m, n], "lu shape, {m}x{n}");
        assert_eq!(ipiv.shape(), &[0], "ipiv shape, {m}x{n}");
    }
}
