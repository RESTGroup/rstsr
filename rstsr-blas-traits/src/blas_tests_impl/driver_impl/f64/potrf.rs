//! Layout-matrix, relation, non-PD and zero-size tests for the POTRF driver.
//!
//! The row-major cells exercise the builder's transposed-view call: the
//! factor must land in the requested triangle of the buffer exactly as the
//! row-major bridge would place it.

use std::cmp::Ordering;

use super::*;
use rstsr_blas_traits::lapack_solve::*;

const ATOL: f64 = 1e-13;
const CASES: [&str; 6] = ["f-contig", "f-padded", "c-contig", "c-padded", "non-contig", "transposed"];
const NS: [usize; 5] = [1, 2, 3, 5, 6];

/// Buffer of a layout case holding the row-major matrix `a` (`n x n`); padding
/// and the skipped entries are `NaN`, so a wrong leading dimension shows up.
fn layout_buffer(device: &DeviceBLAS, case: &str, a: &[f64], n: usize) -> Tensor<f64, DeviceBLAS, Ix2> {
    let at = |v: Vec<f64>, size: usize, order: FlagOrder| -> Tensor<f64, DeviceBLAS, Ix2> {
        match order {
            RowMajor => rt::asarray((v, [size, size].c(), device)).into_dim::<Ix2>(),
            ColMajor => rt::asarray((v, [size, size].f(), device)).into_dim::<Ix2>(),
        }
    };
    // the logical matrix as a Fortran- or C-ordered vector of shape (size, size)
    let packed = |size: usize, order: FlagOrder| -> Vec<f64> {
        let mut v = vec![f64::NAN; size * size];
        for i in 0..n {
            for j in 0..n {
                v[match order {
                    ColMajor => i + j * size,
                    RowMajor => i * size + j,
                }] = a[i * n + j];
            }
        }
        v
    };
    match case {
        "c-contig" => at(a.to_vec(), n, RowMajor),
        "f-contig" => at(packed(n, ColMajor), n, ColMajor),
        // padded columns (Fortran order) or padded rows (C order)
        "f-padded" => at(packed(n + 4, ColMajor), n + 4, ColMajor),
        "c-padded" => at(packed(n + 4, RowMajor), n + 4, RowMajor),
        // every second entry along both axes of a buffer of double size
        "non-contig" => {
            let mut v = vec![f64::NAN; 4 * n * n];
            for i in 0..n {
                for j in 0..n {
                    v[(2 * i) * (2 * n) + 2 * j] = a[i * n + j];
                }
            }
            at(v, 2 * n, RowMajor)
        },
        // the matrix stored as its transpose (a column-major (N, N) buffer has
        // the memory of a row-major (N, N) one)
        "transposed" => at(a.to_vec(), n, ColMajor),
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

#[test]
fn test_dpotrf() {
    let device = DeviceBLAS::default();
    let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // default
    let driver = DPOTRF::default().a(b.view()).build().unwrap();
    let c = driver.run().unwrap();
    let c = c.into_owned();
    println!("fingerprint {:?}", fingerprint(&c));
    assert!((fingerprint(&c) - 35.17266259472725).abs() < 1e-8);

    // upper
    let driver = DPOTRF::default().a(b.view()).uplo(Upper).build().unwrap();
    let c = driver.run().unwrap();
    let c = c.into_owned();
    println!("fingerprint {:?}", fingerprint(&c));
    assert!((fingerprint(&c) - -53.53353704132017).abs() < 1e-8);
}

#[test]
fn test_dpotrf_layout() {
    let src = get_vec::<f64>('b');
    for n in NS {
        // two distinct triangular factors: the lower triangle holds an
        // invertible L (positive diagonal, arbitrary entries below), the
        // upper triangle a different U, so both triangles define a
        // positive-definite matrix (L Lᵀ resp. Uᵀ U) with different values,
        // and a wrong-triangle read shows up
        let b: Vec<f64> = src[..n * n].to_vec();
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = match i.cmp(&j) {
                    Ordering::Greater => b[i * n + j],
                    Ordering::Equal => b[i * n + j].abs() + n as f64 + 1.0,
                    Ordering::Less => -2.0 * b[i * n + j] - 0.25,
                };
            }
        }
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            // the f-contig run is the reference for every other layout
            let mut reference: [Option<Vec<f64>>; 2] = [None, None];
            for (slot, uplo) in [(0usize, Lower), (1, Upper)] {
                let buf = layout_buffer(&device, "f-contig", &a, n);
                let c = DPOTRF::default().a(buf.view()).uplo(uplo).build().unwrap().run().unwrap();
                let c = c.into_owned();
                let mut tri = Vec::new();
                for i in 0..n {
                    for j in 0..n {
                        if match uplo {
                            Lower => i >= j,
                            Upper => i <= j,
                        } {
                            tri.push(c[[i, j]]);
                        }
                    }
                }
                reference[slot] = Some(tri);
            }
            for (slot, uplo) in [(0usize, Lower), (1, Upper)] {
                let reference = reference[slot].as_ref().unwrap();
                for case in CASES {
                    let at = format!("{n}x{n}, {order:?}, {uplo:?}, {case}");
                    let mut buf = layout_buffer(&device, case, &a, n);
                    let buf_ptr = buf.raw().as_ptr();
                    // a read-only input is factored into a fresh buffer
                    let view = layout_view(&buf, case, n);
                    let c = DPOTRF::default().a(view).uplo(uplo).build().unwrap().run().unwrap();
                    assert_ne!(c.view().raw().as_ptr(), buf_ptr, "fresh buffer, {at}");
                    let c = c.into_owned();
                    let mut idx = 0;
                    for i in 0..n {
                        for j in 0..n {
                            if match uplo {
                                Lower => i >= j,
                                Upper => i <= j,
                            } {
                                assert!((c[[i, j]] - reference[idx]).abs() < ATOL, "factor ({i}, {j}), {at}");
                                idx += 1;
                            }
                        }
                    }
                    // a mutable input is factored in place exactly when its
                    // layout is one of the two orders, and written back from a
                    // work buffer otherwise
                    let view = layout_view(&buf, case, n);
                    let inplace = view.f_prefer() || view.c_prefer();
                    let view = layout_view_mut(&mut buf, case, n);
                    let c = DPOTRF::default().a(view).uplo(uplo).build().unwrap().run().unwrap();
                    assert_eq!(c.view().raw().as_ptr() == buf_ptr, inplace, "in place, {at}");
                    drop(c.clone_to_mut());
                    // the factor equals the reference, the other triangle holds
                    // the input's values, the padding is untouched
                    let visible = layout_view(&buf, case, n);
                    let mut idx = 0;
                    for i in 0..n {
                        for j in 0..n {
                            if match uplo {
                                Lower => i >= j,
                                Upper => i <= j,
                            } {
                                assert!(
                                    (visible[[i, j]] - reference[idx]).abs() < ATOL,
                                    "in-place factor ({i}, {j}), {at}"
                                );
                                idx += 1;
                            } else {
                                assert_eq!(visible[[i, j]], a[i * n + j], "other triangle ({i}, {j}), {at}");
                            }
                        }
                    }
                    check_padding(&buf, &device, case, &a, n, &at);
                }
            }
        }
    }
}

/// `L Lᵀ = A` / `Uᵀ U = A` for the whole matrix, on both device orders.
#[test]
fn test_dpotrf_relations() {
    let src = get_vec::<f64>('b');
    for n in [3usize, 5, 6] {
        let b: Vec<f64> = src[..n * n].to_vec();
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] =
                    (0..n).map(|l| b[l * n + i] * b[l * n + j]).sum::<f64>() + if i == j { n as f64 } else { 0.0 };
            }
        }
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let aa = rt::asarray((a.clone(), [n, n].c(), &device)).into_dim::<Ix2>();
            for uplo in [Lower, Upper] {
                let at = format!("{n}x{n}, {order:?}, {uplo:?}");
                let mut buf = rt::asarray((a.clone(), [n, n].c(), &device)).into_dim::<Ix2>();
                DPOTRF::default().a(buf.view_mut()).uplo(uplo).build().unwrap().run().unwrap().clone_to_mut();
                let mut tri: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([n, n].c(), &device)).unwrap().into_dim();
                for i in 0..n {
                    for j in 0..n {
                        if match uplo {
                            Lower => i >= j,
                            Upper => i <= j,
                        } {
                            tri[[i, j]] = buf[[i, j]];
                        }
                    }
                }
                let prod = match uplo {
                    Lower => &tri % tri.t(),
                    Upper => tri.t() % &tri,
                };
                assert!(rt::allclose(prod.view(), aa.view(), Some(1e-10)), "relation, {at}");
            }
        }
    }
}

/// The leading 2x2 minor of these matrices is not positive definite; `run`
/// must report `info = 2` on both device orders and both triangles.
#[test]
fn test_dpotrf_not_positive_definite() {
    let cases: [Vec<f64>; 2] = [vec![1.0, 2.0, 2.0, 1.0], vec![1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]];
    for a in cases {
        let n = (a.len() as f64).sqrt() as usize;
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            for uplo in [Lower, Upper] {
                let at = format!("{n}x{n}, {order:?}, {uplo:?}");
                let buf = rt::asarray((a.clone(), [n, n].c(), &device)).into_dim::<Ix2>();
                let err =
                    DPOTRF::default().a(buf.view()).uplo(uplo).build().unwrap().run().err().expect("not PD must fail");
                assert!(format!("{err}").contains("info = 2"), "error, {at}: {err}");
            }
        }
    }
}

#[test]
fn test_dpotrf_zero_size() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let c = DPOTRF::default().a(a.view()).build().unwrap().run().unwrap();
        assert_eq!(c.view().shape(), &[0, 0], "zero size, {order:?}");
    }
}
