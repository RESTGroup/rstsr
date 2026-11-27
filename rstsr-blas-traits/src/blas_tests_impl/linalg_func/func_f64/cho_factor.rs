use super::*;
use rstsr::prelude::linalg::ChoFactorResult;

/// The result types follow the dims of a 2-D input.
type ChoFactorOut = ChoFactorResult<Tensor<f64, DeviceBLAS, Ix2>>;

/// Symmetric positive definite `n x n` matrix (Gram matrix of the manifest
/// vector plus `n` on the diagonal).
fn spd(a_vec: &[f64], n: usize) -> Vec<f64> {
    let b = &a_vec[..n * n];
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] =
                (0..n).map(|l| b[l * n + i] * b[l * n + j]).sum::<f64>() + if i == j { n as f64 } else { 0.0 };
        }
    }
    a
}

/// Flat data of a stack whose slot `k` holds `slot(k)` (a row-major matrix),
/// placed at the C-order positions of the shape
/// [`stacked`]`(batch, inner, order)` — the element order `rt::asarray`
/// consumes, so the logical slots are `slot(k)` under either device order.
fn stack_data(slot: &dyn Fn(usize) -> Vec<f64>, batch: &[usize], inner: &[usize], order: FlagOrder) -> Vec<f64> {
    let shape = stacked(batch, inner, order);
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for ax in (0..ndim - 1).rev() {
        strides[ax] = strides[ax + 1] * shape[ax + 1];
    }
    let (batch_axes, inner_axes) = match order {
        RowMajor => ((0..ndim - 2).collect::<Vec<_>>(), (ndim - 2..ndim).collect::<Vec<_>>()),
        ColMajor => ((2..ndim).collect::<Vec<_>>(), (0..2).collect::<Vec<_>>()),
    };
    let n_batch: usize = batch.iter().product();
    let (nr, nc) = (inner[0], inner[1]);
    let mut out = vec![0.0; n_batch * nr * nc];
    let mut idx = vec![0usize; batch.len()];
    for bi in 0..n_batch {
        let mut rem = bi;
        for ax in (0..batch.len()).rev() {
            idx[ax] = rem % batch[ax];
            rem /= batch[ax];
        }
        let offset: usize = batch_axes.iter().zip(idx.iter()).map(|(&ax, &v)| v * strides[ax]).sum();
        let s = slot(bi);
        for p in 0..nr {
            for q in 0..nc {
                out[offset + p * strides[inner_axes[0]] + q * strides[inner_axes[1]]] = s[p * nc + q];
            }
        }
    }
    out
}

/// A stack whose slot `k` is `slot(k)`, stored in the memory order `stored`.
fn stack_of(
    slot: &dyn Fn(usize) -> Vec<f64>,
    batch: &[usize],
    inner: &[usize],
    order: FlagOrder,
    stored: FlagOrder,
    device: &DeviceBLAS,
) -> Tensor<f64, DeviceBLAS, IxD> {
    let v = stack_data(slot, batch, inner, order);
    stack_tensor(&v, batch, inner, order, stored, device)
}

/// Assert that the called triangle of `c` equals the reference factor's
/// (the other triangle may legitimately hold the input's values).
fn check_factor_triangle(
    c: TensorView<'_, f64, DeviceBLAS, Ix2>,
    reference: TensorView<'_, f64, DeviceBLAS, Ix2>,
    uplo: FlagUpLo,
    tol: f64,
    at: &str,
) {
    let n = reference.shape()[0];
    for i in 0..n {
        for j in 0..n {
            if match uplo {
                Lower => i >= j,
                Upper => i <= j,
            } {
                let diff = (c[[i, j]] - reference[[i, j]]).abs();
                assert!(
                    diff <= tol + tol * reference[[i, j]].abs(),
                    "({i}, {j}): {} vs {}, {at}",
                    c[[i, j]],
                    reference[[i, j]]
                );
            }
        }
    }
}

#[test]
fn test_cho_factor() {
    let device = DeviceBLAS::default();
    // the factor's manifest is `b` (as for cholesky)
    let b_vec = get_vec::<f64>('b');
    let a = rt::asarray((&b_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // default (the device order's triangle: Lower on a row-major device)
    let factors: ChoFactorOut = rt::linalg::cho_factor(&a);
    assert_eq!(factors.uplo, Lower);
    assert!((fingerprint(&factors.c) - 43.219044785561486).abs() < 1e-8);
    // the other triangle is cleaned for an immutable input
    let c = &factors.c;
    for i in 0..1024 {
        for j in 0..1024 {
            if i < j {
                assert_eq!(c[[i, j]], 0.0, "cleaned ({i}, {j})");
            }
        }
    }
    let l = c.view();
    let llt = &l % l.t();
    assert!(rt::allclose(llt.view(), a.view(), (1e-8, 1e-8)), "L Lᵀ = A");

    // upper
    let factors = rt::linalg::cho_factor((&a, Upper));
    assert_eq!(factors.uplo, Upper);
    assert!((fingerprint(&factors.c) - -25.92565512481678).abs() < 1e-8);
    let u = &factors.c;
    let utu = u.t() % u.view();
    assert!(rt::allclose(utu.view(), a.view(), (1e-8, 1e-8)), "Uᵀ U = A");
}

/// The worked example of the SciPy `cho_factor` docstring.
#[test]
fn test_cho_factor_docstring() {
    let device = DeviceBLAS::default();
    let a: Vec<f64> = vec![9.0, 3.0, 1.0, 5.0, 3.0, 7.0, 5.0, 1.0, 1.0, 5.0, 9.0, 2.0, 5.0, 1.0, 2.0, 6.0];
    let a = rt::asarray((a, [4, 4].c(), &device)).into_dim::<Ix2>();
    let factors = rt::linalg::cho_factor((&a, Upper));
    assert_eq!(factors.uplo, Upper);
    let c = &factors.c;
    let expected: [f64; 16] = [
        3.0,
        1.0,
        1.0 / 3.0,
        5.0 / 3.0,
        0.0,
        2.449489743,
        1.905158689,
        -0.272165527,
        0.0,
        0.0,
        2.293307491,
        0.855952800,
        0.0,
        0.0,
        0.0,
        1.554185626,
    ];
    for (idx, &v) in expected.iter().enumerate() {
        let (i, j) = (idx / 4, idx % 4);
        assert!((c[[i, j]] - v).abs() < 1e-8, "c ({i}, {j}): {} vs {v}", c[[i, j]]);
    }
    // the factor solves the system of the docstring example
    let b: Tensor<f64, DeviceBLAS, Ix1> = rt::asarray((vec![1.0; 4], [4].c(), &device)).into_dim::<Ix1>();
    let x = rt::linalg::cho_solve((&factors, &b));
    let expected: [f64; 4] = [-0.01749271, 0.11953353, 0.01166181, 0.1574344];
    for (idx, &v) in expected.iter().enumerate() {
        assert!((x[[idx]] - v).abs() < 1e-8, "x[{idx}]: {} vs {v}", x[[idx]]);
    }
}

/// The other triangle and the padding of every input kind, over device order
/// and storage order.
#[test]
fn test_cho_factor_relations() {
    let a_vec = get_vec::<f64>('a');
    let n = 5usize;
    let a = spd(&a_vec, n);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        // a matrix (row-major `v`) of shape `(rows, cols)` stored in `storage`
        // (`rt::asarray` consumes `v` in memory order, so convert the layout)
        let mk = |v: &[f64], rows: usize, cols: usize, storage: FlagOrder| -> Tensor<f64, DeviceBLAS, Ix2> {
            rt::asarray((v.to_vec(), [rows, cols].c(), &device))
                .into_dim::<Ix2>()
                .to_contig_f(storage)
                .unwrap()
                .into_owned()
                .into_dim::<Ix2>()
        };
        for uplo in [Lower, Upper] {
            let tag = format!("{order:?}, {uplo:?}");
            let reference =
                rt::linalg::cholesky((rt::asarray((a.clone(), [n, n].c(), &device)).into_dim::<Ix2>().view(), uplo));
            let other = |i: usize, j: usize| match uplo {
                Lower => i < j,
                Upper => i > j,
            };

            // read-only input: fresh buffer, cleaned
            let buf = mk(&a, n, n, order);
            let factors = rt::linalg::cho_factor((buf.view(), uplo));
            assert_eq!(factors.uplo, uplo, "flag, {tag}");
            check_factor_triangle(factors.c.view(), reference.view(), uplo, 1e-12, &format!("factor, {tag}"));
            for i in 0..n {
                for j in 0..n {
                    if other(i, j) {
                        assert_eq!(factors.c[[i, j]], 0.0, "other triangle cleaned ({i}, {j}), {tag}");
                    }
                }
            }

            // owned input in the device order: consumed in place, the other
            // triangle keeps the input's values
            let owned = mk(&a, n, n, order);
            let factors = rt::linalg::cho_factor((owned, uplo));
            check_factor_triangle(factors.c.view(), reference.view(), uplo, 1e-12, &format!("owned factor, {tag}"));
            for i in 0..n {
                for j in 0..n {
                    if other(i, j) {
                        assert_eq!(factors.c[[i, j]], a[i * n + j], "owned other triangle ({i}, {j}), {tag}");
                    }
                }
            }

            // mutable input in the device order: factored in place, the result
            // aliases the input buffer
            let mut buf = mk(&a, n, n, order);
            let ptr = buf.raw().as_ptr();
            let factors = rt::linalg::cho_factor((buf.view_mut(), uplo));
            assert_eq!(factors.c.raw().as_ptr(), ptr, "in place, {tag}");
            check_factor_triangle(factors.c.view(), reference.view(), uplo, 1e-12, &format!("in-place factor, {tag}"));
            drop(factors);
            for i in 0..n {
                for j in 0..n {
                    if other(i, j) {
                        assert_eq!(buf[[i, j]], a[i * n + j], "in-place other triangle ({i}, {j}), {tag}");
                    }
                }
            }

            // mutable input of the other order: factored in a work buffer,
            // written back, not cleaned
            let stored = match order {
                RowMajor => ColMajor,
                ColMajor => RowMajor,
            };
            let mut off = mk(&a, n, n, stored);
            let ptr = off.raw().as_ptr();
            let factors = rt::linalg::cho_factor((off.view_mut(), uplo));
            assert_eq!(factors.c.raw().as_ptr(), ptr, "off-order write-back, {tag}");
            check_factor_triangle(factors.c.view(), reference.view(), uplo, 1e-12, &format!("off-order factor, {tag}"));
            drop(factors);
            for i in 0..n {
                for j in 0..n {
                    if other(i, j) {
                        assert_eq!(off[[i, j]], a[i * n + j], "off-order other triangle ({i}, {j}), {tag}");
                    }
                }
            }

            // padded mutable view in the device order: factored in place, the
            // NaN padding is untouched
            let p = n + 3;
            let mut padded = mk(&vec![f64::NAN; p * p], p, p, order);
            padded.i_mut((0..n, 0..n)).assign_f(mk(&a, n, n, order).view()).unwrap();
            let ptr = padded.raw().as_ptr();
            let view = padded.i_mut((0..n, 0..n)).into_dim::<Ix2>();
            let factors = rt::linalg::cho_factor((view, uplo));
            assert_eq!(factors.c.raw().as_ptr(), ptr, "padded in place, {tag}");
            check_factor_triangle(factors.c.view(), reference.view(), uplo, 1e-12, &format!("padded factor, {tag}"));
            drop(factors);
            for i in 0..p {
                for j in 0..p {
                    if i >= n || j >= n {
                        assert!(padded[[i, j]].is_nan(), "padding ({i}, {j}), {tag}");
                    }
                }
            }
        }
    }
}

/// Batch parity with the per-slice 2-D calls, both device orders and storage
/// orders; the default flag follows the device order and `cho_factor(a).c`
/// equals `cholesky(a)`.
#[test]
fn test_cho_factor_nd() {
    let a_vec = get_vec::<f64>('a');
    let n = 5usize;
    let gram = spd(&a_vec, n);
    // distinct per-slot SPD matrices: the diagonal shift is (1 + k)
    let slot_data = |k: usize| -> Vec<f64> {
        (0..n * n).map(|j| gram[j] + if j / n == j % n { 1.0 + k as f64 } else { 0.0 }).collect()
    };
    for order in [RowMajor, ColMajor] {
        for stored in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let tag = format!("{order:?} device, {stored:?} stack");
            let a = stack_of(&slot_data, &[3], &[n, n], order, stored, &device);

            let factors = match rt::linalg::cho_factor_f(&a) {
                Ok(f) => f,
                Err(e) => panic!("{tag}: POTRF failed: {e}"),
            };
            assert_eq!(
                factors.uplo,
                match order {
                    RowMajor => Lower,
                    ColMajor => Upper,
                },
                "default flag, {tag}"
            );
            assert_eq!(factors.c.shape(), a.shape(), "shape, {tag}");
            let chol = rt::linalg::cholesky(&a);
            for k in 0..3 {
                let at = format!("slot {k}, {tag}");
                let slot = match order {
                    RowMajor => a.i((k, .., ..)),
                    ColMajor => a.i((.., .., k)),
                }
                .into_owned()
                .into_dim::<Ix2>();
                let slot_c = rt::linalg::cho_factor(&slot);
                let c_slot = match order {
                    RowMajor => factors.c.i((k, .., ..)),
                    ColMajor => factors.c.i((.., .., k)),
                }
                .into_owned()
                .into_dim::<Ix2>();
                check_factor_triangle(c_slot.view(), slot_c.c.view(), factors.uplo, 1e-12, &at);
                assert_eq!(slot_c.uplo, factors.uplo, "flag, {at}");
                // read-only inputs are cleaned
                for i in 0..n {
                    for j in 0..n {
                        if match factors.uplo {
                            Lower => i < j,
                            Upper => i > j,
                        } {
                            assert_eq!(c_slot[[i, j]], 0.0, "cleaned ({i}, {j}), {at}");
                        }
                    }
                }
                // cho_factor and cholesky agree on the same triangle
                let chol_slot = match order {
                    RowMajor => chol.i((k, .., ..)),
                    ColMajor => chol.i((.., .., k)),
                }
                .into_owned()
                .into_dim::<Ix2>();
                check_factor_triangle(
                    c_slot.view(),
                    chol_slot.view(),
                    factors.uplo,
                    1e-12,
                    &format!("cholesky parity, {at}"),
                );
            }
        }
    }
}

/// Every overload on a 4x4 block: forms, kinds and the destructuring tuple.
#[test]
fn test_cho_factor_overloads() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let n = 4usize;
    let a = rt::asarray((spd(&a_vec, n), [n, n].c(), &device)).into_dim::<Ix2>();
    let reference = rt::linalg::cho_factor(&a);

    // read-only kinds (cleaned)
    let r: ChoFactorResult<Tensor<f64, DeviceBLAS, Ix2>> = rt::linalg::cho_factor(a.view());
    assert!(rt::allclose(r.c.view(), reference.c.view(), (1e-12, 1e-12)), "view");
    let r = rt::linalg::cho_factor(&a);
    assert!(rt::allclose(r.c.view(), reference.c.view(), (1e-12, 1e-12)), "&");
    let r = rt::linalg::cho_factor((a.view(), Upper));
    assert_eq!(r.uplo, Upper, "view + flag");
    let r = rt::linalg::cho_factor((&a, Lower));
    assert_eq!(r.uplo, Lower, "& + flag");

    // consumed kinds (not cleaned): the called triangle matches
    let r = rt::linalg::cho_factor((a.clone(), None));
    check_factor_triangle(r.c.view(), reference.c.view(), reference.uplo, 1e-12, "owned tuple");
    let mut a_m = a.clone();
    let ptr = a_m.raw().as_ptr();
    let r = rt::linalg::cho_factor(a_m.view_mut());
    assert_eq!(r.c.raw().as_ptr(), ptr, "view_mut");
    check_factor_triangle(r.c.view(), reference.c.view(), reference.uplo, 1e-12, "view_mut");
    let r = rt::linalg::cho_factor(a.clone());
    check_factor_triangle(r.c.view(), reference.c.view(), reference.uplo, 1e-12, "owned bare");
    let reference_u = rt::linalg::cho_factor((&a, Upper));
    let mut a_m2 = a.clone();
    let r = rt::linalg::cho_factor((a_m2.view_mut(), Upper));
    assert_eq!(r.uplo, Upper, "view_mut + flag");
    check_factor_triangle(r.c.view(), reference_u.c.view(), Upper, 1e-12, "view_mut + flag factor");

    // the result destructures as (c, uplo)
    let (c, uplo): (Tensor<f64, DeviceBLAS, Ix2>, FlagUpLo) = rt::linalg::cho_factor(&a).into();
    assert_eq!(uplo, reference.uplo);
    assert!(rt::allclose(c.view(), reference.c.view(), (1e-12, 1e-12)), "tuple");

    // fallible variant
    let r = rt::linalg::cho_factor_f(&a).unwrap();
    assert!(rt::allclose(r.c.view(), reference.c.view(), (1e-12, 1e-12)), "cho_factor_f");
}

#[test]
fn test_cho_factor_errors_zero() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');

    // 1-D input
    let a1 = rt::asarray((&a_vec[..6], [6].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::cho_factor_f(a1.view()).err().expect("1-D must fail"));
    assert!(msg.contains("at least 2 dimensions") || msg.contains("must be square"), "{msg}");

    // non-square input
    let a_rect = rt::asarray((&a_vec[..24], [6usize, 4].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::cho_factor_f(a_rect.view()).err().expect("non-square must fail"));
    assert!(msg.contains("must be square"), "{msg}");

    // not positive definite: the POTRF error with the info
    let a = rt::asarray((vec![1.0, 2.0, 2.0, 1.0], [2usize, 2].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::cho_factor_f(a.view()).err().expect("non-PD must fail"));
    assert!(msg.contains("info = 2"), "{msg}");

    // zero size: empty factor, the flag returned, no driver call
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let factors = rt::linalg::cho_factor(a.view());
        assert_eq!(factors.c.shape(), &[0, 0], "zero size, {order:?}");
        assert_eq!(
            factors.uplo,
            match order {
                RowMajor => Lower,
                ColMajor => Upper,
            },
            "zero size flag, {order:?}"
        );
        // an empty batch
        let a: Tensor<f64, DeviceBLAS, IxD> = rt::zeros_f((stacked(&[0], &[2, 2], order), &device)).unwrap();
        let factors = rt::linalg::cho_factor(a.view());
        assert_eq!(factors.c.shape(), stacked(&[0], &[2, 2], order).as_slice(), "empty batch, {order:?}");
    }
}
