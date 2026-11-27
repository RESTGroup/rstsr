use super::*;

/// Hermitian positive definite `n x n` matrix (Gram matrix BᴴB of the
/// manifest vector plus `n` on the diagonal).
fn hpd(a_vec: &[c64], n: usize) -> Vec<c64> {
    let b = &a_vec[..n * n];
    let mut a = vec![c64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (0..n).map(|l| b[l * n + i].conj() * b[l * n + j]).sum::<c64>()
                + if i == j { c64::new(n as f64, 0.0) } else { c64::new(0.0, 0.0) };
        }
    }
    a
}

/// Flat data of a stack whose slot `k` holds `slot(k)`, at the C-order
/// positions of the shape [`stacked`]`(batch, inner, order)`.
fn stack_data(slot: &dyn Fn(usize) -> Vec<c64>, batch: &[usize], inner: &[usize], order: FlagOrder) -> Vec<c64> {
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
    let mut out = vec![c64::new(0.0, 0.0); n_batch * nr * nc];
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
    slot: &dyn Fn(usize) -> Vec<c64>,
    batch: &[usize],
    inner: &[usize],
    order: FlagOrder,
    stored: FlagOrder,
    device: &DeviceBLAS,
) -> Tensor<c64, DeviceBLAS, IxD> {
    let v = stack_data(slot, batch, inner, order);
    stack_tensor(&v, batch, inner, order, stored, device)
}

/// Assert that the called triangle of `c` equals the reference factor's.
fn check_factor_triangle(
    c: TensorView<'_, c64, DeviceBLAS, Ix2>,
    reference: TensorView<'_, c64, DeviceBLAS, Ix2>,
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
                let diff = (c[[i, j]] - reference[[i, j]]).norm();
                assert!(
                    diff <= tol + tol * reference[[i, j]].norm(),
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
    let b_vec = get_vec::<c64>('b');
    let a = rt::asarray((&b_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // default (the device order's triangle: Lower on a row-major device)
    let factors = rt::linalg::cho_factor(&a);
    assert_eq!(factors.uplo, Lower);
    assert!((fingerprint(&factors.c) - c64!(62.894940653939656, -73.4705544337455)).norm() < 1e-8);
    // the other triangle is cleaned for an immutable input
    let c = &factors.c;
    for i in 0..1024 {
        for j in 0..1024 {
            if i < j {
                assert_eq!(c[[i, j]], c64::new(0.0, 0.0), "cleaned ({i}, {j})");
            }
        }
    }
    let l = c.view();
    let lc = l.view().conj();
    let lh = lc.t();
    let llh = &l % lh;
    assert!(rt::allclose(llh.view(), a.view(), (1e-8, 1e-8)), "L Lᴴ = A");

    // upper
    let factors = rt::linalg::cho_factor((&a, Upper));
    assert_eq!(factors.uplo, Upper);
    assert!((fingerprint(&factors.c) - c64!(13.720509103164034, -1.8066465348487615)).norm() < 1e-8);
    let u = &factors.c;
    let uc = u.view().conj();
    let uh = uc.t();
    let uhu = &uh % u.view();
    assert!(rt::allclose(uhu.view(), a.view(), (1e-8, 1e-8)), "Uᴴ U = A");
}

/// The other triangle and the padding of the input kinds, over device order
/// and storage order (Hermitian matrices).
#[test]
fn test_cho_factor_relations() {
    let a_vec = get_vec::<c64>('a');
    let n = 5usize;
    let a = hpd(&a_vec, n);
    let zero = c64::new(0.0, 0.0);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        // a matrix (row-major `v`) of shape `(rows, cols)` stored in `storage`
        // (`rt::asarray` consumes `v` in memory order, so convert the layout)
        let mk = |v: &[c64], rows: usize, cols: usize, storage: FlagOrder| -> Tensor<c64, DeviceBLAS, Ix2> {
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
            check_factor_triangle(factors.c.view(), reference.view(), uplo, 1e-11, &format!("factor, {tag}"));
            for i in 0..n {
                for j in 0..n {
                    if other(i, j) {
                        assert_eq!(factors.c[[i, j]], zero, "other triangle cleaned ({i}, {j}), {tag}");
                    }
                }
            }

            // owned input in the device order: consumed in place, the other
            // triangle keeps the input's values
            let owned = mk(&a, n, n, order);
            let factors = rt::linalg::cho_factor((owned, uplo));
            check_factor_triangle(factors.c.view(), reference.view(), uplo, 1e-11, &format!("owned factor, {tag}"));
            for i in 0..n {
                for j in 0..n {
                    if other(i, j) {
                        assert_eq!(factors.c[[i, j]], a[i * n + j], "owned other triangle ({i}, {j}), {tag}");
                    }
                }
            }

            // mutable input in the device order: factored in place, the result
            // aliases the input buffer, the other triangle keeps the input's
            // values
            let mut buf = mk(&a, n, n, order);
            let ptr = buf.raw().as_ptr();
            let factors = rt::linalg::cho_factor((buf.view_mut(), uplo));
            assert_eq!(factors.c.raw().as_ptr(), ptr, "in place, {tag}");
            check_factor_triangle(factors.c.view(), reference.view(), uplo, 1e-11, &format!("in-place factor, {tag}"));
            drop(factors);
            for i in 0..n {
                for j in 0..n {
                    if other(i, j) {
                        assert_eq!(buf[[i, j]], a[i * n + j], "in-place other triangle ({i}, {j}), {tag}");
                    }
                }
            }

            // padded mutable view in the device order: factored in place, the
            // NaN padding is untouched
            let p = n + 3;
            let nan = c64::new(f64::NAN, f64::NAN);
            let mut padded = mk(&vec![nan; p * p], p, p, order);
            padded.i_mut((0..n, 0..n)).assign_f(mk(&a, n, n, order).view()).unwrap();
            let ptr = padded.raw().as_ptr();
            let view = padded.i_mut((0..n, 0..n)).into_dim::<Ix2>();
            let factors = rt::linalg::cho_factor((view, uplo));
            assert_eq!(factors.c.raw().as_ptr(), ptr, "padded in place, {tag}");
            check_factor_triangle(factors.c.view(), reference.view(), uplo, 1e-11, &format!("padded factor, {tag}"));
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
    let a_vec = get_vec::<c64>('a');
    let n = 5usize;
    let gram = hpd(&a_vec, n);
    let shift = c64::new(1.0, 0.0);
    let slot_data = |k: usize| -> Vec<c64> {
        (0..n * n)
            .map(|j| gram[j] + if j / n == j % n { shift * (1.0 + k as f64) } else { c64::new(0.0, 0.0) })
            .collect()
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
                check_factor_triangle(c_slot.view(), slot_c.c.view(), factors.uplo, 1e-11, &at);
                assert_eq!(slot_c.uplo, factors.uplo, "flag, {at}");
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
                    1e-11,
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
    let a_vec = get_vec::<c64>('a');
    let n = 4usize;
    let a = rt::asarray((hpd(&a_vec, n), [n, n].c(), &device)).into_dim::<Ix2>();
    let reference = rt::linalg::cho_factor(&a);

    // read-only kinds (cleaned)
    let r = rt::linalg::cho_factor(a.view());
    assert!(rt::allclose(r.c.view(), reference.c.view(), (1e-11, 1e-11)), "view");
    let r = rt::linalg::cho_factor(&a);
    assert!(rt::allclose(r.c.view(), reference.c.view(), (1e-11, 1e-11)), "&");
    let r = rt::linalg::cho_factor((a.view(), Upper));
    assert_eq!(r.uplo, Upper, "view + flag");

    // consumed kinds (not cleaned): the called triangle matches
    let r = rt::linalg::cho_factor(a.clone());
    check_factor_triangle(r.c.view(), reference.c.view(), reference.uplo, 1e-11, "owned");
    let mut a_m = a.clone();
    let ptr = a_m.raw().as_ptr();
    let r = rt::linalg::cho_factor(a_m.view_mut());
    assert_eq!(r.c.raw().as_ptr(), ptr, "view_mut");
    check_factor_triangle(r.c.view(), reference.c.view(), reference.uplo, 1e-11, "view_mut");

    // the result destructures as (c, uplo)
    let (c, uplo): (Tensor<c64, DeviceBLAS, Ix2>, FlagUpLo) = rt::linalg::cho_factor(&a).into();
    assert_eq!(uplo, reference.uplo);
    assert!(rt::allclose(c.view(), reference.c.view(), (1e-11, 1e-11)), "tuple");

    // fallible variant
    let r = rt::linalg::cho_factor_f(&a).unwrap();
    assert!(rt::allclose(r.c.view(), reference.c.view(), (1e-11, 1e-11)), "cho_factor_f");
}

#[test]
fn test_cho_factor_errors_zero() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');

    // 1-D input
    let a1 = rt::asarray((&a_vec[..6], [6].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::cho_factor_f(a1.view()).err().expect("1-D must fail"));
    assert!(msg.contains("at least 2 dimensions") || msg.contains("must be square"), "{msg}");

    // non-square input
    let a_rect = rt::asarray((&a_vec[..24], [6usize, 4].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::cho_factor_f(a_rect.view()).err().expect("non-square must fail"));
    assert!(msg.contains("must be square"), "{msg}");

    // not positive definite: the POTRF error with the info
    let a = rt::asarray((vec![c64!(1.0), c64!(2.0), c64!(2.0), c64!(1.0)], [2usize, 2].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::cho_factor_f(a.view()).err().expect("non-PD must fail"));
    assert!(msg.contains("info = 2"), "{msg}");

    // zero size: empty factor, the flag returned, no driver call
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let a: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
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
        let a: Tensor<c64, DeviceBLAS, IxD> = rt::zeros_f((stacked(&[0], &[2, 2], order), &device)).unwrap();
        let factors = rt::linalg::cho_factor(a.view());
        assert_eq!(factors.c.shape(), stacked(&[0], &[2, 2], order).as_slice(), "empty batch, {order:?}");
    }
}
