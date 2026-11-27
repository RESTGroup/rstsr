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

/// Assert that `x` solves `A x = b` within `1e-9` (Hermitian `A`).
fn check_solution(
    a: TensorView<'_, c64, DeviceBLAS, Ix2>,
    x: TensorView<'_, c64, DeviceBLAS, IxD>,
    b: TensorView<'_, c64, DeviceBLAS, IxD>,
    at: &str,
) {
    let residual = (a % x - b).abs().max_f().unwrap_or(f64::NAN);
    assert!(residual < 1e-9, "residual {residual}, {at}");
}

#[test]
fn test_cho_solve() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    let a = rt::asarray((&b_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let factors = rt::linalg::cho_factor((&a, Lower));
    assert_eq!(factors.uplo, Lower);

    // K = 1 (the `lu_solve` c64 convention: fingerprint of |x|)
    let b = rt::asarray((&a_vec[..1024], [1024].c(), &device)).into_dim::<Ix1>();
    let x = rt::linalg::cho_solve((&factors, &b));
    assert_eq!(x.shape(), &[1024]);
    assert!((fingerprint(&x.abs()) - 5.093725636509232).abs() < 1e-8);

    // upper factor
    let factors = rt::linalg::cho_factor((&a, Upper));
    let b = rt::asarray((&a_vec[..1024], [1024].c(), &device)).into_dim::<Ix1>();
    let x = rt::linalg::cho_solve((&factors, &b));
    assert!((fingerprint(&x.abs()) - 5.093725636509143).abs() < 1e-8);
}

/// `A x = b` for vector and matrix `b`, the conjugated-step residual, a solve
/// from a `cholesky` result, and the insensitivity to the other triangle.
#[test]
fn test_cho_solve_relations() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    for (n, k) in [(1usize, 1usize), (3, 1), (5, 2), (6, 4)] {
        let a = rt::asarray((hpd(&a_vec, n), [n, n].c(), &device)).into_dim::<Ix2>();
        let b: Tensor<c64, DeviceBLAS, IxD> =
            rt::asarray((b_vec[..n * k].to_vec(), [n, k].c(), &device)).into_owned().into_dyn();
        let b_vec_rhs = rt::asarray((b_vec[..n].to_vec(), [n].c(), &device)).into_dim::<Ix1>();
        for uplo in [Lower, Upper] {
            let tag = format!("{n}x{n}, K = {k}, {uplo:?}");
            let factors = rt::linalg::cho_factor((a.view(), uplo));

            // matrix b: the residual and the general solve agree
            let x = rt::linalg::cho_solve((&factors, &b));
            assert_eq!(x.shape(), &[n, k], "shape, {tag}");
            check_solution(a.view(), x.view(), b.view(), &tag);
            let x_g = rt::linalg::solve_general((a.view(), b.view()));
            assert!(rt::allclose(x.view(), x_g.view(), Some(1e-12)), "solve_general parity, {tag}");

            // vector b gives the single column
            let x_v = rt::linalg::cho_solve((&factors, &b_vec_rhs));
            assert_eq!(x_v.shape(), &[n], "vector shape, {tag}");
            let residual = (&a % x_v.view() - b_vec_rhs.view()).abs().max_f().unwrap_or(f64::NAN);
            assert!(residual < 1e-12, "vector residual {residual}, {tag}");

            // the nested form and a cholesky result with a manual flag agree
            let x_n = rt::linalg::cho_solve(((factors.c.view(), uplo), &b));
            assert!(rt::allclose(x_n.view(), x.view(), Some(1e-13)), "nested form, {tag}");
            let chol = rt::linalg::cholesky((a.view(), uplo));
            let x_c = rt::linalg::cho_solve(((chol.view(), uplo), &b));
            assert!(rt::allclose(x_c.view(), x.view(), Some(1e-13)), "cholesky factors, {tag}");

            // garbage in the other triangle changes nothing
            let mut dirty = factors.c.clone();
            let junk = |idx: usize| c64::new(((idx * 7 + 3) % 13) as f64 - 6.0, ((idx * 5 + 1) % 9) as f64 - 4.0);
            for idx in 0..n * n {
                let (i, j) = (idx / n, idx % n);
                if match uplo {
                    Lower => i < j,
                    Upper => i > j,
                } {
                    dirty.raw_mut()[idx] = junk(idx);
                }
            }
            let x_d = rt::linalg::cho_solve(((dirty.view(), uplo), &b));
            assert_eq!(x_d.raw().to_vec(), x.raw().to_vec(), "the other triangle is not read, {tag}");
        }
        // the device-order default flag solves as well
        let factors = rt::linalg::cho_factor(a.view());
        let x = rt::linalg::cho_solve((&factors, &b));
        let x_g = rt::linalg::solve_general((a.view(), b.view()));
        assert!(rt::allclose(x.view(), x_g.view(), Some(1e-12)), "default flag, {n}x{n}");
    }
}

/// Every situation of the flip table over complex Hermitian matrices: device
/// order x factor storage order x triangle x vector or matrix `b`; the
/// off-order cells must be bit-equal to the in-order call on the same library
/// (the conjugated-step rewrite is exact).
#[test]
fn test_cho_solve_situations() {
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    let n = 5usize;
    let a = hpd(&a_vec, n);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let a_t = rt::asarray((a.clone(), [n, n].c(), &device)).into_dim::<Ix2>();
        let b_col = rt::asarray((b_vec[..n].to_vec(), [n, 1].c(), &device)).into_dim::<Ix2>();
        let b_mat = rt::asarray((b_vec[..n * 2].to_vec(), [n, 2].c(), &device)).into_dim::<Ix2>();
        for stored in [RowMajor, ColMajor] {
            let a_stored = a_t.view().to_contig_f(stored).unwrap().into_owned().into_dim::<Ix2>();
            for uplo in [Lower, Upper] {
                for (vec_form, b) in [(true, &b_col), (false, &b_mat)] {
                    let tag = format!(
                        "{order:?} device, {stored:?} factor, {uplo:?}, {}",
                        if vec_form { "vector" } else { "matrix" }
                    );
                    let factors = rt::linalg::cho_factor((a_stored.view(), uplo));
                    assert_eq!(factors.uplo, uplo, "flag, {tag}");
                    let x = rt::linalg::cho_solve((&factors, b));
                    assert_eq!(x.shape(), b.shape(), "shape, {tag}");
                    check_solution(a_t.view(), x.view(), b.view().into_dyn(), &tag);
                    let x_g = rt::linalg::solve_general((a_t.view(), b.view()));
                    assert!(rt::allclose(x.view(), x_g.view(), Some(1e-12)), "solve_general, {tag}");
                    // an off-order factor is solved bit-equal to the in-order
                    // call (the conjugated-step rewrite is exact)
                    if stored != order {
                        let a_in = a_t.view().to_contig_f(order).unwrap().into_owned().into_dim::<Ix2>();
                        let factors_in = rt::linalg::cho_factor((a_in.view(), uplo));
                        let x_in = rt::linalg::cho_solve((&factors_in, b));
                        assert_eq!(x_in.shape(), x.shape(), "reference shape, {tag}");
                        for (idx, (v, r)) in x.raw().iter().zip(x_in.raw().iter()).enumerate() {
                            assert_eq!(v.re.to_bits(), r.re.to_bits(), "bit-equal re [{idx}], {tag}");
                            assert_eq!(v.im.to_bits(), r.im.to_bits(), "bit-equal im [{idx}], {tag}");
                        }
                    }
                }
            }
        }
    }
}

/// Batch parity with the per-slice 2-D calls, factor and `b` broadcasts,
/// vector and matrix forms.
#[test]
fn test_cho_solve_nd() {
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    let (n, k) = (4usize, 2usize);
    let gram = hpd(&a_vec, n);
    let shift = c64::new(1.0, 0.0);
    let slot_data = |s: usize| -> Vec<c64> {
        (0..n * n)
            .map(|j| gram[j] + if j / n == j % n { shift * (1.0 + s as f64) } else { c64::new(0.0, 0.0) })
            .collect()
    };
    let slot0 = |_s: usize| slot_data(0);
    for order in [RowMajor, ColMajor] {
        for stored in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let tag = format!("{order:?} device, {stored:?} stack");
            let mk = |v: &[c64], batch: &[usize], inner: &[usize]| -> Tensor<c64, DeviceBLAS, IxD> {
                stack_tensor(v, batch, inner, order, stored, &device)
            };
            let a = stack_of(&slot_data, &[3], &[n, n], order, stored, &device);
            let factors = rt::linalg::cho_factor(&a);
            let b3 = mk(&b_vec[..3 * n * k], &[3], &[n, k]);
            let b1 = mk(&b_vec[..n * k], &[1], &[n, k]);

            // b of batch 1 broadcasts over the factor batch of 3
            let x = rt::linalg::cho_solve((&factors, &b1));
            assert_eq!(x.shape(), stacked(&[3], &[n, k], order).as_slice(), "b broadcast, {tag}");

            // per-slot parity with the 2-D calls, against b1 (the broadcast
            // right-hand side) and against the per-slot b3
            let x3 = rt::linalg::cho_solve((&factors, &b3));
            for i in 0..3 {
                let at = format!("slot {i}, {tag}");
                let a_i = match order {
                    RowMajor => a.i((i, .., ..)),
                    ColMajor => a.i((.., .., i)),
                }
                .into_owned()
                .into_dim::<Ix2>();
                let b_i = match order {
                    RowMajor => b3.i((i, .., ..)),
                    ColMajor => b3.i((.., .., i)),
                }
                .into_owned()
                .into_dim::<Ix2>();
                let b1_i = match order {
                    RowMajor => b1.i((0, .., ..)),
                    ColMajor => b1.i((.., .., 0)),
                }
                .into_owned()
                .into_dim::<Ix2>();
                let factors_i = rt::linalg::cho_factor(a_i.view());
                let x_i = rt::linalg::cho_solve((&factors_i, &b_i));
                let x_i1 = rt::linalg::cho_solve((&factors_i, &b1_i));
                let x_slot = match order {
                    RowMajor => x.i((i, .., ..)),
                    ColMajor => x.i((.., .., i)),
                };
                let x3_slot = match order {
                    RowMajor => x3.i((i, .., ..)),
                    ColMajor => x3.i((.., .., i)),
                };
                assert!(rt::allclose(x_slot, x_i1.view(), Some(1e-11)), "slot parity, {at}");
                assert!(rt::allclose(x3_slot, x_i.view(), Some(1e-11)), "slot parity (b3), {at}");
            }

            // batched vector form
            let b_vec3 = mk(&b_vec[..3 * n], &[3], &[n]);
            let x = rt::linalg::cho_solve((&factors, &b_vec3));
            assert_eq!(x.shape(), b_vec3.shape(), "vector batch shape, {tag}");
            for i in 0..3 {
                let pick = |t: &Tensor<c64, DeviceBLAS, IxD>| -> Tensor<c64, DeviceBLAS, Ix1> {
                    match order {
                        RowMajor => t.i((i, ..)),
                        ColMajor => t.i((.., i)),
                    }
                    .into_owned()
                    .into_dim::<Ix1>()
                };
                let a_i = match order {
                    RowMajor => a.i((i, .., ..)),
                    ColMajor => a.i((.., .., i)),
                }
                .into_owned()
                .into_dim::<Ix2>();
                let factors_i = rt::linalg::cho_factor(a_i.view());
                let reference = rt::linalg::cho_solve((&factors_i, &pick(&b_vec3)));
                assert!(rt::allclose(pick(&x).view(), reference.view(), Some(1e-11)), "vector slot {i}, {tag}");
            }

            // factor of batch 1 broadcasts over the b batch of 3
            let a1 = stack_of(&slot0, &[1], &[n, n], order, stored, &device);
            let factors1 = rt::linalg::cho_factor(&a1);
            let x_f1 = rt::linalg::cho_solve((&factors1, &b3));
            assert_eq!(x_f1.shape(), b3.shape(), "factor broadcast, {tag}");
        }
    }
}

/// In-place solves: `TensorMut` and owned `b` (same buffer), off-order
/// write-back, padded `b` with the padding untouched.
#[test]
fn test_cho_solve_inplace() {
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    let (n, k) = (6usize, 3usize);
    let a_data = hpd(&a_vec, n);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let tag = format!("{order:?}");
        let mk = |v: &[c64], rows: usize, cols: usize, layout: FlagOrder| -> Tensor<c64, DeviceBLAS, Ix2> {
            match layout {
                RowMajor => rt::asarray((v.to_vec(), [rows, cols].c(), &device)).into_dim::<Ix2>(),
                ColMajor => rt::asarray((v.to_vec(), [rows, cols].f(), &device)).into_dim::<Ix2>(),
            }
        };
        let a = mk(&a_data, n, n, order);
        let factors = rt::linalg::cho_factor(a.view());
        let b = mk(&b_vec[..n * k], n, k, order);
        let reference = rt::linalg::cho_solve((&factors, &b));

        // mutable b in the device order: solved in place, the result aliases b
        let mut b_mut = mk(&b_vec[..n * k], n, k, order);
        let ptr = b_mut.raw().as_ptr();
        let x = rt::linalg::cho_solve((&factors, b_mut.view_mut()));
        assert_eq!(x.raw().as_ptr(), ptr, "in-place alias, {tag}");
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-11)), "in-place, {tag}");

        // owned b: consumed
        let x = rt::linalg::cho_solve((&factors, mk(&b_vec[..n * k], n, k, order)));
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-11)), "owned b, {tag}");

        // a mutable b of the other order is solved in a work buffer, written
        // back into b
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        let mut b_off = b.to_contig_f(other).unwrap().into_owned().into_dim::<Ix2>();
        let ptr = b_off.raw().as_ptr();
        let x = rt::linalg::cho_solve((&factors, b_off.view_mut()));
        assert_eq!(x.raw().as_ptr(), ptr, "off-order b alias, {tag}");
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-11)), "off-order b, {tag}");
        drop(x);
        assert!(rt::allclose(b_off.view(), reference.view(), Some(1e-11)), "off-order b written back, {tag}");

        // padded b in the device order: solved in place, the padding untouched
        let (pn, pk) = (n + 2, k + 3);
        let nan = c64::new(f64::NAN, f64::NAN);
        let mut buf = mk(&vec![nan; pn * pk], pn, pk, order);
        buf.i_mut((0..n, 0..k)).assign_f(b.view()).unwrap();
        let ptr = buf.raw().as_ptr();
        let x = rt::linalg::cho_solve((&factors, buf.i_mut((0..n, 0..k)).into_dim::<Ix2>()));
        assert_eq!(x.raw().as_ptr(), ptr, "the solution is the padded b buffer, {tag}");
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-11)), "padded b, {tag}");
        drop(x);
        for i in 0..pn {
            for j in 0..pk {
                if i >= n || j >= k {
                    assert!(buf[[i, j]].is_nan(), "padding ({i}, {j}), {tag}");
                }
            }
        }
    }
}

/// Every overload on a 4x4 block: flat, nested, result struct, kinds, and the
/// fallible variant.
#[test]
fn test_cho_solve_overloads() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    let (n, k) = (4usize, 2usize);
    let a = rt::asarray((hpd(&a_vec, n), [n, n].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((b_vec[..n * k].to_vec(), [n, k].c(), &device)).into_dim::<Ix2>();
    let factors = rt::linalg::cho_factor(&a);
    let reference = rt::linalg::cho_solve((&factors, &b));

    // read-only b, with every kind of factor
    let r = rt::linalg::cho_solve((factors.c.view(), factors.uplo, &b));
    assert!(rt::allclose(r.view(), reference.view(), Some(1e-11)), "(view, flag, &)");
    let r = rt::linalg::cho_solve((&factors.c, factors.uplo, &b));
    assert!(rt::allclose(r.view(), reference.view(), Some(1e-11)), "(&, flag, &)");
    let r = rt::linalg::cho_solve((factors.c.clone(), factors.uplo, &b));
    assert!(rt::allclose(r.view(), reference.view(), Some(1e-11)), "(owned, flag, &)");
    let r = rt::linalg::cho_solve((factors.c.view(), factors.uplo, b.view()));
    assert!(rt::allclose(r.view(), reference.view(), Some(1e-11)), "(view, flag, view)");

    // consumed b: the result is b's buffer
    let mut b_m = b.clone();
    let ptr = b_m.raw().as_ptr();
    let r = rt::linalg::cho_solve((&factors.c, factors.uplo, b_m.view_mut()));
    assert_eq!(r.raw().as_ptr(), ptr, "view_mut b is solved in place");
    assert!(rt::allclose(r.view().to_dyn(), reference.view(), Some(1e-11)), "(&, flag, view_mut)");

    // the SciPy nested form and the result struct, by value and by reference
    let r = rt::linalg::cho_solve(((factors.c.view(), factors.uplo), &b));
    assert!(rt::allclose(r.view(), reference.view(), Some(1e-11)), "((view, flag), &)");
    let r = rt::linalg::cho_solve((rt::linalg::cho_factor(a.view()), &b));
    assert!(rt::allclose(r.view(), reference.view(), Some(1e-11)), "(ChoFactorResult, &)");
    let r = rt::linalg::cho_solve((&factors, &b));
    assert!(rt::allclose(r.view(), reference.view(), Some(1e-11)), "(&ChoFactorResult, &)");
    let mut b_m = b.clone();
    let ptr = b_m.raw().as_ptr();
    let r = rt::linalg::cho_solve((&factors, b_m.view_mut()));
    assert_eq!(r.raw().as_ptr(), ptr, "(&ChoFactorResult, view_mut) is solved in place");

    // the upper factor through the result struct
    let factors_u = rt::linalg::cho_factor((a.view(), Upper));
    let r = rt::linalg::cho_solve((&factors_u, &b));
    let reference_u = rt::linalg::cho_solve((factors_u.c.view(), factors_u.uplo, &b));
    assert!(rt::allclose(r.view(), reference_u.view(), Some(1e-11)), "upper result struct");

    // fallible variant; the factor is only read
    let r = rt::linalg::cho_solve_f((&factors.c, factors.uplo, &b)).unwrap();
    assert!(rt::allclose(r.view(), reference.view(), Some(1e-11)), "cho_solve_f");
    assert_eq!(factors.c.raw().to_vec(), rt::linalg::cho_factor(a.view()).c.raw().to_vec());
}

#[test]
fn test_cho_solve_errors_zero() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    let a = rt::asarray((hpd(&a_vec, 4), [4usize, 4].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((b_vec[..4 * 2].to_vec(), [4usize, 2].c(), &device)).into_dim::<Ix2>();
    let factors = rt::linalg::cho_factor(a.view());

    // the factor must be square
    let a_rect = rt::asarray((&a_vec[..24], [6usize, 4].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::cho_solve_f((a_rect.view(), Lower, &b)).expect_err("non-square must fail"));
    assert!(msg.contains("must be square"), "{msg}");

    // ... and at least 2-D
    let a1 = rt::asarray((&a_vec[..6], [6].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::cho_solve_f((a1.view(), Lower, &b)).expect_err("1-D factor must fail"));
    assert!(msg.contains("at least 2-D"), "{msg}");

    // b's rows must match
    let b_short = rt::asarray((vec![c64!(1.0), c64!(2.0)], [2usize].c(), &device)).into_dim::<Ix1>();
    let msg = format!(
        "{}",
        rt::linalg::cho_solve_f((factors.c.view(), factors.uplo, &b_short)).expect_err("row mismatch must fail")
    );
    assert!(msg.contains("row dimension must match"), "{msg}");

    // the batch shapes must broadcast
    let slot = |_k: usize| -> Vec<c64> { hpd(&a_vec, 4) };
    let a3 = stack_of(&slot, &[3], &[4, 4], RowMajor, RowMajor, &device);
    let factors3 = rt::linalg::cho_factor(a3.view());
    let b_bad = rt::asarray((b_vec[..32].to_vec(), [4usize, 4, 2].c(), &device)).into_dim::<Ix3>();
    let msg = format!(
        "{}",
        rt::linalg::cho_solve_f((factors3.c.view(), factors3.uplo, &b_bad)).expect_err("batch mismatch must fail")
    );
    assert!(msg.contains("roadcast"), "{msg}");

    // an in-place solve cannot grow b's batch of one to the broadcast batch
    let mut b_small = rt::asarray((b_vec[..8].to_vec(), [1usize, 4, 2].c(), &device)).into_dim::<Ix3>();
    let msg = format!(
        "{}",
        rt::linalg::cho_solve_f((factors3.c.view(), factors3.uplo, b_small.view_mut()))
            .expect_err("in-place growth must fail")
    );
    assert!(msg.contains("in-place solve cannot grow"), "{msg}");
    // ... but a read-only b is broadcast
    let x = rt::linalg::cho_solve((factors3.c.view(), factors3.uplo, &b_small));
    assert_eq!(x.shape(), &[3, 4, 2], "read-only broadcast");

    // zero size: N == 0, K == 0, an empty batch
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let a: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let factors = rt::linalg::cho_factor(a.view());
        let b: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 3], &device)).unwrap().into_dim::<Ix2>();
        let x = rt::linalg::cho_solve((&factors, &b));
        assert_eq!(x.shape(), &[0, 3], "N == 0, {order:?}");
        let b: Tensor<c64, DeviceBLAS, Ix1> = rt::zeros_f(([0usize], &device)).unwrap().into_dim::<Ix1>();
        let x = rt::linalg::cho_solve((&factors, &b));
        assert_eq!(x.shape(), &[0], "N == 0 vector, {order:?}");
        let a = rt::asarray((hpd(&a_vec, 2), [2usize, 2].c(), &device)).into_dim::<Ix2>();
        let factors = rt::linalg::cho_factor(a.view());
        let b: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([2usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let x = rt::linalg::cho_solve((&factors, &b));
        assert_eq!(x.shape(), &[2, 0], "K == 0, {order:?}");
        let a: Tensor<c64, DeviceBLAS, IxD> = rt::zeros_f((stacked(&[0], &[2, 2], order), &device)).unwrap();
        let factors = rt::linalg::cho_factor(a.view());
        let b: Tensor<c64, DeviceBLAS, IxD> = rt::zeros_f((stacked(&[0], &[2, 1], order), &device)).unwrap();
        let x = rt::linalg::cho_solve((&factors, &b));
        assert_eq!(x.shape(), b.shape(), "empty batch, {order:?}");
    }
}
