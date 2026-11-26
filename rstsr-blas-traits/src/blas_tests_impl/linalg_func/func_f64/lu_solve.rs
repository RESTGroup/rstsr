use super::*;
use rstsr::prelude::linalg::{LuFactorResult, LuResult};

type LuOut =
    LuResult<Option<Tensor<usize, DeviceBLAS, Ix1>>, Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>>;
type LuFactorOut = LuFactorResult<Tensor<f64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>>;

/// Shape of a stack in the device's axis convention (matrix axes last under
/// `RowMajor`, first under `ColMajor`).
fn stacked(batch: &[usize], inner: &[usize], order: FlagOrder) -> Vec<usize> {
    match order {
        RowMajor => [batch, inner].concat(),
        ColMajor => [inner, batch].concat(),
    }
}

/// Assert that `x` solves `op(A) x = b` within `tol`.
fn check_solution(
    a: &Tensor<f64, DeviceBLAS, Ix2>,
    x: &Tensor<f64, DeviceBLAS, IxD>,
    b: &Tensor<f64, DeviceBLAS, IxD>,
    trans: FlagTrans,
    tol: f64,
    at: &str,
) {
    let residual = (rt::matmul(op_a(a, trans), x.view()) - b.view()).abs().max_f().unwrap();
    assert!(residual < tol, "residual {residual}, {at}");
}

/// Assert that `x` equals the `reference` solution.
fn op_a(a: &Tensor<f64, DeviceBLAS, Ix2>, trans: FlagTrans) -> TensorView<'_, f64, DeviceBLAS, Ix2> {
    match trans {
        NoTrans => a.view(),
        _ => a.transpose(None),
    }
}

fn check_x(x: TensorView<'_, f64, DeviceBLAS, IxD>, reference: TensorView<'_, f64, DeviceBLAS, IxD>, at: &str) {
    assert!(rt::allclose(x, reference, Some(1e-13)), "{at}");
}

#[test]
fn test_lu_solve() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let b_vec = get_vec::<f64>('b');
    let a = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
    let (lu, piv) = (&packed.lu, &packed.piv);

    // K = 1
    let b = rt::asarray((&b_vec[..1024], [1024].c(), &device)).into_dim::<Ix1>();
    let x = rt::linalg::lu_solve((lu, piv, &b));
    assert_eq!(x.shape(), &[1024]);
    assert!((fingerprint(&x) - -9.120066438822086).abs() < 1e-8);

    // K = 3
    let b = rt::asarray((&b_vec[..1024 * 3], [1024, 3].c(), &device)).into_dim::<Ix2>();
    let x = rt::linalg::lu_solve((lu, piv, &b));
    assert_eq!(x.shape(), &[1024, 3]);
    assert!((fingerprint(&x) - 105.25988778050078).abs() < 1e-8);

    // transposed system
    let b = rt::asarray((&b_vec[..1024], [1024].c(), &device)).into_dim::<Ix1>();
    let x = rt::linalg::lu_solve((lu, piv, &b, FlagTrans::T));
    assert_eq!(x.shape(), &[1024]);
    assert!((fingerprint(&x) - 11.360947052714014).abs() < 1e-8);
}

#[test]
fn test_lu_solve_relations() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let b_vec = get_vec::<f64>('b');
    for (n, k) in [(1usize, 1usize), (3, 1), (5, 2), (6, 4), (7, 3)] {
        let a = rt::asarray((a_vec[..n * n].to_vec(), [n, n].c(), &device)).into_dim::<Ix2>();
        let b: Tensor<f64, DeviceBLAS, IxD> =
            rt::asarray((b_vec[..n * k].to_vec(), [n, k].c(), &device)).into_owned().into_dyn();
        let b_vec_rhs = rt::asarray((b_vec[..n].to_vec(), [n].c(), &device)).into_dim::<Ix1>();
        let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
        let factors: LuOut = rt::linalg::lu(a.view());
        for trans in [NoTrans, Trans, ConjTrans] {
            let tag = format!("{n}x{n}, K = {k}, {trans:?}");

            // the packed form and the explicit factor form agree
            let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b, trans));
            assert_eq!(x.shape(), &[n, k], "shape, {tag}");
            check_solution(&a, &x, &b, trans, 1e-12, &tag);
            let x_f = rt::linalg::lu_solve((&factors, &b, trans));
            assert!(rt::allclose(x_f.view(), x.view(), Some(1e-13)), "factor form, {tag}");
            let x_f = rt::linalg::lu_solve((&factors, b.view(), trans));
            assert!(rt::allclose(x_f.view(), x.view(), Some(1e-13)), "factor form, {tag}");

            // the vector form gives the single column
            let x_v = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b_vec_rhs, trans));
            assert_eq!(x_v.shape(), &[n], "vector shape, {tag}");
            let x_col = rt::linalg::lu_solve((
                &packed.lu,
                &packed.piv,
                b_vec_rhs.view().i((.., None)).into_dim::<Ix2>(),
                trans,
            ));
            assert!(rt::allclose(x_v.view(), x_col.i((.., 0)), Some(1e-13)), "vector form, {tag}");
            let residual = (rt::matmul(op_a(&a, trans), x_v.view()) - b_vec_rhs.view()).abs().max_f().unwrap();
            assert!(residual < 1e-12, "vector residual {residual}, {tag}");
        }
        // without the flag the system is A x = b
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
        let x_n = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b, NoTrans));
        assert!(rt::allclose(x.view(), x_n.view(), Some(1e-15)));
        // ... and equals the general solve
        let x_g = rt::linalg::solve_general((&a, &b));
        assert!(rt::allclose(x.view(), x_g.view(), Some(1e-12)), "solve_general parity, {n}x{n}");

        // a permuted lu result (no p) cannot be solved from
        let permuted: LuOut = rt::linalg::lu((a.view(), true));
        let msg = format!("{}", rt::linalg::lu_solve_f((&permuted, &b)).expect_err("permuted factors must fail"));
        assert!(msg.contains("permuted"), "{msg}");
    }
}

#[test]
fn test_lu_solve_batch() {
    let a_vec = get_vec::<f64>('a');
    let b_vec = get_vec::<f64>('b');
    let (n, k) = (4usize, 2usize);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let mk = |v: &[f64], batch: &[usize], inner: &[usize]| -> Tensor<f64, DeviceBLAS, IxD> {
            rt::asarray((v.to_vec(), stacked(batch, inner, order).c(), &device)).into_owned().into_dyn()
        };
        // factors of batch 3, right-hand sides of batch 3
        let a = mk(&a_vec[..3 * n * n], &[3], &[n, n]);
        let packed = rt::linalg::lu_factor(a.view());
        let b3 = mk(&b_vec[..3 * n * k], &[3], &[n, k]);
        // one right-hand side for every factor, and one factor for every
        // right-hand side (a batch axis of size 1 broadcasts)
        let b1 = mk(&b_vec[..n * k], &[1], &[n, k]);
        let a1 = mk(&a_vec[..n * n], &[1], &[n, n]);
        let packed1 = rt::linalg::lu_factor(a1.view());
        // broadcast on both sides: factors (3, 1) against right-hand sides (1, 3)
        let a31 = mk(&a_vec[..3 * n * n], &[3, 1], &[n, n]);
        let b13 = mk(&b_vec[..3 * n * k], &[1, 3], &[n, k]);
        let packed31 = rt::linalg::lu_factor(a31.view());
        let factors31 = rt::linalg::lu(a31.view());

        // shapes follow the broadcast
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b1));
        assert_eq!(x.shape(), stacked(&[3], &[n, k], order).as_slice(), "b broadcast, {order:?}");
        let x = rt::linalg::lu_solve((&packed1.lu, &packed1.piv, &b3));
        assert_eq!(x.shape(), b3.shape(), "factor broadcast, {order:?}");
        let x = rt::linalg::lu_solve((&packed31.lu, &packed31.piv, &b13));
        assert_eq!(x.shape(), stacked(&[3, 3], &[n, k], order).as_slice(), "two-way broadcast, {order:?}");
        let x_f = rt::linalg::lu_solve((&factors31, &b13));
        // a 2-D factor broadcast against 2-D right-hand sides of batch 3
        let factors1 = rt::linalg::lu(a1.view());
        let x_f1 = rt::linalg::lu_solve((&factors1, &b3));
        assert_eq!(x_f1.shape(), b3.shape(), "factor form, factor broadcast, {order:?}");

        // per-slot parity with the 2-D calls, for every pairing (i, j)
        for i in 0..3 {
            for j in 0..3 {
                let at = format!("pair ({i}, {j}), {order:?}");
                let slot = |t: &Tensor<f64, DeviceBLAS, IxD>, axes: (usize, usize)| -> Tensor<f64, DeviceBLAS, Ix2> {
                    let v = match order {
                        RowMajor => t.i((axes.0, axes.1, .., ..)),
                        ColMajor => t.i((.., .., axes.0, axes.1)),
                    };
                    v.into_owned().into_dim::<Ix2>()
                };
                let a_ij = slot(&a31, (i, 0));
                let b_ij = slot(&b13, (0, j));
                let x_ij = slot(&x, (i, j));
                let x_f_ij = slot(&x_f, (i, j));
                let reference = rt::linalg::solve_general((&a_ij, &b_ij));
                assert!(rt::allclose(x_ij.view(), reference.view(), Some(1e-12)), "solution, {at}");
                assert!(rt::allclose(x_f_ij.view(), reference.view(), Some(1e-12)), "factor form, {at}");
            }
        }

        // batched vector form: b (batch_b, N) gives x (batch_b, N)
        let b_vec3 = mk(&b_vec[..3 * n], &[3], &[n]);
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b_vec3));
        assert_eq!(x.shape(), b_vec3.shape(), "vector batch shape, {order:?}");
        for i in 0..3 {
            let pick = |t: &Tensor<f64, DeviceBLAS, IxD>| -> Tensor<f64, DeviceBLAS, Ix1> {
                let v = match order {
                    RowMajor => t.i((i, ..)),
                    ColMajor => t.i((.., i)),
                };
                v.into_owned().into_dim::<Ix1>()
            };
            let a_i = pick2(&a, i, order);
            let packed_i = rt::linalg::lu_factor(a_i.view());
            let reference = rt::linalg::lu_solve((&packed_i.lu, &packed_i.piv, &pick(&b_vec3)));
            assert!(rt::allclose(pick(&x).view(), reference.view(), Some(1e-13)), "vector slot {i}, {order:?}");
        }
    }
}

/// Slot `i` of a batched matrix stack, as an owned 2-D tensor.
fn pick2(t: &Tensor<f64, DeviceBLAS, IxD>, i: usize, order: FlagOrder) -> Tensor<f64, DeviceBLAS, Ix2> {
    let v = match order {
        RowMajor => t.i((i, .., ..)),
        ColMajor => t.i((.., .., i)),
    };
    v.into_owned().into_dim::<Ix2>()
}

#[test]
fn test_lu_solve_inplace() {
    let a_vec = get_vec::<f64>('a');
    let b_vec = get_vec::<f64>('b');
    let (n, k) = (6usize, 3usize);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let mk = |v: &[f64], rows: usize, cols: usize, layout: FlagOrder| -> Tensor<f64, DeviceBLAS, Ix2> {
            match layout {
                RowMajor => rt::asarray((v.to_vec(), [rows, cols].c(), &device)).into_dim::<Ix2>(),
                ColMajor => rt::asarray((v.to_vec(), [rows, cols].f(), &device)).into_dim::<Ix2>(),
            }
        };
        let a = mk(&a_vec[..n * n], n, n, order);
        let packed = rt::linalg::lu_factor(a.view());
        let b = mk(&b_vec[..n * k], n, k, order);
        let reference = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));

        // mutable b in the device order: solved in place, the result aliases b
        let mut b_mut = mk(&b_vec[..n * k], n, k, order);
        let ptr = b_mut.raw().as_ptr();
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, b_mut.view_mut()));
        assert_eq!(x.raw().as_ptr(), ptr, "in-place alias, {order:?}");
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-13)), "in-place, {order:?}");

        // owned b: consumed
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, mk(&b_vec[..n * k], n, k, order)));
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-13)), "owned b, {order:?}");

        // a mutable b of the other order is solved in a work buffer and written back
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        let mut b_off = b.to_contig_f(other).unwrap().into_owned().into_dim::<Ix2>();
        let ptr = b_off.raw().as_ptr();
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, b_off.view_mut()));
        assert_eq!(x.raw().as_ptr(), ptr, "off-order b alias, {order:?}");
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-13)), "off-order b, {order:?}");
        assert!(rt::allclose(b_off.view(), reference.view(), Some(1e-13)), "off-order b written back, {order:?}");

        // padded b (ld > k) in the device order: in place, padding untouched
        let (pn, pk) = (n + 2, k + 3);
        let mut buf = mk(&vec![f64::NAN; pn * pk], pn, pk, order);
        buf.i_mut((0..n, 0..k)).assign_f(b.view()).unwrap();
        let mut b_padded = buf.i((0..n, 0..k)).into_owned_keep_layout().into_dim::<Ix2>();
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, b_padded.view_mut()));
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-13)), "padded b, {order:?}");
        for i in 0..pn {
            for j in 0..pk {
                if i >= n || j >= k {
                    assert!(buf[[i, j]].is_nan(), "padding ({i}, {j}), {order:?}");
                }
            }
        }

        // factors of the other order, padded factors and fully strided factors
        // (a slot of a batch stack) give the same solution
        let packed_off = rt::linalg::lu_factor(a.to_contig_f(other).unwrap().into_owned().into_dim::<Ix2>().view());
        let x = rt::linalg::lu_solve((&packed_off.lu, &packed_off.piv, &b));
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-13)), "off-order factors, {order:?}");
        let q = n + 3;
        let mut fbuf = mk(&vec![0.0; q * q], q, q, order);
        fbuf.i_mut((0..n, 0..n)).assign_f(packed.lu.view()).unwrap();
        let lu_padded = fbuf.i((0..n, 0..n)).into_owned_keep_layout().into_dim::<Ix2>();
        let x = rt::linalg::lu_solve((lu_padded.view(), &packed.piv, &b));
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-13)), "padded factors, {order:?}");
        let (rows, cols) = match order {
            RowMajor => (2 * n, n),
            ColMajor => (n, 2 * n),
        };
        let stack = mk(&packed.lu.raw().to_vec().repeat(2), rows, cols, order);
        let lu_strided = match order {
            RowMajor => stack.i((n..2 * n, ..)),
            ColMajor => stack.i((.., n..2 * n)),
        }
        .into_owned_keep_layout()
        .into_dim::<Ix2>();
        let x = rt::linalg::lu_solve((lu_strided.view(), &packed.piv, &b));
        assert!(rt::allclose(x.view(), reference.view(), Some(1e-13)), "strided factors, {order:?}");
    }
}

#[test]
fn test_lu_solve_overloads() {
    let device = DeviceBLAS::default();
    let (n, k) = (4usize, 2usize);
    let a_data: Vec<f64> =
        (0..n * n).map(|j| ((j * 7 + 3) % 11) as f64 / 3.0 - 1.5 + if j / n == j % n { 3.0 } else { 0.0 }).collect();
    let b_data: Vec<f64> = (0..n * k).map(|j| ((j * 5 + 1) % 7) as f64 / 2.0 - 1.5).collect();
    let make_a = || rt::asarray((a_data.clone(), [n, n].c(), &device)).into_dim::<Ix2>();
    let make_b = || rt::asarray((b_data.clone(), [n, k].c(), &device)).into_dim::<Ix2>();
    let packed = rt::linalg::lu_factor(make_a());
    let factors = rt::linalg::lu(make_a());
    let reference = rt::linalg::lu_solve((&packed.lu, &packed.piv, &make_b()));
    let reference_t = rt::linalg::lu_solve((&packed.lu, &packed.piv, &make_b(), Trans));

    // read-only b, with every kind of lu and piv
    let r = rt::linalg::lu_solve((&packed.lu, &packed.piv, &make_b()));
    check_x(r.view(), reference.view(), "(&, &, &)");
    let r = rt::linalg::lu_solve((&packed.lu, packed.piv.view(), &make_b()));
    check_x(r.view(), reference.view(), "(&, view, &)");
    let r = rt::linalg::lu_solve((packed.lu.view(), &packed.piv, &make_b()));
    check_x(r.view(), reference.view(), "(view, &, &)");
    let r = rt::linalg::lu_solve((packed.lu.view(), packed.piv.view(), &make_b()));
    check_x(r.view(), reference.view(), "(view, view, &)");
    let r = rt::linalg::lu_solve((packed.lu.view(), packed.piv.view(), make_b().view(), NoTrans));
    check_x(r.view(), reference.view(), "(view, view, view, N)");

    // consumed b: the result is b's buffer, for every kind of lu and piv
    let r = rt::linalg::lu_solve((&packed.lu, &packed.piv, make_b())).into_dyn();
    check_x(r.view(), reference.view(), "(&, &, owned)");
    let r = rt::linalg::lu_solve((packed.lu.view(), packed.piv.view(), make_b())).into_dyn();
    check_x(r.view(), reference.view(), "(view, view, owned)");
    let r = rt::linalg::lu_solve((packed.lu.clone(), packed.piv.clone(), make_b())).into_dyn();
    check_x(r.view(), reference.view(), "(owned, owned, owned)");
    let (mut lu_m, mut piv_m) = (packed.lu.clone(), packed.piv.clone());
    let r = rt::linalg::lu_solve((lu_m.view_mut(), piv_m.view_mut(), make_b())).into_dyn();
    check_x(r.view(), reference.view(), "(view_mut, view_mut, owned)");
    let mut b_m = make_b();
    let ptr = b_m.raw().as_ptr();
    let r = rt::linalg::lu_solve((&packed.lu, &packed.piv, b_m.view_mut()));
    assert_eq!(r.raw().as_ptr(), ptr, "view_mut b is solved in place");
    check_x(r.view().to_dyn(), reference.view(), "(&, &, view_mut)");
    let mut b_m = make_b();
    let ptr = b_m.raw().as_ptr();
    let r = rt::linalg::lu_solve((lu_m.view_mut(), &packed.piv, b_m.view_mut()));
    assert_eq!(r.raw().as_ptr(), ptr, "view_mut b is solved in place");
    check_x(r.view().to_dyn(), reference.view(), "(view_mut, &, view_mut)");

    // the transposition flag
    for trans in [NoTrans, Trans, ConjTrans] {
        let reference = rt::linalg::lu_solve((&packed.lu, &packed.piv, &make_b(), trans));
        let r = rt::linalg::lu_solve((&packed.lu, &packed.piv, make_b().view(), trans));
        check_x(r.view(), reference.view(), "flag, read-only b");
        let r = rt::linalg::lu_solve((&packed.lu, &packed.piv, make_b(), trans)).into_dyn();
        check_x(r.view(), reference.view(), "flag, owned b");
    }

    // the explicit factor form of lu, read-only and consumed b, bare and with the flag
    let r = rt::linalg::lu_solve((&factors, &make_b()));
    check_x(r.view(), reference.view(), "(&LuResult, &)");
    let r = rt::linalg::lu_solve((&factors, make_b().view()));
    check_x(r.view(), reference.view(), "(&LuResult, view)");
    let r = rt::linalg::lu_solve((&factors, &make_b(), NoTrans));
    check_x(r.view(), reference.view(), "(&LuResult, &, N)");
    let r = rt::linalg::lu_solve((&factors, make_b())).into_dyn();
    check_x(r.view(), reference.view(), "(&LuResult, owned)");
    let mut b_m = make_b();
    let ptr = b_m.raw().as_ptr();
    let r = rt::linalg::lu_solve((&factors, b_m.view_mut()));
    assert_eq!(r.raw().as_ptr(), ptr);
    check_x(r.view().to_dyn(), reference.view(), "(&LuResult, view_mut)");
    let mut b_m = make_b();
    let r = rt::linalg::lu_solve((&factors, b_m.view_mut(), NoTrans));
    check_x(r.view().to_dyn(), reference.view(), "(&LuResult, view_mut, N)");

    // SciPy's nested form ((lu, piv), b), with every kind of factors and b
    let r = rt::linalg::lu_solve(((packed.lu.view(), packed.piv.view()), &make_b()));
    check_x(r.view(), reference.view(), "((view, view), &)");
    let r = rt::linalg::lu_solve(((&packed.lu, &packed.piv), make_b().view(), Trans));
    check_x(r.view(), reference_t.view(), "((&, &), view, T)");
    let r = rt::linalg::lu_solve(((packed.lu.clone(), packed.piv.clone()), &make_b()));
    check_x(r.view(), reference.view(), "((owned, owned), &)");
    let r = rt::linalg::lu_solve((packed.lu.clone(), packed.piv.view(), &make_b(), Trans));
    check_x(r.view(), reference_t.view(), "(owned, view, &, T)");
    let r = rt::linalg::lu_solve(((packed.lu.clone(), packed.piv.clone()), make_b())).into_dyn();
    check_x(r.view(), reference.view(), "((owned, owned), owned)");
    let mut b_m = make_b();
    let ptr = b_m.raw().as_ptr();
    let r = rt::linalg::lu_solve(((lu_m.view_mut(), piv_m.view_mut()), b_m.view_mut(), NoTrans));
    assert_eq!(r.raw().as_ptr(), ptr);
    check_x(r.view().to_dyn(), reference.view(), "((view_mut, view_mut), view_mut, N)");

    // the result of lu_factor itself, by value and by reference
    let r = rt::linalg::lu_solve((rt::linalg::lu_factor(make_a()), &make_b()));
    check_x(r.view(), reference.view(), "(LuFactorResult, &)");
    let r = rt::linalg::lu_solve((&packed, &make_b(), Trans));
    check_x(r.view(), reference_t.view(), "(&LuFactorResult, &, T)");
    let r = rt::linalg::lu_solve((rt::linalg::lu_factor(make_a()), make_b().view(), NoTrans));
    check_x(r.view(), reference.view(), "(LuFactorResult, view, N)");
    let r = rt::linalg::lu_solve((rt::linalg::lu_factor(make_a()), make_b(), Trans)).into_dyn();
    check_x(r.view(), reference_t.view(), "(LuFactorResult, owned, T)");
    let mut b_m = make_b();
    let ptr = b_m.raw().as_ptr();
    let r = rt::linalg::lu_solve((&packed, b_m.view_mut()));
    assert_eq!(r.raw().as_ptr(), ptr);
    check_x(r.view().to_dyn(), reference.view(), "(&LuFactorResult, view_mut)");

    // fallible variants
    let r = rt::linalg::lu_solve_f((&packed.lu, &packed.piv, &make_b())).unwrap();
    check_x(r.view(), reference.view(), "lu_solve_f");
    let r = rt::linalg::lu_solve_f((&factors, &make_b(), NoTrans)).unwrap();
    check_x(r.view(), reference.view(), "lu_solve_f, factor form");

    // the factors and piv are only read
    assert_eq!(packed.lu.raw().to_vec(), rt::linalg::lu_factor(make_a()).lu.raw().to_vec());
}

#[test]
fn test_lu_solve_singular() {
    let device = DeviceBLAS::default();
    // a singular system: inf / nan, not an error
    let a: Tensor<f64, DeviceBLAS, Ix2> = rt::asarray((vec![1.0, 2.0, 2.0, 4.0], [2usize, 2].c(), &device)).into_dim();
    let packed = rt::linalg::lu_factor(a.view());
    let b: Tensor<f64, DeviceBLAS, Ix2> = rt::asarray((vec![1.0, 2.0, 3.0, 4.0], [2usize, 2].c(), &device)).into_dim();
    let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
    assert_eq!(x.shape(), &[2, 2]);
    assert!(x.raw().iter().any(|v| !v.is_finite()));
    let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b, Trans));
    assert!(x.raw().iter().any(|v| !v.is_finite()));
    // an all-zero matrix: every entry is nan or inf
    let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([3usize, 3], &device)).unwrap().into_dim();
    let packed = rt::linalg::lu_factor(a.view());
    let b: Tensor<f64, DeviceBLAS, Ix2> = rt::ones_f(([3usize, 1], &device)).unwrap().into_dim();
    let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
    assert!(x.raw().iter().all(|v| !v.is_finite()));
}

#[test]
fn test_lu_solve_zero_size() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        // N == 0: empty factors, empty solution
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let packed = rt::linalg::lu_factor(a.view());
        let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 3], &device)).unwrap().into_dim::<Ix2>();
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
        assert_eq!(x.shape(), &[0, 3], "N == 0, {order:?}");
        let b: Tensor<f64, DeviceBLAS, Ix1> = rt::zeros_f(([0usize], &device)).unwrap().into_dim::<Ix1>();
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
        assert_eq!(x.shape(), &[0], "N == 0 vector, {order:?}");
        let factors = rt::linalg::lu(a.view());
        let x = rt::linalg::lu_solve((&factors, &b));
        assert_eq!(x.shape(), &[0], "N == 0 vector, factor form, {order:?}");
        // K == 0: no right-hand sides
        let a = rt::asarray((vec![2.0, 1.0, 1.0, 3.0], [2usize, 2].c(), &device)).into_dim::<Ix2>();
        let packed = rt::linalg::lu_factor(a.view());
        let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([2usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
        assert_eq!(x.shape(), &[2, 0], "K == 0, {order:?}");
        // an empty batch
        let a: Tensor<f64, DeviceBLAS, IxD> = rt::zeros_f((stacked(&[0], &[2, 2], order), &device)).unwrap();
        let packed = rt::linalg::lu_factor(a.view());
        let b: Tensor<f64, DeviceBLAS, IxD> = rt::zeros_f((stacked(&[0], &[2, 1], order), &device)).unwrap();
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
        assert_eq!(x.shape(), b.shape());
    }
}

#[test]
fn test_lu_solve_errors() {
    let device = DeviceBLAS::default();
    let a = rt::asarray(((0..36).map(|j| (j % 7) as f64 + 1.0).collect::<Vec<_>>(), [6usize, 6].c(), &device))
        .into_dim::<Ix2>();
    let b =
        rt::asarray(((0..18).map(|j| (j % 5) as f64).collect::<Vec<_>>(), [6usize, 3].c(), &device)).into_dim::<Ix2>();
    let packed = rt::linalg::lu_factor(a.view());

    // the factorization must be square
    let a_rect =
        rt::asarray(((0..24).map(|j| (j % 7) as f64).collect::<Vec<_>>(), [6usize, 4].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::lu_solve_f((&a_rect, &packed.piv, &b)).expect_err("non-square must fail"));
    assert!(msg.contains("must be square"), "{msg}");
    // ... and at least 2-D
    let a1 = rt::asarray(((0..6).map(|j| (j % 7) as f64).collect::<Vec<_>>(), [6usize].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lu_solve_f((a1.view(), &packed.piv, &b)).expect_err("1-D factors must fail"));
    assert!(msg.contains("at least 2-D"), "{msg}");

    // piv must hold exactly N entries per slice
    let piv_short = packed.piv.i(..5).into_owned().into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lu_solve_f((&packed.lu, &piv_short, &b)).expect_err("short piv must fail"));
    assert!(msg.contains("piv must have"), "{msg}");

    // b's rows must match
    let b_short = rt::asarray((vec![1.0, 2.0], [2usize].c(), &device)).into_dim::<Ix1>();
    let msg =
        format!("{}", rt::linalg::lu_solve_f((&packed.lu, &packed.piv, &b_short)).expect_err("row mismatch must fail"));
    assert!(msg.contains("row dimension must match"), "{msg}");

    // the batch shapes of the factors and b must broadcast
    let a3 = rt::asarray(((0..72).map(|j| (j % 7) as f64 + 1.0).collect::<Vec<_>>(), [2usize, 3, 3].c(), &device))
        .into_dim::<Ix3>();
    let packed3 = rt::linalg::lu_factor(a3.view());
    let b_bad = rt::asarray(((0..24).map(|j| (j % 5) as f64).collect::<Vec<_>>(), [3usize, 3, 1].c(), &device))
        .into_dim::<Ix3>();
    let msg = format!(
        "{}",
        rt::linalg::lu_solve_f((&packed3.lu, &packed3.piv, &b_bad)).expect_err("batch mismatch must fail")
    );
    assert!(msg.contains("roadcast"), "{msg}");

    // an in-place solve cannot grow b's batch of one to the broadcast batch of two
    let mut b_small = rt::asarray(((0..6).map(|j| (j % 5) as f64).collect::<Vec<_>>(), [1usize, 3, 2].c(), &device))
        .into_dim::<Ix3>();
    let msg = format!(
        "{}",
        rt::linalg::lu_solve_f((&packed3.lu, &packed3.piv, b_small.view_mut())).expect_err("in-place growth must fail")
    );
    assert!(msg.contains("in-place solve cannot grow"), "{msg}");
    // ... but a read-only b is broadcast
    let x = rt::linalg::lu_solve((&packed3.lu, &packed3.piv, &b_small));
    assert_eq!(x.shape(), &[2, 3, 2]);

    // l and u of the explicit factors must share one shape
    let factors: LuOut = rt::linalg::lu(a.view());
    let msg = format!(
        "{}",
        rt::linalg::lu_solve_f((
            &LuResult {
                p: factors.p,
                l: factors.l.clone(),
                u: factors.u.i((..5, ..)).into_owned_keep_layout().into_dim::<Ix2>()
            },
            &b
        ))
        .expect_err("mismatched l and u must fail")
    );
    assert!(msg.contains("same shape"), "{msg}");
}
