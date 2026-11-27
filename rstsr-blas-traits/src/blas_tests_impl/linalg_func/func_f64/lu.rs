use super::*;
use rstsr::prelude::linalg::{LuFactorResult, LuResult};

type LuOut =
    LuResult<Option<Tensor<usize, DeviceBLAS, IxD>>, Tensor<f64, DeviceBLAS, IxD>, Tensor<f64, DeviceBLAS, IxD>>;
// the result types follow the dims of a 2-D input
type LuOut2 =
    LuResult<Option<Tensor<usize, DeviceBLAS, Ix1>>, Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>>;
type LuTuple2 = (Tensor<usize, DeviceBLAS, Ix1>, Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>);
type LuFactorOut2 = LuFactorResult<Tensor<f64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>>;

/// Rows `p` of `l` stacked: `l[p, :]`.
fn gather_rows(l: &Tensor<f64, DeviceBLAS, Ix2>, p: &Tensor<usize, DeviceBLAS, Ix1>) -> Tensor<f64, DeviceBLAS, Ix2> {
    let mut lp: Tensor<f64, DeviceBLAS, Ix2> =
        rt::zeros_f(([l.shape()[0], l.shape()[1]], l.device())).unwrap().into_dim();
    for i in 0..p.shape()[0] {
        lp.i_mut((i, ..)).assign_f(l.i((p.raw()[i], ..))).unwrap();
    }
    lp
}

/// The permutation `p` implied by the pivot sequence `piv`: the swaps applied in reverse order to
/// `0..m`.
fn pivots_to_perm(piv: &[usize], m: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..m).collect();
    for i in (0..piv.len()).rev() {
        perm.swap(i, piv[i]);
    }
    perm
}

/// Check the structural properties of one `lu` result against `a`.
fn check_lu(result: &LuOut2, a: &TensorView<'_, f64, DeviceBLAS, Ix2>, at: &str) {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let k = m.min(n);
    let p = result.p.as_ref().expect("p must be present");
    let (l, u) = (&result.l, &result.u);
    assert_eq!(l.shape(), &[m, k], "l shape, {at}");
    assert_eq!(u.shape(), &[k, n], "u shape, {at}");
    let mut sorted = p.raw().to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..m).collect::<Vec<_>>(), "p is a permutation, {at}");
    assert!(rt::allclose(rt::matmul(gather_rows(l, p), u.view()), a.view(), Some(1e-12)), "A = L[P] @ U, {at}");
    for i in 0..k {
        assert_eq!(l[[i, i]], 1.0, "unit diagonal of l, {at}");
        for j in (i + 1)..k {
            assert_eq!(l[[i, j]], 0.0, "zeros above the diagonal of l, {at}");
        }
        for ii in (i + 1)..k {
            assert_eq!(u[[ii, i]], 0.0, "zeros below the diagonal of u, {at}");
        }
    }
}

#[test]
fn test_lu() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');

    // square 1024 x 1024
    let a = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let (p, l, u): (Tensor<usize, DeviceBLAS, Ix1>, Tensor<f64, DeviceBLAS, Ix2>, _) = rt::linalg::lu(a.view()).into();
    assert_eq!(p.shape(), &[1024]);
    assert_eq!(l.shape(), &[1024, 1024]);
    assert_eq!(u.shape(), &[1024, 1024]);
    assert!((fingerprint(&p.map(|&v| v as f64)) - -10740.895650398168).abs() < 1e-8);
    assert!((fingerprint(&l) - 57.86561443958445).abs() < 1e-8);
    assert!((fingerprint(&u) - 5339.538409894577).abs() < 1e-8);

    // tall 1024 x 512: l is (M, K), u is (K, N)
    let a = rt::asarray((&a_vec[..1024 * 512], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let (p, l, u): (Tensor<usize, DeviceBLAS, Ix1>, Tensor<f64, DeviceBLAS, Ix2>, _) = rt::linalg::lu(a.view()).into();
    assert_eq!(p.shape(), &[1024]);
    assert_eq!(l.shape(), &[1024, 512]);
    assert_eq!(u.shape(), &[512, 512]);
    assert!((fingerprint(&p.map(|&v| v as f64)) - 7700.11297530026).abs() < 1e-8);
    assert!((fingerprint(&l) - -34.02884676033965).abs() < 1e-8);
    assert!((fingerprint(&u) - -697.3808817705896).abs() < 1e-8);

    // wide 512 x 1024: l is (K, K), u is (K, N)
    let a = rt::asarray((&a_vec[..512 * 1024], [512, 1024].c(), &device)).into_dim::<Ix2>();
    let (p, l, u): (Tensor<usize, DeviceBLAS, Ix1>, Tensor<f64, DeviceBLAS, Ix2>, _) = rt::linalg::lu(a.view()).into();
    assert_eq!(p.shape(), &[512]);
    assert_eq!(l.shape(), &[512, 512]);
    assert_eq!(u.shape(), &[512, 1024]);
    assert!((fingerprint(&p.map(|&v| v as f64)) - -1288.6783832588026).abs() < 1e-8);
    assert!((fingerprint(&l) - -49.97322299293718).abs() < 1e-8);
    assert!((fingerprint(&u) - -2688.4884623865323).abs() < 1e-8);

    // permute_l: (P L, U) without p
    let a = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let (pl, u): (Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>) = rt::linalg::lu((a.view(), true)).into();
    assert!((fingerprint(&pl) - -171.8956779075001).abs() < 1e-8);
    assert!((fingerprint(&u) - 5339.538409894577).abs() < 1e-8);
}

#[test]
fn test_lu_relations() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    for (m, n) in [(5usize, 5usize), (6, 4), (3, 5), (1, 1), (4, 1), (1, 4), (2, 3)] {
        let k = m.min(n);
        let a = rt::asarray((&a_vec[..m * n], [m, n].c(), &device)).into_dim::<Ix2>();
        let at = format!("{m}x{n}");

        let result: LuOut2 = rt::linalg::lu(a.view());
        check_lu(&result, &a.view(), &at);
        let p = result.p.as_ref().unwrap();

        // rows K..M of a tall l hold multipliers: l is not unit-triangular there
        if m > n {
            assert!(result.l.i((n..m, ..)).view().abs().sum_f().unwrap() > 0.0, "tall multipliers, {at}");
        }

        // permute_l gives P L (rows p of l) and the same u
        let permuted: LuOut2 = rt::linalg::lu((a.view(), true));
        assert!(permuted.p.is_none(), "{at}");
        assert!(rt::allclose(gather_rows(&result.l, p), permuted.l.view(), Some(1e-15)), "PL rows, {at}");
        assert!(rt::allclose(&permuted.u, &result.u, Some(1e-15)), "u unchanged, {at}");
        assert!(rt::allclose(rt::matmul(&permuted.l, &permuted.u), a.view(), Some(1e-12)), "A = PL @ U, {at}");

        // the packed factorization holds the same factors: U on and above the
        // diagonal, the multipliers of L below it; its pivots imply p
        let packed: LuFactorOut2 = rt::linalg::lu_factor(a.view());
        assert_eq!(packed.piv.shape(), &[k], "piv shape, {at}");
        assert_eq!(pivots_to_perm(&packed.piv.to_vec(), m), p.raw().to_vec(), "piv implies p, {at}");
        for i in 0..m.min(k) {
            for j in 0..n.min(k) {
                let expected = if i > j { result.l[[i, j]] } else { result.u[[i, j]] };
                assert!((packed.lu[[i, j]] - expected).abs() < 1e-15, "packed lu ({i}, {j}), {at}");
            }
        }
        for i in k..m {
            for j in 0..n {
                assert!((packed.lu[[i, j]] - result.l[[i, j]]).abs() < 1e-15, "packed multipliers ({i}, {j}), {at}");
            }
        }
    }
}

#[test]
fn test_lu_batch() {
    let a_vec = get_vec::<f64>('a');
    let (m, n) = (5usize, 4usize);
    // the stack is stored in the memory order of the device (in order) or in
    // the other one
    for order in [RowMajor, ColMajor] {
        for stored in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let tag = format!("{order:?} device, {stored:?} stack");
            let data = &a_vec[..2 * 3 * m * n];
            let a = stack_tensor(data, &[2, 3], &[m, n], order, stored, &device);
            assert_eq!(a.f_contig(), stored == ColMajor, "stack layout, {tag}");
            assert_eq!(a.c_contig(), stored == RowMajor, "stack layout, {tag}");

            let result: LuOut = rt::linalg::lu(a.view());
            let (pl_shape, u_shape): (Vec<usize>, Vec<usize>) = match order {
                RowMajor => (vec![2, 3, m, m.min(n)], vec![2, 3, m.min(n), n]),
                ColMajor => (vec![m, m.min(n), 2, 3], vec![m.min(n), n, 2, 3]),
            };
            assert_eq!(result.p.as_ref().unwrap().shape(), &match order {
                RowMajor => vec![2, 3, m],
                ColMajor => vec![m, 2, 3],
            });
            assert_eq!(result.l.shape(), &pl_shape[..], "l shape, {tag}");
            assert_eq!(result.u.shape(), &u_shape[..], "u shape, {tag}");
            for i in 0..2 {
                for j in 0..3 {
                    let at = format!("slot ({i}, {j}), {tag}");
                    let (a_ij, p_ij, l_ij, u_ij) = match order {
                        RowMajor => (
                            a.i((i, j, .., ..)),
                            result.p.as_ref().unwrap().i((i, j, ..)),
                            result.l.i((i, j, .., ..)),
                            result.u.i((i, j, .., ..)),
                        ),
                        ColMajor => (
                            a.i((.., .., i, j)),
                            result.p.as_ref().unwrap().i((.., i, j)),
                            result.l.i((.., .., i, j)),
                            result.u.i((.., .., i, j)),
                        ),
                    };
                    let a_ij = a_ij.into_dim::<Ix2>();
                    let single: LuOut2 = rt::linalg::lu(a_ij.view());
                    check_lu(&single, &a_ij, &at);
                    assert_eq!(p_ij.to_vec(), single.p.as_ref().unwrap().raw().to_vec(), "p, {at}");
                    assert!(rt::allclose(l_ij, single.l.view(), Some(1e-14)), "l, {at}");
                    assert!(rt::allclose(u_ij, single.u.view(), Some(1e-14)), "u, {at}");
                }
            }
            let permuted: LuOut = rt::linalg::lu((a.view(), true));
            assert!(permuted.p.is_none());
            assert_eq!(permuted.l.shape(), &pl_shape[..], "PL shape, {tag}");

            // a mutable stack gives the same factors; it is destroyed in place
            // when stored in the device order, and left untouched otherwise
            let mut a_mut = stack_tensor(data, &[2, 3], &[m, n], order, stored, &device);
            let before = a_mut.raw().to_vec();
            let from_mut: LuOut = rt::linalg::lu(a_mut.view_mut());
            assert_eq!(
                from_mut.p.as_ref().unwrap().raw().to_vec(),
                result.p.as_ref().unwrap().raw().to_vec(),
                "p, mutable {tag}"
            );
            assert!(rt::allclose(from_mut.l.view(), result.l.view(), Some(1e-14)), "l, mutable {tag}");
            assert!(rt::allclose(from_mut.u.view(), result.u.view(), Some(1e-14)), "u, mutable {tag}");
            assert_eq!(a_mut.raw() != &before, stored == order, "destructive rule, {tag}");
        }
    }
}

#[test]
fn test_lu_layouts() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let (m, n) = (7usize, 5usize);
    let a = rt::asarray((&a_vec[..m * n], [m, n].c(), &device)).into_dim::<Ix2>();
    let reference: LuOut2 = rt::linalg::lu(a.view());

    // the same values stored in Fortran order (the other order on this device)
    let mut fortran = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            fortran[j * m + i] = a_vec[i * n + j];
        }
    }
    let a_f = rt::asarray((&fortran[..], [m, n].f(), &device)).into_dim::<Ix2>();
    let from_f: LuOut2 = rt::linalg::lu(a_f.view());
    assert_eq!(from_f.p.as_ref().unwrap().raw().to_vec(), reference.p.as_ref().unwrap().raw().to_vec());
    assert!(rt::allclose(&from_f.l, &reference.l, Some(1e-14)), "f-contiguous l");
    assert!(rt::allclose(&from_f.u, &reference.u, Some(1e-14)), "f-contiguous u");

    // padded rows and columns in a bigger buffer (ld > n), sliced on both axes
    let (pm, pn) = (m + 3, n + 2);
    let mut padded: Vec<f64> = (0..pm * pn).map(|j| ((j * 7 + 3) % 11) as f64).collect();
    for i in 0..m {
        for j in 0..n {
            padded[i * pn + j] = a_vec[i * n + j];
        }
    }
    let buf = rt::asarray((padded, [pm, pn].c(), &device)).into_dim::<Ix2>();
    let a_strided = buf.i((0..m, 0..n)).into_owned_keep_layout().into_dim::<Ix2>();
    assert!(!a_strided.f_contig() && !a_strided.c_contig());
    let from_strided: LuOut2 = rt::linalg::lu(a_strided.view());
    assert_eq!(from_strided.p.as_ref().unwrap().raw().to_vec(), reference.p.as_ref().unwrap().raw().to_vec());
    assert!(rt::allclose(&from_strided.l, &reference.l, Some(1e-14)), "strided l");
    assert!(rt::allclose(&from_strided.u, &reference.u, Some(1e-14)), "strided u");

    // a view of a batch slot
    let stack = rt::asarray((a_vec[..2 * m * n].to_vec(), [2, m, n].c(), &device)).into_dim::<Ix3>();
    let from_slot: LuOut2 = rt::linalg::lu(stack.i((1, .., ..)).into_dim::<Ix2>());
    let single: LuOut2 = rt::linalg::lu(stack.i((1, .., ..)).into_owned().into_dim::<Ix2>().view());
    assert!(rt::allclose(&from_slot.l, &single.l, Some(1e-14)), "batch slot l");
    assert!(rt::allclose(&from_slot.u, &single.u, Some(1e-14)), "batch slot u");
}

#[test]
fn test_lu_inplace() {
    let (m, n) = (6usize, 5usize);
    let a_data: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 / 3.0 - 1.5).collect();
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let mk = |v: &[f64], rows: usize, cols: usize, layout: FlagOrder| -> Tensor<f64, DeviceBLAS, Ix2> {
            match layout {
                RowMajor => rt::asarray((v.to_vec(), [rows, cols].c(), &device)).into_dim::<Ix2>(),
                ColMajor => rt::asarray((v.to_vec(), [rows, cols].f(), &device)).into_dim::<Ix2>(),
            }
        };
        let a = mk(&a_data, m, n, order);
        let reference: LuOut2 = rt::linalg::lu(a.view());
        let check = |result: &LuOut2, at: &str| {
            check_lu(result, &a.view(), &format!("{at}, {order:?}"));
            assert!(rt::allclose(&result.l, &reference.l, Some(1e-13)), "l, {at}, {order:?}");
            assert!(rt::allclose(&result.u, &reference.u, Some(1e-13)), "u, {at}, {order:?}");
        };

        // owned and mutable in-order inputs give the same factors
        let owned: LuOut2 = rt::linalg::lu(mk(&a_data, m, n, order));
        check(&owned, "owned");
        let mut a_mut = mk(&a_data, m, n, order);
        let from_mut: LuOut2 = rt::linalg::lu(a_mut.view_mut());
        check(&from_mut, "mutable");
        // the mutable input is destroyed: it holds the packed factors of the call layout
        let packed = rt::linalg::lu_factor(mk(&a_data, m, n, order)).lu;
        for i in 0..m {
            for j in 0..n {
                assert!((a_mut[[i, j]] - packed[[i, j]]).abs() < 1e-13, "destroyed a ({i}, {j}), {order:?}");
            }
        }

        // a mutable input of the other order is factored from a work buffer and left untouched
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        let mut a_off = a.to_contig_f(other).unwrap().into_owned().into_dim::<Ix2>();
        let a_off_data = a_off.raw().to_vec();
        let from_off: LuOut2 = rt::linalg::lu(a_off.view_mut());
        check(&from_off, "off-order");
        assert_eq!(a_off.raw().to_vec(), a_off_data, "off-order a untouched, {order:?}");

        // padded in-order input (a mutable view of a buffer with ld > n):
        // consumed in place, the padding is not touched
        let (pm, pn) = (m + 4, n + 4);
        let mut buf = mk(&vec![f64::NAN; pm * pn], pm, pn, order);
        buf.i_mut((0..m, 0..n)).assign_f(a.view()).unwrap();
        let from_padded: LuOut2 = rt::linalg::lu(buf.i_mut((0..m, 0..n)).into_dim::<Ix2>());
        check(&from_padded, "padded");
        for i in 0..pm {
            for j in 0..pn {
                if i >= m || j >= n {
                    assert!(buf[[i, j]].is_nan(), "padding ({i}, {j}), {order:?}");
                } else {
                    assert!((buf[[i, j]] - packed[[i, j]]).abs() < 1e-13, "destroyed padded a ({i}, {j}), {order:?}");
                }
            }
        }
    }
}

#[test]
fn test_lu_overloads() {
    let device = DeviceBLAS::default();
    let (m, n) = (4usize, 4usize);
    let a_data: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 / 3.0 - 1.5).collect();
    let make = || rt::asarray((a_data.clone(), [m, n].c(), &device)).into_dim::<Ix2>();
    let reference: LuOut2 = rt::linalg::lu(&make());
    let permuted_reference: LuOut2 = rt::linalg::lu((&make(), true));
    let check = |result: &LuOut2| {
        assert!(result.p.is_some(), "p must be present");
        assert!(rt::allclose(&result.l, &reference.l, Some(1e-14)));
        assert!(rt::allclose(&result.u, &reference.u, Some(1e-14)));
        assert_eq!(result.p.as_ref().unwrap().raw().to_vec(), reference.p.as_ref().unwrap().raw().to_vec());
    };
    let check_permuted = |result: &LuOut2| {
        assert!(result.p.is_none(), "p must be absent");
        assert!(rt::allclose(&result.l, &permuted_reference.l, Some(1e-14)));
        assert!(rt::allclose(&result.u, &permuted_reference.u, Some(1e-14)));
    };

    // the four input kinds, bare and with permute_l
    check(&rt::linalg::lu(&make()));
    check(&rt::linalg::lu((&make(), false)));
    check(&rt::linalg::lu(make().view()));
    check(&rt::linalg::lu((make().view(), false)));
    check(&rt::linalg::lu(make()));
    check(&rt::linalg::lu((make(), false)));
    check(&rt::linalg::lu(make().view_mut()));
    check(&rt::linalg::lu((make().view_mut(), false)));
    check_permuted(&rt::linalg::lu((&make(), true)));
    check_permuted(&rt::linalg::lu((make().view(), true)));
    check_permuted(&rt::linalg::lu((make(), true)));
    check_permuted(&rt::linalg::lu((make().view_mut(), true)));
    check(&rt::linalg::lu_f(&make()).unwrap());
    check_permuted(&rt::linalg::lu_f((&make(), true)).unwrap());

    // the result converts to (p, l, u), a permuted one to (pl, u)
    let (p, l, u): (Tensor<usize, DeviceBLAS, Ix1>, Tensor<f64, DeviceBLAS, Ix2>, _) = reference.into();
    assert_eq!(p.shape(), &[m]);
    assert_eq!(l.shape(), &[m, n]);
    assert_eq!(u.shape(), &[m, n]);
    let (pl, u): (Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>) = rt::linalg::lu((&make(), true)).into();
    assert_eq!(pl.shape(), &[m, n]);
    assert_eq!(u.shape(), &[m, n]);

    // static dims: p drops one axis
    let a = make();
    let result: LuOut2 = rt::linalg::lu(a.view());
    assert_eq!(result.p.unwrap().shape(), &[m]);
    let result = rt::linalg::lu(a.view());
    let p: Tensor<usize, DeviceBLAS, Ix1> = result.p.unwrap();
    let l: Tensor<f64, DeviceBLAS, Ix2> = result.l;
    assert_eq!(p.shape(), &[m]);
    assert_eq!(l.shape(), &[m, n]);
}

// the two rejected tuple conversions (see `test_lu_overloads`)

#[test]
#[should_panic(expected = "LuResult: the factors are permuted (permute_l); convert to (L, U) instead")]
fn test_lu_result_rejects_triple_when_permuted() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((vec![2.0, 1.0, 1.0, 3.0], [2usize, 2].c(), &device)).into_dim::<Ix2>();
    let permuted: LuOut2 = rt::linalg::lu((a.view(), true));
    let _: LuTuple2 = permuted.into();
}

#[test]
#[should_panic(expected = "LuResult: the factors are not permuted; convert to (P, L, U) instead")]
fn test_lu_result_rejects_pair_when_unpermuted() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((vec![2.0, 1.0, 1.0, 3.0], [2usize, 2].c(), &device)).into_dim::<Ix2>();
    let factors: LuOut2 = rt::linalg::lu(a.view());
    let _: (Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>) = factors.into();
}

#[test]
fn test_lu_singular() {
    let device = DeviceBLAS::default();
    // rank-deficient matrices are factored without an error
    let a =
        rt::asarray((vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0], [3usize, 3].c(), &device)).into_dim::<Ix2>();
    let result: LuOut2 = rt::linalg::lu(a.view());
    check_lu(&result, &a.view(), "singular");
    let diag: Vec<f64> = (0..3).map(|i| result.u[[i, i]]).collect();
    assert!(diag.contains(&0.0), "an exactly-zero pivot on the diagonal of u, {diag:?}");

    // an all-zero matrix
    let a = rt::zeros_f(([3usize, 3], &device)).unwrap().into_dim::<Ix2>();
    let result: LuOut2 = rt::linalg::lu(a.view());
    check_lu(&result, &a.view(), "zero matrix");
    assert_eq!(result.p.unwrap().raw().to_vec(), vec![0, 1, 2]);
    assert!(result.l.raw().iter().enumerate().all(|(idx, v)| {
        let (r, c) = (idx / 3, idx % 3);
        (r == c && *v == 1.0) || (r != c && *v == 0.0)
    }));
    assert!(result.u.raw().iter().all(|v| *v == 0.0));

    // lu_factor is silent as well; lu_solve of the singular system is not finite
    let packed: LuFactorOut2 = rt::linalg::lu_factor(a.view());
    assert!(packed.lu.raw().iter().all(|v| *v == 0.0));
    let b = rt::ones_f(([3usize, 2], &device)).unwrap().into_dim::<Ix2>();
    let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, b.view()));
    assert!(x.raw().iter().any(|v| !v.is_finite()));
}

#[test]
fn test_lu_zero_size() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);

        // (0, 0), (0, N) and (M, 0) matrices: p the identity, zero l and u
        for (m, n) in [(0usize, 0usize), (0, 4), (3, 0)] {
            let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([m, n], &device)).unwrap().into_dim::<Ix2>();
            let result: LuOut2 = rt::linalg::lu(a.view());
            assert_eq!(result.p.as_ref().unwrap().raw().to_vec(), (0..m).collect::<Vec<_>>(), "{m}x{n}, {order:?}");
            assert_eq!(result.l.shape(), &[m, m.min(n)]);
            assert_eq!(result.u.shape(), &[m.min(n), n]);
            let permuted: LuOut2 = rt::linalg::lu((a.view(), true));
            assert!(permuted.p.is_none());
            let packed: LuFactorOut2 = rt::linalg::lu_factor(a.view());
            assert_eq!(packed.lu.shape(), &[m, n]);
            assert_eq!(packed.piv.shape(), &[m.min(n)]);
        }

        // an empty batch
        let shape = match order {
            RowMajor => vec![0usize, 3, 3],
            ColMajor => vec![3usize, 3, 0],
        };
        let a = rt::zeros_f((shape, &device)).unwrap();
        let result: LuOut = rt::linalg::lu(a.view());
        assert_eq!(result.l.shape(), a.shape());
        assert_eq!(result.p.unwrap().shape().len(), a.shape().len() - 1);
    }
}

#[test]
fn test_lu_errors() {
    let device = DeviceBLAS::default();
    // fewer than two axes
    let v = rt::asarray((vec![1.0, 2.0, 3.0], [3usize].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lu_f(v.view()).err().expect("1-D a must fail"));
    assert!(msg.contains("expected at least 2 dimensions"), "{msg}");
}
