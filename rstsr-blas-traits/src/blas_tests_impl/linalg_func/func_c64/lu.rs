use super::*;
use rstsr::prelude::linalg::{LuFactorResult, LuResult};

// the result types follow the dims of a 2-D input
type LuOut =
    LuResult<Option<Tensor<usize, DeviceBLAS, Ix1>>, Tensor<c64, DeviceBLAS, Ix2>, Tensor<c64, DeviceBLAS, Ix2>>;
type LuFactorOut = LuFactorResult<Tensor<c64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>>;
type LuOutDyn =
    LuResult<Option<Tensor<usize, DeviceBLAS, IxD>>, Tensor<c64, DeviceBLAS, IxD>, Tensor<c64, DeviceBLAS, IxD>>;

#[test]
fn test_lu() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');

    // square 1024 x 1024
    let a = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let (p, l, u): (Tensor<usize, DeviceBLAS, Ix1>, Tensor<c64, DeviceBLAS, Ix2>, _) = rt::linalg::lu(a.view()).into();
    assert_eq!(p.shape(), &[1024]);
    assert_eq!(l.shape(), &[1024, 1024]);
    assert_eq!(u.shape(), &[1024, 1024]);
    assert!((fingerprint(&p.map(|&v| v as f64)) - 10021.799952379515).abs() < 1e-8);
    assert!((fingerprint(&l.abs()) - 101.3884312729007).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -2781.9082366770954).abs() < 1e-8);

    // tall 1024 x 512: l is (M, K), u is (K, N)
    let a = rt::asarray((&a_vec[..1024 * 512], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let (p, l, u): (Tensor<usize, DeviceBLAS, Ix1>, Tensor<c64, DeviceBLAS, Ix2>, _) = rt::linalg::lu(a.view()).into();
    assert_eq!(p.shape(), &[1024]);
    assert_eq!(l.shape(), &[1024, 512]);
    assert_eq!(u.shape(), &[512, 512]);
    assert!((fingerprint(&p.map(|&v| v as f64)) - -358.16906335896806).abs() < 1e-8);
    assert!((fingerprint(&l.abs()) - 93.90580050538996).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -350.42517036956383).abs() < 1e-8);

    // permute_l: (P L, U) without p
    let a = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let (pl, u): (Tensor<c64, DeviceBLAS, Ix2>, Tensor<c64, DeviceBLAS, Ix2>) = rt::linalg::lu((a.view(), true)).into();
    assert!((fingerprint(&pl.abs()) - -78.03219179967243).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -2781.9082366770954).abs() < 1e-8);
}

#[test]
fn test_lu_relations() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    for (m, n) in [(5usize, 5usize), (6, 4), (3, 5), (1, 4), (4, 1)] {
        let k = m.min(n);
        let at = format!("{m}x{n}");
        let a = rt::asarray((a_vec[..m * n].to_vec(), [m, n].c(), &device)).into_dim::<Ix2>();
        let result: LuOut = rt::linalg::lu(a.view());
        let p = result.p.as_ref().unwrap();
        let (l, u) = (&result.l, &result.u);
        assert_eq!(l.shape(), &[m, k], "l shape, {at}");
        assert_eq!(u.shape(), &[k, n], "u shape, {at}");
        let mut sorted = p.raw().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..m).collect::<Vec<_>>(), "p is a permutation, {at}");
        // A = L[P] @ U
        let mut lp: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([m, k], &device)).unwrap().into_dim();
        for i in 0..m {
            lp.i_mut((i, ..)).assign_f(l.i((p.raw()[i], ..))).unwrap();
        }
        let residual = (rt::matmul(lp.view(), u.view()) - a.view()).abs().max_f().unwrap();
        assert!(residual < 1e-12, "A = L[P] @ U, {at}");
        // L: unit diagonal, zeros above it; U: zeros below the diagonal
        for i in 0..k {
            assert_eq!(l[[i, i]], c64::new(1.0, 0.0), "unit diagonal, {at}");
            for j in (i + 1)..k {
                assert_eq!(l[[i, j]], c64::new(0.0, 0.0), "zeros above the diagonal, {at}");
            }
            for ii in (i + 1)..k {
                assert_eq!(u[[ii, i]], c64::new(0.0, 0.0), "zeros below the diagonal, {at}");
            }
        }
        // permute_l gives P L and the same u
        let permuted: LuOut = rt::linalg::lu((a.view(), true));
        assert!(permuted.p.is_none(), "{at}");
        assert!(((permuted.l.view() - lp.view()).abs().max_f().unwrap()) < 1e-14, "PL rows, {at}");
        assert!(((permuted.u.view() - u.view()).abs().max_f().unwrap()) < 1e-14, "u unchanged, {at}");
        // the packed factorization holds the same factors; its pivots imply p
        let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
        for i in 0..m {
            for j in 0..n {
                let expected = if i > j { l[[i, j]] } else { u[[i, j]] };
                assert!(((packed.lu[[i, j]] - expected).norm()) < 1e-14, "packed lu ({i}, {j}), {at}");
            }
        }
        let mut perm: Vec<usize> = (0..m).collect();
        for i in (0..k).rev() {
            perm.swap(i, packed.piv[[i]]);
        }
        assert_eq!(perm, p.raw().to_vec(), "piv implies p, {at}");
    }

    // a batch stack on both device orders, stored in the device order (in
    // order) or in the other memory order; and the in-place rule
    let (m, n) = (4usize, 3usize);
    for order in [RowMajor, ColMajor] {
        for stored in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let tag = format!("{order:?} device, {stored:?} stack");
            let data = a_vec[..2 * m * n].to_vec();
            let stack = stack_tensor(&data, &[2], &[m, n], order, stored, &device);
            let result: LuOutDyn = rt::linalg::lu(stack.view());
            for i in 0..2 {
                let (slot, l_i, u_i, p_i) = match order {
                    RowMajor => (
                        stack.i((i, .., ..)),
                        result.l.i((i, .., ..)),
                        result.u.i((i, .., ..)),
                        result.p.as_ref().unwrap().i((i, ..)),
                    ),
                    ColMajor => (
                        stack.i((.., .., i)),
                        result.l.i((.., .., i)),
                        result.u.i((.., .., i)),
                        result.p.as_ref().unwrap().i((.., i)),
                    ),
                };
                let slot = slot.into_owned().into_dim::<Ix2>();
                let single: LuOut = rt::linalg::lu(slot.view());
                assert_eq!(p_i.to_vec(), single.p.unwrap().raw().to_vec(), "p, slot {i}, {tag}");
                assert!(((l_i.to_owned() - single.l.view()).abs().max_f().unwrap()) < 1e-14, "l, slot {i}, {tag}");
                assert!(((u_i.to_owned() - single.u.view()).abs().max_f().unwrap()) < 1e-14, "u, slot {i}, {tag}");
            }
            // a mutable stack gives the same factors; it is destroyed in place
            // when stored in the device order, and left untouched otherwise
            let mut a_mut = stack_tensor(&data, &[2], &[m, n], order, stored, &device);
            let before = a_mut.raw().to_vec();
            let from_mut = rt::linalg::lu(a_mut.view_mut());
            assert!(((from_mut.l.view() - result.l.view()).abs().max_f().unwrap()) < 1e-14, "mutable l, {tag}");
            assert_eq!(a_mut.raw() != &before, stored == order, "destructive rule, {tag}");
        }
    }
}

#[test]
fn test_lu_layouts() {
    let a_vec = get_vec::<c64>('a');
    let (m, n) = (5usize, 4usize);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let a = rt::asarray((a_vec[..m * n].to_vec(), [m, n].c(), &device)).into_dim::<Ix2>();
        let reference: LuOut = rt::linalg::lu(a.view());
        let check = |result: &LuOut, at: &str| {
            assert_eq!(
                result.p.as_ref().unwrap().raw().to_vec(),
                reference.p.as_ref().unwrap().raw().to_vec(),
                "p, {at}, {order:?}"
            );
            assert!(((result.l.view() - reference.l.view()).abs().max_f().unwrap()) < 1e-13, "l, {at}, {order:?}");
            assert!(((result.u.view() - reference.u.view()).abs().max_f().unwrap()) < 1e-13, "u, {at}, {order:?}");
        };
        // the same values stored in Fortran order
        let mut a_f = a.to_contig_f(ColMajor).unwrap().into_owned().into_dim::<Ix2>();
        check(&rt::linalg::lu(a_f.view()), "f-contiguous");
        check(&rt::linalg::lu(a_f.view_mut()), "f-contiguous, mutable");
        // every second row and column of a larger buffer: no order prefers it
        let strided = (slice!(0, 2 * m, 2), slice!(0, 2 * n, 2));
        let mut buf = rt::zeros_f(([2 * m, 2 * n], &device)).unwrap().into_dim::<Ix2>();
        buf.i_mut(strided).assign_f(a.view()).unwrap();
        let a_s = buf.i(strided).into_dim::<Ix2>();
        assert!(!a_s.f_contig() && !a_s.c_contig(), "the input is fully strided, {order:?}");
        check(&rt::linalg::lu(a_s), "strided");
    }
}

#[test]
fn test_lu_overloads() {
    let device = DeviceBLAS::default();
    let (m, n) = (4usize, 3usize);
    let a_data: Vec<c64> = (0..m * n)
        .map(|j| c64::new(((j * 7 + 3) % 11) as f64 / 3.0 - 1.5, ((j * 5 + 1) % 7) as f64 / 2.0 - 1.5))
        .collect();
    let make = || rt::asarray((a_data.clone(), [m, n].c(), &device)).into_dim::<Ix2>();
    let reference: LuOut = rt::linalg::lu(&make());
    let permuted: LuOut = rt::linalg::lu((&make(), true));
    let check = |result: &LuOut, at: &str| {
        assert!(result.p.is_some(), "p must be present, {at}");
        assert_eq!(result.p.as_ref().unwrap().raw().to_vec(), reference.p.as_ref().unwrap().raw().to_vec(), "p, {at}");
        assert!(((result.l.view() - reference.l.view()).abs().max_f().unwrap()) < 1e-14, "l, {at}");
        assert!(((result.u.view() - reference.u.view()).abs().max_f().unwrap()) < 1e-14, "u, {at}");
    };
    let check_permuted = |result: &LuOut, at: &str| {
        assert!(result.p.is_none(), "p must be absent, {at}");
        assert!(((result.l.view() - permuted.l.view()).abs().max_f().unwrap()) < 1e-14, "P L, {at}");
        assert!(((result.u.view() - permuted.u.view()).abs().max_f().unwrap()) < 1e-14, "u, {at}");
    };
    // the four input kinds, bare and with permute_l
    check(&rt::linalg::lu(&make()), "&Tensor");
    check(&rt::linalg::lu((&make(), false)), "(&Tensor, false)");
    check(&rt::linalg::lu(make().view()), "TensorView");
    check(&rt::linalg::lu((make().view(), false)), "(TensorView, false)");
    check(&rt::linalg::lu(make()), "Tensor");
    check(&rt::linalg::lu((make(), false)), "(Tensor, false)");
    check(&rt::linalg::lu(make().view_mut()), "TensorMut");
    check(&rt::linalg::lu((make().view_mut(), false)), "(TensorMut, false)");
    check_permuted(&rt::linalg::lu((&make(), true)), "(&Tensor, true)");
    check_permuted(&rt::linalg::lu((make().view(), true)), "(TensorView, true)");
    check_permuted(&rt::linalg::lu((make(), true)), "(Tensor, true)");
    check_permuted(&rt::linalg::lu((make().view_mut(), true)), "(TensorMut, true)");
    check(&rt::linalg::lu_f(&make()).unwrap(), "lu_f");
    check_permuted(&rt::linalg::lu_f((&make(), true)).unwrap(), "lu_f, permute_l");
}

#[test]
fn test_lu_singular_and_zero_size() {
    // a rank-deficient matrix is factored without an error
    let device = DeviceBLAS::default();
    let a = rt::asarray((
        vec![
            c64::new(1.0, 0.0),
            c64::new(2.0, 0.0),
            c64::new(3.0, 0.0),
            c64::new(2.0, 0.0),
            c64::new(4.0, 0.0),
            c64::new(6.0, 0.0),
            c64::new(0.0, 1.0),
            c64::new(1.0, 0.0),
            c64::new(1.0, 0.0),
        ],
        [3usize, 3].c(),
        &device,
    ))
    .into_dim::<Ix2>();
    let result: LuOut = rt::linalg::lu(a.view());
    let p = result.p.as_ref().unwrap();
    let mut lp: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([3, 3], &device)).unwrap().into_dim();
    for i in 0..3 {
        lp.i_mut((i, ..)).assign_f(result.l.i((p.raw()[i], ..))).unwrap();
    }
    let residual = (rt::matmul(lp.view(), result.u.view()) - a.view()).abs().max_f().unwrap();
    assert!(residual < 1e-12, "A = L[P] @ U, singular");
    let diag: Vec<f64> = (0..3).map(|i| result.u[[i, i]].norm()).collect();
    assert!(diag.contains(&0.0), "an exactly-zero pivot on the diagonal of u, {diag:?}");

    // (0, 0), (0, N) and (M, 0) matrices: p the identity, zero l and u
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        for (m, n) in [(0usize, 0usize), (0, 4), (3, 0)] {
            let a: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([m, n], &device)).unwrap().into_dim();
            let result: LuOut = rt::linalg::lu(a.view());
            assert_eq!(result.p.as_ref().unwrap().raw().to_vec(), (0..m).collect::<Vec<_>>(), "p, {m}x{n}, {order:?}");
            assert_eq!(result.l.shape(), &[m, m.min(n)], "l, {m}x{n}, {order:?}");
            assert_eq!(result.u.shape(), &[m.min(n), n], "u, {m}x{n}, {order:?}");
            let permuted: LuOut = rt::linalg::lu((a.view(), true));
            assert!(permuted.p.is_none(), "permute_l, {m}x{n}, {order:?}");
        }
    }
}

#[test]
fn test_lu_errors() {
    let device = DeviceBLAS::default();
    // fewer than two axes
    let v = rt::asarray((vec![c64::new(1.0, 0.0), c64::new(2.0, 1.0)], [2usize].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lu_f(v.view()).err().expect("1-D a must fail"));
    assert!(msg.contains("expected at least 2 dimensions"), "{msg}");
}
