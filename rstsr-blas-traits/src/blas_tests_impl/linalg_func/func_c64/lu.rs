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

    // a batch stack on both device orders, and the in-place rule
    let (m, n) = (4usize, 3usize);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let shape: Vec<usize> = match order {
            RowMajor => vec![2, m, n],
            ColMajor => vec![m, n, 2],
        };
        let data = a_vec[..2 * m * n].to_vec();
        let stack = rt::asarray((data.clone(), shape.c(), &device)).into_owned().into_dyn();
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
            assert_eq!(p_i.to_vec(), single.p.unwrap().raw().to_vec(), "p, slot {i}, {order:?}");
            assert!(((l_i.to_owned() - single.l.view()).abs().max_f().unwrap()) < 1e-14, "l, slot {i}, {order:?}");
            assert!(((u_i.to_owned() - single.u.view()).abs().max_f().unwrap()) < 1e-14, "u, slot {i}, {order:?}");
        }
        // a mutable in-order input is consumed (destroyed), an owned one too
        let mut a_mut = rt::asarray((data.clone(), shape.c(), &device)).into_owned().into_dyn();
        let from_mut = rt::linalg::lu(a_mut.view_mut());
        assert!(((from_mut.l.view() - result.l.view()).abs().max_f().unwrap()) < 1e-14, "mutable l, {order:?}");
        // an in-order input is destroyed; an off-order one is left untouched
        let destroyed = a_mut.raw().to_vec() != data;
        assert_eq!(destroyed, order == RowMajor, "destructive rule, {order:?}");
    }
}
