use super::*;
use rstsr::prelude::linalg::{LuFactorResult, LuResult};

// the result types follow the dims of a 2-D input
type LuOut =
    LuResult<Option<Tensor<usize, DeviceBLAS, Ix1>>, Tensor<c64, DeviceBLAS, Ix2>, Tensor<c64, DeviceBLAS, Ix2>>;
type LuFactorOut = LuFactorResult<Tensor<c64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>>;

#[test]
fn test_lu_solve() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    let a = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
    let (lu, piv) = (&packed.lu, &packed.piv);

    // K = 1
    let b = rt::asarray((&b_vec[..1024], [1024, 1].c(), &device)).into_dim::<Ix2>();
    let x = rt::linalg::lu_solve((lu, piv, &b));
    assert_eq!(x.shape(), &[1024, 1]);
    assert!((fingerprint(&x.abs()) - 22.15319665971149).abs() < 1e-8);

    // conjugate-transposed system Aᴴ x = b
    let x = rt::linalg::lu_solve((lu, piv, &b, FlagTrans::C));
    assert_eq!(x.shape(), &[1024, 1]);
    assert!((fingerprint(&x.abs()) - 1.9240075469181859).abs() < 1e-8);
}

#[test]
fn test_lu_solve_relations() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    for (n, k) in [(1usize, 1usize), (3, 1), (5, 2), (6, 4)] {
        let a = rt::asarray((a_vec[..n * n].to_vec(), [n, n].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec[..n * k].to_vec(), [n, k].c(), &device)).into_dim::<Ix2>();
        let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
        let factors: LuOut = rt::linalg::lu(a.view());
        for trans in [NoTrans, Trans, ConjTrans] {
            let tag = format!("{n}x{n}, K = {k}, {trans:?}");
            let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b, trans));
            assert_eq!(x.shape(), &[n, k], "shape, {tag}");
            // op(A) x = b
            let op_a = match trans {
                NoTrans => a.clone(),
                Trans => a.t().into_owned(),
                ConjTrans => a.clone().conj().t().into_owned(),
                FlagTrans::CN => unreachable!(),
            };
            let residual = (rt::matmul(op_a.view(), x.view()) - b.view()).abs().max_f().unwrap();
            assert!(residual < 1e-12, "residual {residual}, {tag}");
            // the explicit factor form agrees, for a reference to b and a view of b
            let x_f = rt::linalg::lu_solve((&factors, &b, trans));
            assert!(((x_f.view() - x.view()).abs().max_f().unwrap()) < 1e-13, "factor form, {tag}");
            let x_f = rt::linalg::lu_solve((&factors, &b.view(), trans));
            assert!(((x_f.view() - x.view()).abs().max_f().unwrap()) < 1e-13, "factor form, TensorView of b, {tag}");
        }
        // no flag: A x = b; and the general solve agrees
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
        let x_n = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b, NoTrans));
        assert!(((x.view() - x_n.view()).abs().max_f().unwrap()) < 1e-14);
        let x_g = rt::linalg::solve_general((&a, &b));
        assert!(((x.view() - x_g.view()).abs().max_f().unwrap()) < 1e-12, "solve_general parity, {n}x{n}");
    }

    // a batch stack on both device orders, stored in the device order (in
    // order) or in the other memory order: parity with the 2-D calls, and the
    // in-place rule
    let (n, k) = (4usize, 2usize);
    for order in [RowMajor, ColMajor] {
        for stored in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let tag = format!("{order:?} device, {stored:?} stack");
            let mk = |v: &[c64], batch: &[usize], inner: &[usize]| -> Tensor<c64, DeviceBLAS, IxD> {
                stack_tensor(v, batch, inner, order, stored, &device)
            };
            let a_data = a_vec[..3 * n * n].to_vec();
            let b_data = b_vec[..3 * n * k].to_vec();
            let a = mk(&a_data, &[3], &[n, n]);
            let b = mk(&b_data, &[3], &[n, k]);
            let packed = rt::linalg::lu_factor(a.view());
            let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
            assert_eq!(x.shape(), b.shape(), "{tag}");
            for i in 0..3 {
                let pick = |t: &Tensor<c64, DeviceBLAS, IxD>| -> Tensor<c64, DeviceBLAS, Ix2> {
                    let v = match order {
                        RowMajor => t.i((i, ..)),
                        ColMajor => t.i((.., .., i)),
                    };
                    v.into_owned_keep_layout().into_dim::<Ix2>()
                };
                let reference = rt::linalg::solve_general((&pick(&a), &pick(&b)));
                let x_i = pick(&x);
                assert!(((x_i.view() - reference.view()).abs().max_f().unwrap()) < 1e-12, "slot {i}, {tag}");
            }
            // a mutable b stack is solved in its own buffer when stored in the
            // device order, and in a work buffer written back otherwise; the
            // result is b either way
            let mut b_mut = mk(&b_data, &[3], &[n, k]);
            let ptr = b_mut.raw().as_ptr();
            let x_m = rt::linalg::lu_solve((&packed.lu, &packed.piv, b_mut.view_mut()));
            assert_eq!(x_m.raw().as_ptr(), ptr, "in place, {tag}");
            assert!(((x_m.view() - x.view()).abs().max_f().unwrap()) < 1e-14, "mutable b stack, {tag}");
            drop(x_m);
            assert!(
                ((b_mut.view() - x.view()).abs().max_f().unwrap()) < 1e-14,
                "the b stack holds the solution, {tag}"
            );
        }
    }
}

#[test]
fn test_lu_solve_layouts() {
    let a_vec = get_vec::<c64>('a');
    let b_vec = get_vec::<c64>('b');
    let (n, k) = (5usize, 2usize);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let a = rt::asarray((a_vec[..n * n].to_vec(), [n, n].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_vec[..n * k].to_vec(), [n, k].c(), &device)).into_dim::<Ix2>();
        let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
        let reference = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
        let check = |x: Tensor<c64, DeviceBLAS, IxD>, at: &str| {
            assert!(((x.view() - reference.view()).abs().max_f().unwrap()) < 1e-12, "{at}, {order:?}");
        };
        // factors of the other memory order
        let a_f = a.to_contig_f(ColMajor).unwrap().into_owned().into_dim::<Ix2>();
        let packed_f: LuFactorOut = rt::linalg::lu_factor(a_f.view());
        check(rt::linalg::lu_solve((packed_f.lu.view(), &packed_f.piv, &b)), "f-contiguous factors");
        // right-hand side of the other memory order
        let b_f = b.to_contig_f(ColMajor).unwrap().into_owned().into_dim::<Ix2>();
        check(rt::linalg::lu_solve((&packed.lu, &packed.piv, &b_f)), "f-contiguous b");
        // fully non-contiguous factors (every second row and column of a larger
        // buffer) and pivots (every second entry)
        let strided = (slice!(0, 2 * n, 2), slice!(0, 2 * n, 2));
        let mut fbuf = rt::zeros_f(([2 * n, 2 * n], &device)).unwrap().into_dim::<Ix2>();
        fbuf.i_mut(strided).assign_f(packed.lu.view()).unwrap();
        let lu_s = fbuf.i(strided).into_dim::<Ix2>();
        assert!(lu_s.stride().iter().all(|&s| s != 1), "the factors are fully strided, {order:?}");
        let mut piv_data = vec![usize::MAX; 2 * n];
        for (i, &p) in packed.piv.raw().iter().enumerate() {
            piv_data[2 * i] = p;
        }
        let piv_buf = rt::asarray((piv_data, [2 * n].c(), &device)).into_dim::<Ix1>();
        let piv_s = piv_buf.i(slice!(0, 2 * n, 2)).into_dim::<Ix1>();
        assert_eq!(piv_s.stride()[0], 2, "the pivots are strided, {order:?}");
        check(rt::linalg::lu_solve((lu_s, piv_s, &b)), "strided factors and pivots");
    }
}

#[test]
fn test_lu_solve_overloads() {
    let device = DeviceBLAS::default();
    let (n, k) = (4usize, 2usize);
    let a_data: Vec<c64> = (0..n * n)
        .map(|j| c64::new(((j * 7 + 3) % 11) as f64 / 3.0 - 1.5, ((j * 5 + 1) % 7) as f64 / 2.0 - 1.5))
        .collect();
    let b_data: Vec<c64> = (0..n * k)
        .map(|j| c64::new(((j * 5 + 1) % 7) as f64 / 2.0 - 1.5, ((j * 3 + 2) % 5) as f64 / 2.0 - 1.0))
        .collect();
    let make_a = || rt::asarray((a_data.clone(), [n, n].c(), &device)).into_dim::<Ix2>();
    let make_b = || rt::asarray((b_data.clone(), [n, k].c(), &device)).into_dim::<Ix2>();
    let packed: LuFactorOut = rt::linalg::lu_factor(&make_a());
    let factors: LuOut = rt::linalg::lu(make_a());
    for trans in [NoTrans, Trans, ConjTrans] {
        let reference = rt::linalg::lu_solve((&packed.lu, &packed.piv, &make_b(), trans));
        let check = |x: TensorView<'_, c64, DeviceBLAS, IxD>, at: &str| {
            assert!(((x - reference.view()).abs().max_f().unwrap()) < 1e-13, "{at}, {trans:?}");
        };
        // the packed form, flat
        check(
            rt::linalg::lu_solve((&packed.lu, &packed.piv, make_b().view(), trans)).view().to_dyn(),
            "(TensorView of b, flag)",
        );
        check(
            rt::linalg::lu_solve((packed.lu.view(), packed.piv.view(), make_b(), trans)).view().to_dyn(),
            "(owned b, flag)",
        );
        // SciPy's nested form
        check(
            rt::linalg::lu_solve(((&packed.lu, &packed.piv), make_b().view(), trans)).view().to_dyn(),
            "((lu, piv), view, flag)",
        );
        // the factor form of lu
        check(rt::linalg::lu_solve((&factors, &make_b(), trans)).view(), "(&LuResult, &Tensor, flag)");
        // the result of lu_factor, by value and by reference
        check(
            rt::linalg::lu_solve((rt::linalg::lu_factor(make_a()), &make_b(), trans)).view().to_dyn(),
            "(LuFactorResult, &Tensor, flag)",
        );
        check(rt::linalg::lu_solve((&packed, make_b().view(), trans)).view().to_dyn(), "(&LuFactorResult, view, flag)");
    }
    // consumed b is solved in place; the fallible variants
    let reference = rt::linalg::lu_solve((&packed.lu, &packed.piv, &make_b()));
    let mut b_m = make_b();
    let ptr = b_m.raw().as_ptr();
    let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, b_m.view_mut()));
    assert_eq!(x.raw().as_ptr(), ptr, "mutable b is solved in place");
    assert!(((x.view().to_dyn() - reference.view()).abs().max_f().unwrap()) < 1e-13);
    // fallible variants
    let x = rt::linalg::lu_solve_f((&packed.lu, &packed.piv, &make_b())).unwrap();
    assert_eq!(x.shape(), &[n, k]);
    assert!(((x.view().to_dyn() - reference.view()).abs().max_f().unwrap()) < 1e-13);
    let x = rt::linalg::lu_solve_f((&factors, &make_b(), ConjTrans)).unwrap();
    assert_eq!(x.shape(), &[n, k]);
}

#[test]
fn test_lu_solve_errors() {
    let device = DeviceBLAS::default();
    let a_data: Vec<c64> = (0..16).map(|j| c64::new((j % 7) as f64 + 1.0, (j % 5) as f64)).collect();
    let b_data: Vec<c64> = (0..12).map(|j| c64::new((j % 5) as f64, (j % 3) as f64)).collect();
    let a = rt::asarray((a_data.clone(), [4usize, 4].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((b_data.clone(), [4usize, 3].c(), &device)).into_dim::<Ix2>();
    let packed = rt::linalg::lu_factor(a.view());

    // a permuted factorization (lu with permute_l) has no p to solve from
    let permuted: LuOut = rt::linalg::lu((a.view(), true));
    let msg = format!("{}", rt::linalg::lu_solve_f((&permuted, &b)).expect_err("permuted factors must fail"));
    assert!(msg.contains("permuted"), "{msg}");
    // the factorization must be square ...
    let a_rect = rt::asarray((a_data[..12].to_vec(), [4usize, 3].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::lu_solve_f((&a_rect, &packed.piv, &b)).expect_err("non-square must fail"));
    assert!(msg.contains("must be square"), "{msg}");
    // ... piv must hold exactly N entries ...
    let piv_short = packed.piv.i(..3).into_owned().into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lu_solve_f((&packed.lu, &piv_short, &b)).expect_err("short piv must fail"));
    assert!(msg.contains("piv must have"), "{msg}");
    // ... and b's rows must match
    let b_short = rt::asarray((b_data[..2].to_vec(), [2usize].c(), &device)).into_dim::<Ix1>();
    let msg =
        format!("{}", rt::linalg::lu_solve_f((&packed.lu, &packed.piv, &b_short)).expect_err("row mismatch must fail"));
    assert!(msg.contains("row dimension must match"), "{msg}");
}

#[test]
fn test_lu_solve_singular_and_zero_size() {
    let device = DeviceBLAS::default();
    // a singular system gives inf / nan, not an error
    let a = rt::asarray((
        vec![c64::new(1.0, 0.0), c64::new(2.0, 0.0), c64::new(2.0, 0.0), c64::new(4.0, 0.0)],
        [2usize, 2].c(),
        &device,
    ))
    .into_dim::<Ix2>();
    let packed = rt::linalg::lu_factor(a.view());
    let b: Tensor<c64, DeviceBLAS, Ix2> = rt::ones_f(([2usize, 1], &device)).unwrap().into_dim();
    let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
    assert!(x.raw().iter().any(|v| !v.re.is_finite() || !v.im.is_finite()));

    // N == 0 and K == 0
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let a: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim();
        let packed = rt::linalg::lu_factor(a.view());
        let b: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 3], &device)).unwrap().into_dim();
        assert_eq!(rt::linalg::lu_solve((&packed.lu, &packed.piv, &b)).shape(), &[0, 3], "N == 0, {order:?}");
        let a = rt::asarray((
            vec![c64::new(2.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 1.0), c64::new(3.0, 0.0)],
            [2usize, 2].c(),
            &device,
        ))
        .into_dim::<Ix2>();
        let packed = rt::linalg::lu_factor(a.view());
        let b: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([2usize, 0], &device)).unwrap().into_dim();
        assert_eq!(rt::linalg::lu_solve((&packed.lu, &packed.piv, &b)).shape(), &[2, 0], "K == 0, {order:?}");
    }
}
