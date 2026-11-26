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
            // the explicit factor form agrees
            let x_f = rt::linalg::lu_solve((&factors, &b, trans));
            assert!(((x_f.view() - x.view()).abs().max_f().unwrap()) < 1e-13, "factor form, {tag}");
            let x_f = rt::linalg::lu_solve((&factors, &b.view(), trans));
            let _ = x_f;
        }
        // no flag: A x = b; and the general solve agrees
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
        let x_n = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b, NoTrans));
        assert!(((x.view() - x_n.view()).abs().max_f().unwrap()) < 1e-14);
        let x_g = rt::linalg::solve_general((&a, &b));
        assert!(((x.view() - x_g.view()).abs().max_f().unwrap()) < 1e-12, "solve_general parity, {n}x{n}");
    }

    // a batch stack on both device orders: parity with the 2-D calls, and the
    // in-place rule
    let (n, k) = (4usize, 2usize);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let stacked = |batch: &[usize], inner: &[usize]| -> Vec<usize> {
            match order {
                RowMajor => [batch, inner].concat(),
                ColMajor => [inner, batch].concat(),
            }
        };
        let a_data = a_vec[..3 * n * n].to_vec();
        let b_data = b_vec[..3 * n * k].to_vec();
        let a = rt::asarray((a_data.clone(), stacked(&[3], &[n, n]).c(), &device)).into_owned().into_dyn();
        let b = rt::asarray((b_data.clone(), stacked(&[3], &[n, k]).c(), &device)).into_owned().into_dyn();
        let packed = rt::linalg::lu_factor(a.view());
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, &b));
        assert_eq!(x.shape(), b.shape(), "{order:?}");
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
            assert!(((x_i.view() - reference.view()).abs().max_f().unwrap()) < 1e-12, "slot {i}, {order:?}");
        }
        // mutable b: solved in place
        let mut b_mut = rt::asarray((b_data, stacked(&[3], &[n, k]).c(), &device)).into_owned().into_dyn();
        let ptr = b_mut.raw().as_ptr();
        let x = rt::linalg::lu_solve((&packed.lu, &packed.piv, b_mut.view_mut()));
        assert_eq!(x.raw().as_ptr(), ptr, "in place, {order:?}");
        assert!(
            ((x.view() - rt::linalg::lu_solve((&packed.lu, &packed.piv, &b)).view()).abs().max_f().unwrap()) < 1e-14
        );
    }
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
