use super::*;
use rstsr::prelude::linalg::{LuFactorResult, LuResult};

// the result types follow the dims of a 2-D input
type LuFactorOut = LuFactorResult<Tensor<c64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>>;
type LuOut =
    LuResult<Option<Tensor<usize, DeviceBLAS, Ix1>>, Tensor<c64, DeviceBLAS, Ix2>, Tensor<c64, DeviceBLAS, Ix2>>;

#[test]
fn test_lu_factor() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');

    // square 1024 x 1024
    let a = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let (lu, piv): (Tensor<c64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>) =
        rt::linalg::lu_factor(a.view()).into();
    assert_eq!(lu.shape(), &[1024, 1024]);
    assert_eq!(piv.shape(), &[1024]);
    assert!((fingerprint(&lu.abs()) - -2680.7252882699167).abs() < 1e-8);
    assert!((fingerprint(&piv.map(|&v| v as f64)) - 1553.3051188216255).abs() < 1e-8);
    assert!(piv.raw().iter().enumerate().all(|(i, &v)| v >= i));

    // tall 1024 x 512
    let a = rt::asarray((&a_vec[..1024 * 512], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let (lu, piv): (Tensor<c64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>) =
        rt::linalg::lu_factor(a.view()).into();
    assert_eq!(lu.shape(), &[1024, 512]);
    assert_eq!(piv.shape(), &[512]);
    assert!((fingerprint(&lu.abs()) - -256.51823408143457).abs() < 1e-8);
    assert!((fingerprint(&piv.map(|&v| v as f64)) - -562.1343902757835).abs() < 1e-8);
}

#[test]
fn test_lu_factor_relations() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    for (m, n) in [(5usize, 5usize), (6, 4), (3, 5), (1, 1)] {
        let k = m.min(n);
        let at = format!("{m}x{n}");
        let a = rt::asarray((a_vec[..m * n].to_vec(), [m, n].c(), &device)).into_dim::<Ix2>();
        let data = a.raw().to_vec();
        let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
        let explicit: LuOut = rt::linalg::lu(a.view());
        assert_eq!(packed.lu.shape(), &[m, n], "lu shape, {at}");
        assert_eq!(packed.piv.shape(), &[k], "piv shape, {at}");
        for i in 0..m {
            for j in 0..n {
                let expected = if i > j { explicit.l[[i, j]] } else { explicit.u[[i, j]] };
                assert!(((packed.lu[[i, j]] - expected).norm()) < 1e-13, "lu ({i}, {j}), {at}");
            }
        }
        let mut perm: Vec<usize> = (0..m).collect();
        for i in (0..k).rev() {
            perm.swap(i, packed.piv[[i]]);
        }
        assert_eq!(perm, explicit.p.unwrap().raw().to_vec(), "piv implies p, {at}");
        // the read-only call leaves the input untouched
        assert_eq!(a.raw().to_vec(), data, "input untouched, {at}");

        // an owned input is consumed in place: the result is the input buffer
        let a = rt::asarray((a_vec[..m * n].to_vec(), [m, n].c(), &device)).into_dim::<Ix2>();
        let ptr = a.raw().as_ptr();
        let owned = rt::linalg::lu_factor(a);
        assert_eq!(owned.lu.raw().as_ptr(), ptr, "owned a is factored in place, {at}");
        assert!(((owned.lu.view() - packed.lu.view()).abs().max_f().unwrap()) < 1e-14, "owned lu, {at}");
        assert_eq!(owned.piv.raw().to_vec(), packed.piv.raw().to_vec(), "owned piv, {at}");
        // ... and a mutable input of the other order is written back
        let a = rt::asarray((a_vec[..m * n].to_vec(), [m, n].c(), &device)).into_dim::<Ix2>();
        let mut a_off = a.to_contig_f(ColMajor).unwrap().into_owned().into_dim::<Ix2>();
        let from_off = rt::linalg::lu_factor(a_off.view_mut());
        assert!(((from_off.lu.view() - packed.lu.view()).abs().max_f().unwrap()) < 1e-14, "off-order lu, {at}");
        assert!(((a_off.view() - packed.lu.view()).abs().max_f().unwrap()) < 1e-14, "off-order a written back, {at}");
    }

    // a singular matrix is factored silently
    let a: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([3usize, 3], &device)).unwrap().into_dim();
    let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
    assert!(packed.lu.raw().iter().all(|v| *v == c64::new(0.0, 0.0)));
    assert_eq!(packed.piv.raw().to_vec(), vec![0, 1, 2]);
}

#[test]
fn test_lu_factor_batch() {
    let a_vec = get_vec::<c64>('a');
    let (m, n) = (4usize, 3usize);
    // the stack is stored in the memory order of the device (in order) or in
    // the other one
    for order in [RowMajor, ColMajor] {
        for stored in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let tag = format!("{order:?} device, {stored:?} stack");
            let data = a_vec[..2 * m * n].to_vec();
            let a = stack_tensor(&data, &[2], &[m, n], order, stored, &device);
            let result = rt::linalg::lu_factor(a.view());
            for i in 0..2 {
                let at = format!("slot {i}, {tag}");
                let a_i = match order {
                    RowMajor => a.i((i, .., ..)),
                    ColMajor => a.i((.., .., i)),
                }
                .into_owned()
                .into_dim::<Ix2>();
                let single: LuFactorOut = rt::linalg::lu_factor(a_i.view());
                let lu_i = match order {
                    RowMajor => result.lu.i((i, .., ..)),
                    ColMajor => result.lu.i((.., .., i)),
                }
                .into_owned()
                .into_dim::<Ix2>();
                assert!(((lu_i.view() - single.lu.view()).abs().max_f().unwrap()) < 1e-13, "lu, {at}");
                let piv_i = match order {
                    RowMajor => result.piv.i((i, ..)),
                    ColMajor => result.piv.i((.., i)),
                };
                assert_eq!(piv_i.to_vec(), single.piv.raw().to_vec(), "piv, {at}");
            }

            // a mutable stack is factored in its own buffer when stored in the
            // device order, and in a work buffer written back otherwise; the
            // result is the input either way, which then holds the factors
            let mut a_mut = stack_tensor(&data, &[2], &[m, n], order, stored, &device);
            let ptr = a_mut.raw().as_ptr();
            let from_mut = rt::linalg::lu_factor(a_mut.view_mut());
            assert_eq!(from_mut.lu.raw().as_ptr(), ptr, "the result is the input stack, {tag}");
            assert!(((from_mut.lu.view() - result.lu.view()).abs().max_f().unwrap()) < 1e-13, "lu, mutable {tag}");
            drop(from_mut);
            assert!(
                ((a_mut.view() - result.lu.view()).abs().max_f().unwrap()) < 1e-13,
                "the input stack holds the factors, {tag}"
            );
        }
    }
}

#[test]
fn test_lu_factor_layouts() {
    let a_vec = get_vec::<c64>('a');
    let (m, n) = (5usize, 4usize);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let a = rt::asarray((a_vec[..m * n].to_vec(), [m, n].c(), &device)).into_dim::<Ix2>();
        let reference: LuFactorOut = rt::linalg::lu_factor(a.view());
        let check = |lu: TensorView<'_, c64, DeviceBLAS, Ix2>, piv: &Tensor<usize, DeviceBLAS, Ix1>, at: &str| {
            assert_eq!(piv.raw().to_vec(), reference.piv.raw().to_vec(), "piv, {at}, {order:?}");
            assert!(((lu - reference.lu.view()).abs().max_f().unwrap()) < 1e-13, "lu, {at}, {order:?}");
        };
        // the same values stored in Fortran order
        let mut a_f = a.to_contig_f(ColMajor).unwrap().into_owned().into_dim::<Ix2>();
        let from_f: LuFactorOut = rt::linalg::lu_factor(a_f.view());
        check(from_f.lu.view(), &from_f.piv, "f-contiguous");
        let from_f = rt::linalg::lu_factor(a_f.view_mut());
        check(from_f.lu.view(), &from_f.piv, "f-contiguous, mutable");
        // every second row and column of a larger buffer: no order prefers it
        let strided = (slice!(0, 2 * m, 2), slice!(0, 2 * n, 2));
        let mut buf = rt::zeros_f(([2 * m, 2 * n], &device)).unwrap().into_dim::<Ix2>();
        buf.i_mut(strided).assign_f(a.view()).unwrap();
        let a_s = buf.i(strided).into_dim::<Ix2>();
        assert!(!a_s.f_contig() && !a_s.c_contig(), "the input is fully strided, {order:?}");
        let from_s: LuFactorOut = rt::linalg::lu_factor(a_s);
        check(from_s.lu.view(), &from_s.piv, "strided");
    }
}

#[test]
fn test_lu_factor_zero_size() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        for (m, n) in [(0usize, 0usize), (0, 4), (3, 0)] {
            let a: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([m, n], &device)).unwrap().into_dim();
            let result: LuFactorOut = rt::linalg::lu_factor(a.view());
            assert_eq!(result.lu.shape(), &[m, n], "lu, {m}x{n}, {order:?}");
            assert_eq!(result.piv.shape(), &[m.min(n)], "piv, {m}x{n}, {order:?}");
        }
    }
}

#[test]
fn test_lu_factor_errors() {
    let device = DeviceBLAS::default();
    // fewer than two axes
    let v = rt::asarray((vec![c64::new(1.0, 0.0), c64::new(2.0, 1.0)], [2usize].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lu_factor_f(v.view()).err().expect("1-D a must fail"));
    assert!(msg.contains("expected at least 2 dimensions"), "{msg}");
}
