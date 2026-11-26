use super::*;
use rstsr::prelude::linalg::LuFactorResult;

// the result types follow the dims of a 2-D input
type LuFactorOut = LuFactorResult<Tensor<f64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>>;
type LuFactorOutMut<'a> = LuFactorResult<TensorMut<'a, f64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>>;

#[test]
fn test_lu_factor() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');

    // square 1024 x 1024
    let a = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let (lu, piv): (Tensor<f64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>) =
        rt::linalg::lu_factor(a.view()).into();
    assert_eq!(lu.shape(), &[1024, 1024]);
    assert_eq!(piv.shape(), &[1024]);
    assert!((fingerprint(&lu) - 5397.198541468439).abs() < 1e-8);
    assert!((fingerprint(&piv.map(|&v| v as f64)) - -14.694714160748731).abs() < 1e-8);
    // 0-based pivots, LAPACK's `ipiv[i] >= i`
    assert!(piv.raw().iter().enumerate().all(|(i, &v)| v >= i));

    // tall 1024 x 512: piv has min(M, N) entries
    let a = rt::asarray((&a_vec[..1024 * 512], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let (lu, piv): (Tensor<f64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>) =
        rt::linalg::lu_factor(a.view()).into();
    assert_eq!(lu.shape(), &[1024, 512]);
    assert_eq!(piv.shape(), &[512]);
    assert!((fingerprint(&lu) - -731.4085927481875).abs() < 1e-8);
    assert!((fingerprint(&piv.map(|&v| v as f64)) - -1903.892619915529).abs() < 1e-8);
}

/// Check one packed factorization against the explicit factors of `lu`: the
/// multipliers of `L` strictly below the diagonal, `U` on and above it, and
/// pivots whose reverse swaps give `p`.
fn check_lu_factor(
    lu: &TensorView<'_, f64, DeviceBLAS, Ix2>,
    piv: &Tensor<usize, DeviceBLAS, Ix1>,
    a: &Tensor<f64, DeviceBLAS, Ix2>,
    at: &str,
) {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let k = m.min(n);
    let explicit = rt::linalg::lu(a.view());
    assert_eq!(lu.shape(), &[m, n], "lu shape, {at}");
    assert_eq!(piv.shape(), &[k], "piv shape, {at}");
    for i in 0..m {
        for j in 0..n {
            let expected = if i > j { explicit.l[[i, j]] } else { explicit.u[[i, j]] };
            assert!((lu[[i, j]] - expected).abs() < 1e-13, "lu ({i}, {j}), {at}");
        }
    }
    let mut perm: Vec<usize> = (0..m).collect();
    for i in (0..k).rev() {
        perm.swap(i, piv[[i]]);
    }
    assert_eq!(perm, explicit.p.unwrap().raw().to_vec(), "piv implies p, {at}");
}

#[test]
fn test_lu_factor_relations() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    for (m, n) in [(5usize, 5usize), (6, 4), (3, 5), (1, 1), (4, 1), (1, 4)] {
        let a = rt::asarray((a_vec[..m * n].to_vec(), [m, n].c(), &device)).into_dim::<Ix2>();
        let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
        check_lu_factor(&packed.lu.view(), &packed.piv, &a, &format!("{m}x{n}"));
        // the input is not modified by a read-only call
        assert_eq!(a.raw().to_vec(), a_vec[..m * n].to_vec());
    }

    // singular matrices are factored silently
    let a =
        rt::asarray((vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0], [3usize, 3].c(), &device)).into_dim::<Ix2>();
    let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
    assert!(packed.lu.raw().iter().all(|v| v.is_finite()));
    check_lu_factor(&packed.lu.view(), &packed.piv, &a, "singular");
    let a = rt::zeros_f(([3usize, 3], &device)).unwrap().into_dim::<Ix2>();
    let packed: LuFactorOut = rt::linalg::lu_factor(a.view());
    assert!(packed.lu.raw().iter().all(|v| *v == 0.0));
    assert_eq!(packed.piv.raw().to_vec(), vec![0, 1, 2]);
}

#[test]
fn test_lu_factor_batch() {
    let a_vec = get_vec::<f64>('a');
    let (m, n) = (5usize, 4usize);
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let shape: Vec<usize> = match order {
            RowMajor => vec![2, 3, m, n],
            ColMajor => vec![m, n, 2, 3],
        };
        let a = rt::asarray((a_vec[..2 * 3 * m * n].to_vec(), shape.c(), &device)).into_owned().into_dyn();
        let result: LuFactorResult<Tensor<f64, DeviceBLAS, IxD>, Tensor<usize, DeviceBLAS, IxD>> =
            rt::linalg::lu_factor(a.view());
        assert_eq!(result.lu.shape(), a.shape(), "lu shape, {order:?}");
        let piv_shape: Vec<usize> = match order {
            RowMajor => vec![2, 3, m.min(n)],
            ColMajor => vec![m.min(n), 2, 3],
        };
        assert_eq!(result.piv.shape(), &piv_shape[..], "piv shape, {order:?}");
        for i in 0..2 {
            for j in 0..3 {
                let at = format!("slot ({i}, {j}), {order:?}");
                let (a_ij, lu_ij, piv_ij) = match order {
                    RowMajor => (a.i((i, j, .., ..)), result.lu.i((i, j, .., ..)), result.piv.i((i, j, ..))),
                    ColMajor => (a.i((.., .., i, j)), result.lu.i((.., .., i, j)), result.piv.i((.., i, j))),
                };
                let a_ij = a_ij.into_owned().into_dim::<Ix2>();
                let single: LuFactorOut = rt::linalg::lu_factor(a_ij.view());
                check_lu_factor(&single.lu.view(), &single.piv, &a_ij, &at);
                assert!(rt::allclose(lu_ij, single.lu.view(), Some(1e-14)), "lu, {at}");
                assert_eq!(piv_ij.to_vec(), single.piv.raw().to_vec(), "piv, {at}");
            }
        }

        // owned and reference inputs give the same factors as the view call
        let make =
            || rt::asarray((a_vec[..2 * 3 * m * n].to_vec(), shape.clone().c(), &device)).into_owned().into_dyn();
        for (at, candidate) in [("owned", rt::linalg::lu_factor(make())), ("reference", rt::linalg::lu_factor(&make()))]
        {
            assert!(rt::allclose(candidate.lu.view(), result.lu.view(), Some(1e-14)), "lu, {at}, {order:?}");
            assert_eq!(candidate.piv.raw().to_vec(), result.piv.raw().to_vec(), "piv, {at}, {order:?}");
        }
    }
}

#[test]
fn test_lu_factor_layouts() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let (m, n) = (7usize, 5usize);
    let a = rt::asarray((a_vec[..m * n].to_vec(), [m, n].c(), &device)).into_dim::<Ix2>();
    let reference: LuFactorOut = rt::linalg::lu_factor(a.view());

    // the same values stored in Fortran order (the other order on this device)
    let mut fortran = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            fortran[j * m + i] = a_vec[i * n + j];
        }
    }
    let a_f = rt::asarray((fortran, [m, n].f(), &device)).into_dim::<Ix2>();
    let from_f: LuFactorOut = rt::linalg::lu_factor(a_f.view());
    assert_eq!(from_f.piv.raw().to_vec(), reference.piv.raw().to_vec());
    assert!(rt::allclose(from_f.lu.view(), reference.lu.view(), Some(1e-14)), "f-contiguous");

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
    let from_strided: LuFactorOut = rt::linalg::lu_factor(a_strided.view());
    assert_eq!(from_strided.piv.raw().to_vec(), reference.piv.raw().to_vec());
    assert!(rt::allclose(from_strided.lu.view(), reference.lu.view(), Some(1e-14)), "strided");
}

#[test]
fn test_lu_factor_inplace() {
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
        let reference: LuFactorOut = rt::linalg::lu_factor(a.view());
        let check = |lu: &TensorView<'_, f64, DeviceBLAS, Ix2>, piv: &Tensor<usize, DeviceBLAS, Ix1>, at: &str| {
            check_lu_factor(lu, piv, &a, &format!("{at}, {order:?}"));
            assert!(rt::allclose(lu.view(), reference.lu.view(), Some(1e-13)), "lu, {at}, {order:?}");
            assert_eq!(piv.raw().to_vec(), reference.piv.raw().to_vec(), "piv, {at}, {order:?}");
        };

        // owned input: consumed, the result is the consumed buffer
        let owned = rt::linalg::lu_factor(mk(&a_data, m, n, order));
        check(&owned.lu.view(), &owned.piv, "owned");

        // mutable in-order input: factored in place, the result aliases the input
        let mut a_mut = mk(&a_data, m, n, order);
        let ptr = a_mut.raw().as_ptr();
        let from_mut: LuFactorOutMut = rt::linalg::lu_factor(a_mut.view_mut());
        assert_eq!(from_mut.lu.raw().as_ptr(), ptr, "the result is the input buffer, {order:?}");
        check(&from_mut.lu.view(), &from_mut.piv, "mutable");
        // the input buffer now holds the packed factors
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (from_mut.lu[[i, j]] - reference.lu[[i, j]]).abs() < 1e-13,
                    "in-place lu ({i}, {j}), {order:?}"
                );
            }
        }

        // a mutable input of the other order is factored from a work buffer
        // that is written back into the input
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        let mut a_off = a.to_contig_f(other).unwrap().into_owned().into_dim::<Ix2>();
        let from_off: LuFactorOutMut = rt::linalg::lu_factor(a_off.view_mut());
        check(&from_off.lu.view(), &from_off.piv, "off-order");
        let written_back = from_off.lu.to_owned();
        for i in 0..m {
            for j in 0..n {
                assert!((a_off[[i, j]] - reference.lu[[i, j]]).abs() < 1e-13, "written-back a ({i}, {j}), {order:?}");
                assert!((written_back[[i, j]] - reference.lu[[i, j]]).abs() < 1e-13, "result ({i}, {j}), {order:?}");
            }
        }

        // padded in-order input: factored in place with ld > n, padding untouched
        let (pm, pn) = (m + 4, n + 4);
        let mut buf = mk(&vec![f64::NAN; pm * pn], pm, pn, order);
        buf.i_mut((0..m, 0..n)).assign_f(a.view()).unwrap();
        let mut a_padded = buf.i((0..m, 0..n)).into_owned_keep_layout().into_dim::<Ix2>();
        let from_padded: LuFactorOutMut = rt::linalg::lu_factor(a_padded.view_mut());
        check(&from_padded.lu.view(), &from_padded.piv, "padded");
        for i in 0..pm {
            for j in 0..pn {
                if i >= m || j >= n {
                    assert!(buf[[i, j]].is_nan(), "padding ({i}, {j}), {order:?}");
                }
            }
        }
    }
}

#[test]
fn test_lu_factor_overloads() {
    let device = DeviceBLAS::default();
    let (m, n) = (4usize, 4usize);
    let a_data: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 / 3.0 - 1.5).collect();
    let make = || rt::asarray((a_data.clone(), [m, n].c(), &device)).into_dim::<Ix2>();
    let reference: LuFactorOut = rt::linalg::lu_factor(&make());
    let check = |lu: &TensorView<'_, f64, DeviceBLAS, Ix2>, piv: &Tensor<usize, DeviceBLAS, Ix1>| {
        assert!(rt::allclose(lu, reference.lu.view(), Some(1e-14)));
        assert_eq!(piv.raw().to_vec(), reference.piv.raw().to_vec());
    };

    // every input kind
    let r = rt::linalg::lu_factor(&make());
    check(&r.lu.view(), &r.piv);
    let r = rt::linalg::lu_factor(make().view());
    check(&r.lu.view(), &r.piv);
    let r = rt::linalg::lu_factor(make());
    check(&r.lu.view(), &r.piv);
    let mut a = make();
    let r = rt::linalg::lu_factor(a.view_mut());
    check(&r.lu.view(), &r.piv);
    let r = rt::linalg::lu_factor_f(&make()).unwrap();
    check(&r.lu.view(), &r.piv);

    // tuple conversion
    let (lu, piv): (Tensor<f64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>) = reference.into();
    assert_eq!(lu.shape(), &[m, n]);
    assert_eq!(piv.shape(), &[m]);
    let result = LuFactorResult::from((lu, piv));
    assert_eq!(result.piv.shape(), &[m]);

    // an n-d input gives dynamic-dim results
    let stack = rt::asarray((a_data.repeat(2), [2, m, n].c(), &device)).into_owned().into_dyn();
    let r = rt::linalg::lu_factor(stack.view());
    assert_eq!(r.lu.shape(), &[2, m, n]);
    assert_eq!(r.piv.shape(), &[2, m]);
}

#[test]
fn test_lu_factor_zero_size() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        for (m, n) in [(0usize, 0usize), (0, 4), (3, 0)] {
            let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([m, n], &device)).unwrap().into_dim::<Ix2>();
            let result: LuFactorOut = rt::linalg::lu_factor(a.view());
            assert_eq!(result.lu.shape(), &[m, n], "{m}x{n}, {order:?}");
            assert_eq!(result.piv.shape(), &[m.min(n)], "{m}x{n}, {order:?}");
            assert!(result.lu.raw().iter().all(|v| *v == 0.0));
        }
        // an empty batch
        let shape = match order {
            RowMajor => vec![0usize, 3, 3],
            ColMajor => vec![3usize, 3, 0],
        };
        let a: Tensor<f64, DeviceBLAS, IxD> = rt::zeros_f((shape, &device)).unwrap();
        let result = rt::linalg::lu_factor(a.view());
        assert_eq!(result.lu.shape(), a.shape());
        assert_eq!(result.piv.shape().len(), a.shape().len() - 1);
    }
}

#[test]
fn test_lu_factor_errors() {
    let device = DeviceBLAS::default();
    let v = rt::asarray((vec![1.0, 2.0, 3.0], [3usize].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lu_factor_f(v.view()).err().expect("1-D a must fail"));
    assert!(msg.contains("expected at least 2 dimensions"), "{msg}");
}
