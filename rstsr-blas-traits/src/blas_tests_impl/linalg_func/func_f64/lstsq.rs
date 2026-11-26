use super::*;
use rstsr::prelude::linalg::LstsqResult;

type LstsqOut = LstsqResult<
    Tensor<f64, DeviceBLAS, IxD>,
    Tensor<f64, DeviceBLAS, IxD>,
    Tensor<usize, DeviceBLAS, IxD>,
    Tensor<f64, DeviceBLAS, IxD>,
>;

#[test]
fn test_lstsq() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
    let b_vec = get_vec::<f64>('b')[..1024 * 3].to_vec();

    // tall, vector b: the residual is a scalar
    let a = rt::asarray((&a_vec[..], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((&b_vec[..1024], [1024].c(), &device)).into_dim::<Ix1>();
    let (x, residuals, rank, s) = rt::linalg::lstsq((a.view(), b.view())).into();
    assert_eq!(x.shape(), &[512]);
    assert!(residuals.shape().is_empty());
    assert!(rank.shape().is_empty());
    assert_eq!(s.shape(), &[512]);
    assert!((fingerprint(&x) - 0.6714195293796388).abs() < 1e-8);
    assert!((fingerprint(&s) - 33.96933907104208).abs() < 1e-8);
    assert!((fingerprint(&residuals) - 495.06381572902126).abs() < 1e-8);
    assert_eq!(rank.raw()[0], 512);

    // tall, matrix b (K = 3)
    let b = rt::asarray((&b_vec[..], [1024, 3].c(), &device)).into_dim::<Ix2>();
    let (x, residuals, rank, s) = rt::linalg::lstsq((a.view(), b.view())).into();
    assert_eq!(x.shape(), &[512, 3]);
    assert_eq!(residuals.shape(), &[3]);
    assert_eq!(s.shape(), &[512]);
    assert!((fingerprint(&x) - -2.417996890572109).abs() < 1e-8);
    assert!((fingerprint(&s) - 33.96933907104208).abs() < 1e-8);
    assert!((fingerprint(&residuals) - 544.9834618414125).abs() < 1e-8);
    assert_eq!(rank.raw()[0], 512);

    // wide, vector b: minimum-norm solution, empty residuals
    let a_vec = get_vec::<f64>('a')[..512 * 1024].to_vec();
    let a = rt::asarray((&a_vec[..], [512, 1024].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((&b_vec[..512], [512].c(), &device)).into_dim::<Ix1>();
    let (x, residuals, rank, s) = rt::linalg::lstsq((a.view(), b.view())).into();
    assert_eq!(x.shape(), &[1024]);
    assert_eq!(residuals.shape(), &[0]);
    assert_eq!(s.shape(), &[512]);
    assert!((fingerprint(&x) - 0.1316952775202293).abs() < 1e-8);
    assert!((fingerprint(&s) - 32.27742168207754).abs() < 1e-8);
    assert_eq!(rank.raw()[0], 512);

    // rank-deficient: rank < n, the residual is NaN
    let mut a_data = get_vec::<f64>('a')[..1024 * 512].to_vec();
    for i in 0..1024 {
        a_data[i * 512 + 2] = a_data[i * 512] + a_data[i * 512 + 1];
    }
    let a = rt::asarray((&a_data[..], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let b_vec = get_vec::<f64>('b');
    let b = rt::asarray((&b_vec[..1024], [1024].c(), &device)).into_dim::<Ix1>();
    let (x, residuals, rank, s) = rt::linalg::lstsq((a.view(), b.view())).into();
    assert_eq!(x.shape(), &[512]);
    assert!(residuals.shape().is_empty());
    assert_eq!(rank.raw()[0], 511);
    assert!((fingerprint(&x) - 0.6203201140848792).abs() < 1e-8);
    assert!((fingerprint(&s) - 46.463897960241695).abs() < 1e-8);
    assert!(residuals.raw()[0].is_nan());

    // explicit rcond: a value >= 1 is clamped to eps by LAPACK
    let a = rt::asarray((&a_data[..], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let b_vec = get_vec::<f64>('b');
    let b = rt::asarray((&b_vec[..1024], [1024].c(), &device)).into_dim::<Ix1>();
    let (x, _residuals, rank, _s) = rt::linalg::lstsq((a.view(), b.view(), 1.0)).into();
    assert_eq!(rank.raw()[0], 511);
    assert!(x.raw().iter().all(|x| x.is_finite()));

    // matrix b with a rank-deficient a: every residual column is NaN
    let a = rt::asarray((&a_data[..], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((&b_vec[..1024 * 3], [1024, 3].c(), &device)).into_dim::<Ix2>();
    let (x, residuals, rank, s) = rt::linalg::lstsq((a.view(), b.view())).into();
    assert_eq!(x.shape(), &[512, 3]);
    assert_eq!(residuals.shape(), &[3]);
    assert_eq!(rank.raw()[0], 511);
    assert!((fingerprint(&x) - -2.475933070459636).abs() < 1e-8);
    assert!((fingerprint(&s) - 46.463897960241695).abs() < 1e-8);
    assert!(residuals.raw().iter().all(|x| x.is_nan()));
}

#[test]
fn test_lstsq_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    let (m, n, k) = (6, 4, 2);
    let a_data: Vec<f64> = (0..3 * m * n).map(|j| ((j * 7 + 3) % 11) as f64 - 5.0).collect();
    let b_data: Vec<f64> = (0..3 * m * k).map(|j| ((j * 5 + 1) % 7) as f64 - 3.0).collect();
    let a = rt::asarray((&a_data[..], [3, m, n].c(), &device)).into_dim::<Ix3>();
    let b = rt::asarray((&b_data[..], [3, m, k].c(), &device)).into_dim::<Ix3>();

    let result = rt::linalg::lstsq((a.view(), b.view()));
    assert_eq!(result.x.shape(), &[3, n, k]);
    assert_eq!(result.s.shape(), &[3, n]);
    assert_eq!(result.rank.shape(), &[3]);
    assert_eq!(result.residuals.shape(), &[3, k]);
    for i in 0..3 {
        let a_i = a.i((i, .., ..)).into_dim::<Ix2>();
        let b_i = b.i((i, .., ..)).into_dim::<Ix2>();
        let reference = rt::linalg::lstsq((a_i.view(), b_i.view()));
        assert!((fingerprint(&result.x.i((i, .., ..))) - fingerprint(&reference.x)).abs() < 1e-10);
        assert!((fingerprint(&result.s.i((i, ..))) - fingerprint(&reference.s)).abs() < 1e-10);
        assert_eq!(result.rank.raw()[i], reference.rank.raw()[0]);
        assert!((fingerprint(&result.residuals.i((i, ..))) - fingerprint(&reference.residuals)).abs() < 1e-10);
    }

    // batched vector b: residuals are batch-shaped
    let b_vec_data: Vec<f64> = (0..3 * m).map(|j| ((j * 5 + 1) % 7) as f64 - 3.0).collect();
    let b_vec = rt::asarray((&b_vec_data[..], [3, m].c(), &device)).into_dim::<Ix2>();
    let result = rt::linalg::lstsq((a.view(), b_vec.view()));
    assert_eq!(result.x.shape(), &[3, n]);
    assert_eq!(result.residuals.shape(), &[3]);
    for i in 0..3 {
        let a_i = a.i((i, .., ..)).into_dim::<Ix2>();
        let b_i = b_vec.i((i, ..)).into_dim::<Ix1>();
        let reference = rt::linalg::lstsq((a_i.view(), b_i.view()));
        assert!((fingerprint(&result.x.i((i, ..))) - fingerprint(&reference.x)).abs() < 1e-10);
        assert_eq!(result.rank.raw()[i], reference.rank.raw()[0]);
        assert!((result.residuals.raw()[i] - reference.residuals.raw()[0]).abs() < 1e-10);
    }

    // owned inputs give the same result; view inputs are left untouched
    let a_owned = rt::asarray((&a_data[..], [3, m, n].c(), &device)).into_dim::<Ix3>();
    let b_owned = rt::asarray((&b_data[..], [3, m, k].c(), &device)).into_dim::<Ix3>();
    let owned = rt::linalg::lstsq((a_owned, b_owned));
    assert_eq!(owned.x.shape(), &[3, n, k]);
    let reference = rt::linalg::lstsq((a.view(), b.view()));
    assert!(rt::allclose(owned.x.view(), reference.x.view(), Some(1e-12)));
    assert!(rt::allclose(owned.s.view(), reference.s.view(), Some(1e-12)));
    assert_eq!(a.raw().to_vec(), a_data);
    assert_eq!(b.raw().to_vec(), b_data);
    assert_eq!(b_vec.raw().to_vec(), b_vec_data);
}

#[test]
fn test_lstsq_order() {
    // the same logical input (strided F layout) on a row-major and a
    // column-major device must give identical results
    let (m, n, k) = (6, 4, 2);
    let a_data: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 / 3.0 - 1.5).collect();
    let b_data: Vec<f64> = (0..m * k).map(|j| ((j * 5 + 1) % 7) as f64 / 2.0 - 1.5).collect();

    let mut rm_device = DeviceBLAS::default();
    rm_device.set_default_order(RowMajor);
    let mut cm_device = DeviceBLAS::default();
    cm_device.set_default_order(ColMajor);

    let a_rm = rt::asarray((&a_data[..], [m, n].f(), &rm_device)).into_dim::<Ix2>();
    let b_rm = rt::asarray((&b_data[..], [m, k].f(), &rm_device)).into_dim::<Ix2>();
    let rm = rt::linalg::lstsq((a_rm.view(), b_rm.view()));

    let a_cm = rt::asarray((&a_data[..], [m, n].f(), &cm_device)).into_dim::<Ix2>();
    let b_cm = rt::asarray((&b_data[..], [m, k].f(), &cm_device)).into_dim::<Ix2>();
    let cm = rt::linalg::lstsq((a_cm.view(), b_cm.view()));

    assert_eq!(cm.x.shape(), &[n, k]);
    assert_eq!(cm.s.shape(), &[n]);
    assert_eq!(cm.rank.raw()[0], rm.rank.raw()[0]);
    // the two devices are distinct instances, so compare values, not tensors
    for i in 0..n {
        for j in 0..k {
            assert!((cm.x[[i, j]] - rm.x[[i, j]]).abs() < 1e-12);
        }
        assert!((cm.s[[i]] - rm.s[[i]]).abs() < 1e-12);
    }
    for j in 0..k {
        assert!((cm.residuals[[j]] - rm.residuals[[j]]).abs() < 1e-12);
    }

    // a genuinely column-major buffer gives the same values
    let a_cm_f = rt::asarray((&a_data[..], [m, n].f(), &cm_device))
        .into_contig_f(ColMajor)
        .unwrap()
        .into_owned()
        .into_dim::<Ix2>();
    let cm_f = rt::linalg::lstsq((a_cm_f.view(), b_cm.view()));
    for i in 0..n {
        for j in 0..k {
            assert!((cm_f.x[[i, j]] - rm.x[[i, j]]).abs() < 1e-12);
        }
    }
}

#[test]
fn test_lstsq_overloads() {
    let device = DeviceBLAS::default();
    let (m, n, k) = (6, 4, 2);
    let a_data: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 / 3.0 - 1.5).collect();
    let b_data: Vec<f64> = (0..m * k).map(|j| ((j * 5 + 1) % 7) as f64 / 2.0 - 1.5).collect();
    let make = || {
        let a = rt::asarray((a_data.clone(), [m, n].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((b_data.clone(), [m, k].c(), &device)).into_dim::<Ix2>();
        (a, b)
    };
    let check = |result: &LstsqOut, reference: &LstsqOut| {
        assert!(rt::allclose(result.x.view(), reference.x.view(), Some(1e-12)));
        assert!(rt::allclose(&result.s, &reference.s, Some(1e-12)));
        assert!(rt::allclose(result.residuals.view(), reference.residuals.view(), Some(1e-12)));
        assert_eq!(result.rank.raw()[0], reference.rank.raw()[0]);
    };

    let (a, b) = make();
    let reference = rt::linalg::lstsq((a.view(), b.view()));

    // every input-kind pair of {&, view, view_mut, owned} x {&, view, view_mut, owned}
    let (a, b) = make();
    check(&rt::linalg::lstsq((&a, &b)), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((&a, b.view())), &reference);
    let (a, mut b) = make();
    check(&rt::linalg::lstsq((&a, b.view_mut())), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((&a, b)), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((a.view(), &b)), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((a.view(), b.view())), &reference);
    let (a, mut b) = make();
    check(&rt::linalg::lstsq((a.view(), b.view_mut())), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((a.view(), b)), &reference);
    let (mut a, b) = make();
    check(&rt::linalg::lstsq((a.view_mut(), &b)), &reference);
    let (mut a, b) = make();
    check(&rt::linalg::lstsq((a.view_mut(), b.view())), &reference);
    let (mut a, mut b) = make();
    check(&rt::linalg::lstsq((a.view_mut(), b.view_mut())), &reference);
    let (mut a, b) = make();
    check(&rt::linalg::lstsq((a.view_mut(), b)), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((a, &b)), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((a, b.view())), &reference);
    let (a, mut b) = make();
    check(&rt::linalg::lstsq((a, b.view_mut())), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((a, b)), &reference);

    // the three-argument form with mixed input kinds, and the fallible variants
    let (a, b) = make();
    check(&rt::linalg::lstsq((&a, &b, 1e-12)), &reference);
    check(&rt::linalg::lstsq_f((&a, &b)).unwrap(), &reference);
    let (mut a, b) = make();
    check(&rt::linalg::lstsq((a.view_mut(), &b, 1e-12)), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq_f((&a, b.view(), 1e-12)).unwrap(), &reference);
    let (a, mut b) = make();
    check(&rt::linalg::lstsq_f((a, b.view_mut(), 1e-12)).unwrap(), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((a, b, 1e-12)), &reference);

    // b is only read; a mutable a is consumed by the factorization
    let (mut a, mut b) = make();
    rt::linalg::lstsq((a.view_mut(), b.view_mut()));
    assert_eq!(b.raw().to_vec(), b_data);
}

#[test]
fn test_lstsq_inplace() {
    // an owned or mutable a whose batch needs no broadcast is consumed by the
    // factorization; every input route gives the result of the read-only call
    let (m, n, k) = (6usize, 4usize, 2usize);
    let a_data: Vec<f64> = (0..m * n).map(|j| ((j * 7 + 3) % 11) as f64 / 3.0 - 1.5).collect();
    let b_data: Vec<f64> = (0..m * k).map(|j| ((j * 5 + 1) % 7) as f64 / 2.0 - 1.5).collect();

    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let mk = |v: &[f64], rows: usize, cols: usize, layout: FlagOrder| -> Tensor<f64, DeviceBLAS, Ix2> {
            match layout {
                RowMajor => rt::asarray((v.to_vec(), [rows, cols].c(), &device)).into_dim::<Ix2>(),
                ColMajor => rt::asarray((v.to_vec(), [rows, cols].f(), &device)).into_dim::<Ix2>(),
            }
        };
        // the same logical values of a row-major buffer, stored in Fortran order
        let fortran = |v: &[f64], rows: usize, cols: usize| -> Vec<f64> {
            let mut w = vec![0.0; v.len()];
            for i in 0..rows {
                for j in 0..cols {
                    w[j * rows + i] = v[i * cols + j];
                }
            }
            w
        };
        let a = mk(&a_data, m, n, order);
        let b = mk(&b_data, m, k, order);
        let reference = rt::linalg::lstsq((a.view(), b.view()));
        let check = |result: &LstsqOut, at: &str| {
            assert!(rt::allclose(result.x.view(), reference.x.view(), Some(1e-10)), "x, {at}, {order:?}");
            assert!(rt::allclose(&result.s, &reference.s, Some(1e-10)), "s, {at}, {order:?}");
            assert!(
                rt::allclose(result.residuals.view(), reference.residuals.view(), Some(1e-10)),
                "residuals, {at}, {order:?}"
            );
            assert_eq!(result.rank.raw()[0], reference.rank.raw()[0], "rank, {at}, {order:?}");
        };

        // contiguous in the device order: consumed in place
        let mut a_mut = mk(&a_data, m, n, order);
        check(&rt::linalg::lstsq((a_mut.view_mut(), b.view())), "mutable a");
        assert_eq!(b.raw().to_vec(), b_data);

        // owned a
        let a_owned = mk(&a_data, m, n, order);
        check(&rt::linalg::lstsq((a_owned, b.view())), "owned a");

        // layout of the other order: factored from a work buffer, a is left untouched
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        let mut a_off = a.to_contig_f(other).unwrap().into_owned().into_dim::<Ix2>();
        let a_off_data = a_off.raw().to_vec();
        check(&rt::linalg::lstsq((a_off.view_mut(), b.view())), "off-order a");
        assert_eq!(a_off.raw().to_vec(), a_off_data, "off-order a, {order:?}");

        // padded rows in the device order: consumed in place with ld > n
        let mut buf = mk(&vec![0.0; (m + 4) * (n + 4)], m + 4, n + 4, order);
        buf.i_mut((0..m, 0..n)).assign_f(mk(&a_data, m, n, order).view()).unwrap();
        let mut a_padded = buf.i((0..m, 0..n)).into_owned_keep_layout().into_dim::<Ix2>();
        check(&rt::linalg::lstsq((a_padded.view_mut(), b.view())), "padded a");

        if order == RowMajor {
            // the destroyed a put back by the row-major bridge is the LAPACK
            // factorization in the caller's layout: it equals what a direct
            // column-major call leaves in an f-contiguous a
            let mut cm_device = DeviceBLAS::default();
            cm_device.set_default_order(ColMajor);
            // the same logical matrices, stored in Fortran order
            let mut a_cm = rt::asarray((fortran(&a_data, m, n), [m, n].f(), &cm_device)).into_dim::<Ix2>();
            let b_cm = rt::asarray((fortran(&b_data, m, k), [m, k].f(), &cm_device)).into_dim::<Ix2>();
            rt::linalg::lstsq((a_cm.view_mut(), b_cm.view()));
            // distinct device instances: compare values, not tensors
            for i in 0..m {
                for j in 0..n {
                    assert!((a_padded[[i, j]] - a_cm[[i, j]]).abs() < 1e-10, "destroyed a ({i}, {j})");
                }
            }
        }

        // a batch of a smaller than the plan is broadcast: a cannot be consumed,
        // and every slice still gets the full solution
        let (mut a_b, b_b): (Tensor<f64, DeviceBLAS, Ix3>, Tensor<f64, DeviceBLAS, Ix3>) = match order {
            RowMajor => (
                rt::asarray((a_data.clone(), [1usize, m, n].c(), &device)).into_dim::<Ix3>(),
                rt::asarray((b_data.repeat(2), [2usize, m, k].c(), &device)).into_dim::<Ix3>(),
            ),
            ColMajor => (
                rt::asarray((a_data.clone(), [m, n, 1usize].f(), &device)).into_dim::<Ix3>(),
                rt::asarray((b_data.repeat(2), [m, k, 2usize].f(), &device)).into_dim::<Ix3>(),
            ),
        };
        let result = rt::linalg::lstsq((a_b.view_mut(), b_b.view()));
        for i in 0..2 {
            let (x_i, s_i, res_i) = match order {
                RowMajor => (result.x.i((i, .., ..)), result.s.i((i, ..)), result.residuals.i((i, ..))),
                ColMajor => (result.x.i((.., .., i)), result.s.i((.., i)), result.residuals.i((.., i))),
            };
            let at = format!("broadcast slice {i}");
            assert!(rt::allclose(x_i, reference.x.view(), Some(1e-10)), "x, {at}, {order:?}");
            assert!(rt::allclose(s_i, reference.s.view(), Some(1e-10)), "s, {at}, {order:?}");
            assert!(rt::allclose(res_i, reference.residuals.view(), Some(1e-10)), "residuals, {at}, {order:?}");
            assert_eq!(result.rank[[i]], reference.rank.raw()[0], "rank, {at}, {order:?}");
        }
        assert_eq!(a_b.raw().to_vec(), a_data, "broadcast a, {order:?}");
    }
}

#[test]
fn test_lstsq_errors() {
    let device = DeviceBLAS::default();
    let a_data: Vec<f64> = (0..24).map(|j| ((j * 7 + 3) % 11) as f64).collect();
    let b_data: Vec<f64> = (0..12).map(|j| ((j * 5 + 1) % 7) as f64).collect();
    let a = rt::asarray((&a_data[..], [6, 4].c(), &device)).into_dim::<Ix2>();

    // a must be at least 2-D
    let a1 = rt::asarray((&a_data[..6], [6].c(), &device)).into_dim::<Ix1>();
    let b1 = rt::asarray((&b_data[..6], [6].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lstsq_f((a1.view(), b1.view())).err().expect("1-D a must fail"));
    assert!(msg.contains("a must be at least 2-D"), "{msg}");

    // b's rows must match a's rows
    let b_short = rt::asarray((&b_data[..5], [5].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lstsq_f((a.view(), b_short.view())).err().expect("row mismatch must fail"));
    assert!(msg.contains("must match a"), "{msg}");

    // NaN rcond is rejected
    let b = rt::asarray((&b_data[..], [6, 2].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::lstsq_f((a.view(), b.view(), f64::NAN)).err().expect("NaN rcond must fail"));
    assert!(msg.contains("rcond must not be NaN"), "{msg}");

    // b's rank is rank(a) or one less: a b that is too short is an error, not a panic
    let a4 = rt::asarray(((0..36).map(|j| (j % 7) as f64).collect::<Vec<_>>(), [2usize, 2, 3, 3].c(), &device))
        .into_dim::<Ix4>();
    let b_short = rt::asarray((vec![1.0f64, 2.0, 3.0], [3usize].c(), &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::lstsq_f((a4.view(), b_short.view())).err().expect("short b must fail"));
    assert!(msg.contains("b must be (..., M, K) or (..., M) for a's rank"), "{msg}");

    let b3 = rt::asarray(((0..72).map(|j| (j % 7) as f64).collect::<Vec<_>>(), [2usize, 6, 2].c(), &device))
        .into_dim::<Ix3>();
    let msg = format!("{}", rt::linalg::lstsq_f((a.view(), b3.view())).err().expect("3-D b for 2-D a must fail"));
    assert!(msg.contains("b must be (..., M, K) or (..., M) for a's rank"), "{msg}");

    // the batch shapes of a and b must broadcast
    let a3 = rt::asarray(((0..72).map(|j| (j % 7) as f64).collect::<Vec<_>>(), [3usize, 6, 4].c(), &device))
        .into_dim::<Ix3>();
    let b_bad_batch = rt::asarray(((0..24).map(|j| (j % 7) as f64).collect::<Vec<_>>(), [2usize, 6, 2].c(), &device))
        .into_dim::<Ix3>();
    let msg =
        format!("{}", rt::linalg::lstsq_f((a3.view(), b_bad_batch.view())).err().expect("batch mismatch must fail"));
    assert!(msg.contains("roadcast"), "{msg}");
}

#[test]
fn test_lstsq_rcond() {
    let device = DeviceBLAS::default();
    // a diagonal matrix with graded singular values
    let scales = [1.0, 1e-1, 1e-2, 1e-3, 1e-4];
    let (m, n) = (8usize, 5usize);
    let a_data: Vec<f64> = (0..m * n).map(|j| if j % n == j / n { scales[j % n] } else { 0.0 }).collect();
    let b_data: Vec<f64> = (0..m * 2).map(|j| (((j * 7 + 3) % 11) as f64 - 5.0) / 3.0).collect();
    let a = rt::asarray((&a_data[..], [m, n].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((&b_data[..], [m, 2].c(), &device)).into_dim::<Ix2>();

    // LAPACK clamps rcond <= 0 and rcond >= 1 to machine epsilon
    let default = rt::linalg::lstsq((a.view(), b.view()));
    assert_eq!(default.rank.raw()[0], 5);
    for rcond in [1e-6, 3e-2, 1e-1, 0.0, -1.0, 1.0, 2.0, f64::EPSILON * m as f64] {
        let result = rt::linalg::lstsq((a.view(), b.view(), rcond));
        let clamped = if rcond <= 0.0 || rcond >= 1.0 { f64::EPSILON } else { rcond };
        let expected = scales.iter().filter(|s| **s > clamped * scales[0]).count();
        assert_eq!(result.rank.raw()[0], expected, "rcond {rcond}");
        // the residual is NaN on every column of a rank-deficient system
        let all_nan = result.residuals.raw().iter().all(|v| v.is_nan());
        assert_eq!(all_nan, expected < n, "rcond {rcond}");
        assert_eq!(result.residuals.shape(), &[2]);
    }
    let clamped = rt::linalg::lstsq((a.view(), b.view(), -1.0));
    assert!(rt::allclose(clamped.x.view(), default.x.view(), Some(1e-12)));
}

#[test]
fn test_lstsq_zero_size() {
    // zero-size operands never reach the driver; the outputs are synthesized
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);

        // m == 0: zero solution block, no singular values, empty residuals
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 5], &device)).unwrap().into_dim::<Ix2>();
        let b: Tensor<f64, DeviceBLAS, Ix1> = rt::zeros_f(([0usize], &device)).unwrap().into_dim::<Ix1>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), &[5]);
        assert_eq!(result.s.shape(), &[0]);
        assert!(result.rank.shape().is_empty());
        assert_eq!(result.rank.raw()[0], 0);
        assert_eq!(result.residuals.shape(), &[0]);
        assert!(result.x.raw().iter().all(|v| *v == 0.0));

        let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 3], &device)).unwrap().into_dim::<Ix2>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), &[5, 3]);
        assert_eq!(result.residuals.shape(), &[0]);
        assert!(result.x.raw().iter().all(|v| *v == 0.0));

        // n == 0: the residuals of the right-hand sides are synthesized
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([5usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let b: Tensor<f64, DeviceBLAS, Ix1> =
            rt::asarray((vec![1.0, 2.0, 3.0, 4.0, 5.0], [5].c(), &device)).into_dim::<Ix1>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), &[0]);
        assert_eq!(result.s.shape(), &[0]);
        assert!(result.residuals.shape().is_empty());
        assert_eq!(result.rank.raw()[0], 0);
        assert_eq!(result.residuals.raw()[0], 55.0);

        let b: Tensor<f64, DeviceBLAS, Ix2> =
            rt::asarray(((1..=10).map(|v| v as f64).collect::<Vec<_>>(), [5usize, 2].c(), &device)).into_dim::<Ix2>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), &[0, 2]);
        assert_eq!(result.residuals.shape(), &[2]);
        // column sums of squares of the C-ordered 1..10 right-hand side
        assert_eq!(result.residuals.raw(), &[165.0, 220.0]);

        // n == 0 with k == 0: no residual column to synthesize
        let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([5usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), &[0, 0]);
        assert_eq!(result.residuals.shape(), &[0]);

        // m == 0 and n == 0: empty residuals (the NumPy rule)
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 3], &device)).unwrap().into_dim::<Ix2>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), &[0, 3]);
        assert_eq!(result.s.shape(), &[0]);
        assert_eq!(result.rank.raw()[0], 0);
        assert_eq!(result.residuals.shape(), &[0]);
        let b: Tensor<f64, DeviceBLAS, Ix1> = rt::zeros_f(([0usize], &device)).unwrap().into_dim::<Ix1>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), &[0]);
        assert_eq!(result.residuals.shape(), &[0]);

        // k == 0: one zero column is padded for the driver, then trimmed
        // (a Vandermonde matrix of six distinct rows has full column rank)
        let a_data: Vec<f64> = (0..6).flat_map(|i| (0..4).map(move |j| (i as f64 + 1.0).powi(j))).collect();
        let a: Tensor<f64, DeviceBLAS, Ix2> = rt::asarray((a_data, [6usize, 4].c(), &device)).into_dim::<Ix2>();
        let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([6usize, 0], &device)).unwrap().into_dim::<Ix2>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), &[4, 0]);
        assert_eq!(result.s.shape(), &[4]);
        assert_eq!(result.residuals.shape(), &[0]);
        assert_eq!(result.rank.raw()[0], 4);

        // batched shapes: (batch, M, N) under RowMajor, (M, N, batch) under ColMajor
        let a_shape = |batch: &[usize], rows: usize, cols: usize| -> Vec<usize> {
            match order {
                RowMajor => [batch, &[rows, cols]].concat(),
                ColMajor => [&[rows, cols], batch].concat(),
            }
        };
        let out_shape = |batch: &[usize], inner: &[usize]| -> Vec<usize> {
            match order {
                RowMajor => [batch, inner].concat(),
                ColMajor => [inner, batch].concat(),
            }
        };

        // empty batch
        let a: Tensor<f64, DeviceBLAS, Ix3> = rt::zeros_f((a_shape(&[0], 4, 5), &device)).unwrap().into_dim::<Ix3>();
        let b: Tensor<f64, DeviceBLAS, Ix3> = rt::zeros_f((a_shape(&[0], 4, 2), &device)).unwrap().into_dim::<Ix3>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), out_shape(&[0], &[5, 2]).as_slice());
        assert_eq!(result.s.shape(), out_shape(&[0], &[4]).as_slice());
        assert_eq!(result.rank.shape(), &[0]);
        assert_eq!(result.residuals.shape(), out_shape(&[0], &[0]).as_slice());

        // a batch of empty-row systems
        let a: Tensor<f64, DeviceBLAS, Ix3> = rt::zeros_f((a_shape(&[3], 0, 5), &device)).unwrap().into_dim::<Ix3>();
        let b: Tensor<f64, DeviceBLAS, Ix3> = rt::zeros_f((a_shape(&[3], 0, 2), &device)).unwrap().into_dim::<Ix3>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), out_shape(&[3], &[5, 2]).as_slice());
        assert_eq!(result.rank.shape(), &[3]);
        assert!(result.rank.raw().iter().all(|r| *r == 0));
        assert_eq!(result.residuals.shape(), out_shape(&[3], &[0]).as_slice());

        // a batch of empty-column systems: one synthesized residual per slice
        let a: Tensor<f64, DeviceBLAS, Ix3> = rt::zeros_f((a_shape(&[3], 5, 0), &device)).unwrap().into_dim::<Ix3>();
        let b: Tensor<f64, DeviceBLAS, Ix3> = rt::ones_f((a_shape(&[3], 5, 1), &device)).unwrap().into_dim::<Ix3>();
        let result = rt::linalg::lstsq((a.view(), b.view()));
        assert_eq!(result.x.shape(), out_shape(&[3], &[0, 1]).as_slice());
        assert_eq!(result.residuals.shape(), out_shape(&[3], &[1]).as_slice());
        assert!(result.residuals.raw().iter().all(|v| *v == 5.0));
    }
}

#[test]
fn test_lstsq_batch_ranks() {
    // the slice (1, 0) of a (2, 2) batch is rank deficient
    fn entry(i: usize, p: usize, j0: usize, j1: usize) -> f64 {
        let v = (((i * 7 + p * 3 + j0 + j1) % 11) as f64 - 5.0) / 3.0;
        if (j0, j1) == (1, 0) && p == 1 {
            entry(i, 0, j0, j1) + entry(i, 2, j0, j1)
        } else {
            v
        }
    }
    let rhs = |j0: usize, j1: usize, i: usize| (((i * 5 + j0 + 2 * j1) % 7) as f64 - 3.0) / 2.0;
    let (m, n) = (6usize, 4usize);

    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);

        let a_data: Vec<f64> = match order {
            // logical index (j0, j1, i, p)
            RowMajor => (0..4 * m * n).map(|f| entry((f / n) % m, f % n, f / (n * m) / 2, f / (n * m) % 2)).collect(),
            // logical index (i, p, j0, j1)
            ColMajor => (0..m * n * 4).map(|f| entry(f / (n * 4), (f / 4) % n, (f / 2) % 2, f % 2)).collect(),
        };
        let b_data: Vec<f64> = match order {
            // logical index (j0, j1, i)
            RowMajor => (0..4 * m).map(|f| rhs(f / (m * 2), f / m % 2, f % m)).collect(),
            // logical index (i, j0, j1)
            ColMajor => (0..m * 4).map(|f| rhs((f / 2) % 2, f % 2, f / 4)).collect(),
        };
        let (a, b): (Tensor<f64, DeviceBLAS, Ix4>, Tensor<f64, DeviceBLAS, Ix3>) = match order {
            RowMajor => (
                rt::asarray((a_data, [2usize, 2, m, n].c(), &device)).into_dim::<Ix4>(),
                rt::asarray((b_data, [2usize, 2, m].c(), &device)).into_dim::<Ix3>(),
            ),
            ColMajor => (
                rt::asarray((a_data, [m, n, 2usize, 2].c(), &device)).into_dim::<Ix4>(),
                rt::asarray((b_data, [m, 2usize, 2].c(), &device)).into_dim::<Ix3>(),
            ),
        };
        let result = rt::linalg::lstsq((a.view(), b.view()));
        let stack = |inner: &[usize]| match order {
            RowMajor => [&[2, 2][..], inner].concat(),
            ColMajor => [inner, &[2, 2][..]].concat(),
        };
        assert_eq!(result.x.shape(), stack(&[n]).as_slice());
        assert_eq!(result.s.shape(), stack(&[n]).as_slice());
        assert_eq!(result.rank.shape(), &[2, 2]);
        assert_eq!(result.residuals.shape(), &[2, 2]);
        for j0 in 0..2 {
            for j1 in 0..2 {
                let (a_i, b_i, x_i, s_i) = match order {
                    RowMajor => (
                        a.i((j0, j1, .., ..)).into_dim::<Ix2>(),
                        b.i((j0, j1, ..)).into_dim::<Ix1>(),
                        result.x.i((j0, j1, ..)),
                        result.s.i((j0, j1, ..)),
                    ),
                    ColMajor => (
                        a.i((.., .., j0, j1)).into_dim::<Ix2>(),
                        b.i((.., j0, j1)).into_dim::<Ix1>(),
                        result.x.i((.., j0, j1)),
                        result.s.i((.., j0, j1)),
                    ),
                };
                let reference = rt::linalg::lstsq((a_i.view(), b_i.view()));
                let at = format!("{order:?} slice ({j0}, {j1})");
                assert_eq!(result.rank[[j0, j1]], reference.rank.raw()[0], "rank, {at}");
                assert!(rt::allclose(x_i, reference.x.view(), Some(1e-10)), "x, {at}");
                assert!(rt::allclose(s_i, reference.s.view(), Some(1e-10)), "s, {at}");
                let res = result.residuals[[j0, j1]];
                let res_ref = reference.residuals.raw()[0];
                assert!((res.is_nan() && res_ref.is_nan()) || (res - res_ref).abs() < 1e-10, "residual, {at}");
            }
        }
        assert_eq!(result.rank[[1, 0]], n - 1, "{order:?}");

        // the batch of b broadcasts against the batch of a: batch (2, 3) of a,
        // batch (1, 3) of b (b carries a batch axis of size 1)
        let a_data: Vec<f64> = (0..6 * m * n).map(|f| (((f * 7 + 3) % 11) as f64 - 5.0) / 3.0).collect();
        let b_data: Vec<f64> = match order {
            // logical index (j, i, kk) of b (1, 3, M, K)
            RowMajor => (0..3 * m * 2).map(|f| rhs(f / (m * 2), 0, (f / 2) % m) + (f % 2) as f64).collect(),
            // logical index (i, kk, j) of b (M, K, 1, 3)
            ColMajor => (0..m * 2 * 3).map(|f| rhs(f % 3, 0, f / 6) + ((f / 3) % 2) as f64).collect(),
        };
        let (a, b): (Tensor<f64, DeviceBLAS, Ix4>, Tensor<f64, DeviceBLAS, Ix4>) = match order {
            RowMajor => (
                rt::asarray((a_data, [2usize, 3, m, n].c(), &device)).into_dim::<Ix4>(),
                rt::asarray((b_data, [1usize, 3, m, 2].c(), &device)).into_dim::<Ix4>(),
            ),
            ColMajor => (
                rt::asarray((a_data, [m, n, 2usize, 3].c(), &device)).into_dim::<Ix4>(),
                rt::asarray((b_data, [m, 2usize, 1, 3].c(), &device)).into_dim::<Ix4>(),
            ),
        };
        let result = rt::linalg::lstsq((a.view(), b.view()));
        for i0 in 0..2 {
            for j in 0..3 {
                let (a_i, b_i, x_i, s_i) = match order {
                    RowMajor => (
                        a.i((i0, j, .., ..)).into_dim::<Ix2>(),
                        b.i((0, j, .., ..)).into_dim::<Ix2>(),
                        result.x.i((i0, j, .., ..)),
                        result.s.i((i0, j, ..)),
                    ),
                    ColMajor => (
                        a.i((.., .., i0, j)).into_dim::<Ix2>(),
                        b.i((.., .., 0, j)).into_dim::<Ix2>(),
                        result.x.i((.., .., i0, j)),
                        result.s.i((.., i0, j)),
                    ),
                };
                let reference = rt::linalg::lstsq((a_i.view(), b_i.view()));
                let at = format!("{order:?} slice ({i0}, {j})");
                let res_i = match order {
                    RowMajor => result.residuals.i((i0, j, ..)),
                    ColMajor => result.residuals.i((.., i0, j)),
                };
                assert!(rt::allclose(x_i, reference.x.view(), Some(1e-10)), "x, {at}");
                assert!(rt::allclose(s_i, reference.s.view(), Some(1e-10)), "s, {at}");
                assert!(rt::allclose(res_i, reference.residuals.view(), Some(1e-10)), "residual, {at}");
            }
        }
        let stack = |inner: &[usize]| match order {
            RowMajor => [&[2, 3][..], inner].concat(),
            ColMajor => [inner, &[2, 3][..]].concat(),
        };
        assert_eq!(result.x.shape(), stack(&[n, 2]).as_slice());
        assert_eq!(result.s.shape(), stack(&[n]).as_slice());
        assert_eq!(result.rank.shape(), &[2, 3]);
        assert_eq!(result.residuals.shape(), stack(&[2]).as_slice());
    }
}
