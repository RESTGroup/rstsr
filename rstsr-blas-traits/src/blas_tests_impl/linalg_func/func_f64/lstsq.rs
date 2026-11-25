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

    // owned inputs are consumed in place, views are copied
    let a_owned = rt::asarray((&a_data[..], [3, m, n].c(), &device)).into_dim::<Ix3>();
    let b_owned = rt::asarray((&b_data[..], [3, m, k].c(), &device)).into_dim::<Ix3>();
    let owned = rt::linalg::lstsq((a_owned, b_owned));
    assert_eq!(owned.x.shape(), &[3, n, k]);
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

    // a genuinely column-major buffer is consumed in place
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

    let (a, b) = make();
    check(&rt::linalg::lstsq((&a, &b)), &reference);
    check(&rt::linalg::lstsq((&a, &b, 1e-12)), &reference);
    check(&rt::linalg::lstsq_f((&a, &b)).unwrap(), &reference);
    let (mut a, mut b) = make();
    check(&rt::linalg::lstsq((a.view_mut(), b.view_mut())), &reference);
    let (a, b) = make();
    check(&rt::linalg::lstsq((a, b)), &reference);
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
}
