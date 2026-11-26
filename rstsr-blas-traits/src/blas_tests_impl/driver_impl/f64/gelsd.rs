use super::*;
use rstsr_blas_traits::lapack_lstsq::*;
use rstsr_blas_traits::lapack_svd::DGESDD;

#[test]
fn test_dgelsd() {
    let device = DeviceBLAS::default();
    let a_vec = &get_vec::<f64>('a')[..1024 * 512];
    let b_vec = &get_vec::<f64>('b')[..1024 * 3];
    let rcond = f64::EPSILON * 1024.0;

    // tall, K = 1 (m > n: residuals defined)
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((&b_vec[..1024], [1024, 1].c(), &device)).into_dim::<Ix2>();
    let (x, s, rank, residuals) =
        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
    assert!((fingerprint(&x) - 0.6714195293796388).abs() < 1e-8);
    assert!((fingerprint(&s) - 33.96933907104208).abs() < 1e-8);
    assert_eq!(rank, 512);
    assert!((fingerprint(&residuals.unwrap()) - 495.06381572902126).abs() < 1e-8);

    // tall, K = 3
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((b_vec, [1024, 3].c(), &device)).into_dim::<Ix2>();
    let (x, s, rank, residuals) =
        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
    assert!((fingerprint(&x) - -2.417996890572109).abs() < 1e-8);
    assert!((fingerprint(&s) - 33.96933907104208).abs() < 1e-8);
    assert_eq!(rank, 512);
    assert!((fingerprint(&residuals.unwrap()) - 544.9834618414125).abs() < 1e-8);

    // wide, K = 1: minimum-norm solution, empty residuals
    let a_vec = &get_vec::<f64>('a')[..512 * 1024];
    let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((&b_vec[..512], [512, 1].c(), &device)).into_dim::<Ix2>();
    let (x, s, rank, residuals) =
        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
    assert!((fingerprint(&x) - 0.1316952775202293).abs() < 1e-8);
    assert!((fingerprint(&s) - 32.27742168207754).abs() < 1e-8);
    assert_eq!(rank, 512);
    assert!(residuals.is_none());

    // rank-deficient: rank < n, residual is undefined
    let mut a_data = get_vec::<f64>('a')[..1024 * 512].to_vec();
    for i in 0..1024 {
        a_data[i * 512 + 2] = a_data[i * 512] + a_data[i * 512 + 1];
    }
    let a = rt::asarray((&a_data[..], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((&b_vec[..1024], [1024, 1].c(), &device)).into_dim::<Ix2>();
    let (x, s, rank, residuals) =
        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
    assert!((fingerprint(&x) - 0.6203201140848792).abs() < 1e-8);
    assert!((fingerprint(&s) - 46.463897960241695).abs() < 1e-8);
    assert_eq!(rank, 511);
    assert!(residuals.is_none());
}

fn make_mat(
    v: &[f64],
    m: usize,
    n: usize,
    device: &DeviceBLAS,
    f_order: bool,
    padded: bool,
) -> Tensor<f64, DeviceBLAS, Ix2> {
    let base = rt::asarray((&v[..m * n], [m, n].c(), device)).into_dim::<Ix2>();
    let base = if f_order { base.to_contig_f(ColMajor).unwrap().into_owned() } else { base.into_owned() };
    if !padded {
        return base;
    }
    let mut buf = if f_order {
        rt::zeros_f(([m + 4, n + 4], ColMajor, device)).unwrap().into_dim::<Ix2>()
    } else {
        rt::zeros_f(([m + 4, n + 4], RowMajor, device)).unwrap().into_dim::<Ix2>()
    };
    buf.i_mut((0..m, 0..n)).assign_f(base.view()).unwrap();
    buf.i((0..m, 0..n)).into_owned_keep_layout().into_dim::<Ix2>()
}

#[test]
fn test_dgelsd_layout() {
    let device = DeviceBLAS::default();
    let rcond = f64::EPSILON;
    for (m, n, k) in [(8, 5, 2), (5, 8, 1)] {
        let a_vec = &get_vec::<f64>('a')[..(m + 4) * (n + 4)];
        let b_vec = &get_vec::<f64>('b')[..(m + 4) * (k + 4)];

        let reference = {
            let a = make_mat(a_vec, m, n, &device, false, false);
            let b = make_mat(b_vec, m, k, &device, false, false);
            let (x, s, rank, residuals) =
                DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
            (fingerprint(&x), fingerprint(&s), rank, residuals.map(|r| fingerprint(&r)))
        };

        for f_order_a in [false, true] {
            for f_order_b in [false, true] {
                for padded in [false, true] {
                    let a = make_mat(a_vec, m, n, &device, f_order_a, padded);
                    let b = make_mat(b_vec, m, k, &device, f_order_b, padded);
                    let (x, s, rank, residuals) =
                        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
                    assert!((fingerprint(&x) - reference.0).abs() < 1e-8);
                    assert!((fingerprint(&s) - reference.1).abs() < 1e-8);
                    assert_eq!(rank, reference.2);
                    match (residuals.map(|r| fingerprint(&r)), reference.3) {
                        (Some(r), Some(r0)) => assert!((r - r0).abs() < 1e-8),
                        (None, None) => {},
                        _ => panic!("residual presence mismatch"),
                    }
                }
            }
        }

        // fully non-contiguous (stride 2 in both axes) a and b, as views and
        // as owned tensors keeping that layout (copied, not consumed in place)
        let strided = |tag: char, rows: usize, cols: usize| {
            let v = &get_vec::<f64>(tag)[..4 * rows * cols];
            let buf = rt::asarray((v, [2 * rows, 2 * cols].c(), &device)).into_dim::<Ix2>().into_owned();
            let view = buf.i((slice!(None, None, 2), slice!(None, None, 2))).into_dim::<Ix2>();
            let view_ref = &view;
            let gathered = (0..rows).flat_map(|i| (0..cols).map(move |j| view_ref[[i, j]])).collect::<Vec<_>>();
            let contiguous = rt::asarray((gathered, [rows, cols].c(), &device)).into_dim::<Ix2>();
            (view.into_owned_keep_layout().into_dim::<Ix2>(), contiguous)
        };
        let (mut a_strided, a_gathered) = strided('a', m, n);
        let (mut b_strided, b_gathered) = strided('b', m, k);
        let reference_strided = {
            let (x, s, rank, residuals) = DGELSD::default()
                .a(a_gathered.view())
                .b(b_gathered.view())
                .rcond(rcond)
                .build()
                .unwrap()
                .run()
                .unwrap();
            (fingerprint(&x), fingerprint(&s), rank, residuals.map(|r| fingerprint(&r)))
        };
        let (x, s, rank, residuals) =
            DGELSD::default().a(a_strided.view()).b(b_strided.view()).rcond(rcond).build().unwrap().run().unwrap();
        assert!((fingerprint(&x) - reference_strided.0).abs() < 1e-8);
        assert!((fingerprint(&s) - reference_strided.1).abs() < 1e-8);
        assert_eq!(rank, reference_strided.2);
        assert_eq!(residuals.map(|r| fingerprint(&r)), reference_strided.3);
        let (x, s, rank, residuals) = DGELSD::default()
            .a(a_strided.view_mut())
            .b(b_strided.view_mut())
            .rcond(rcond)
            .build()
            .unwrap()
            .run()
            .unwrap();
        assert!((fingerprint(&x) - reference_strided.0).abs() < 1e-8);
        assert!((fingerprint(&s) - reference_strided.1).abs() < 1e-8);
        assert_eq!(rank, reference_strided.2);
        assert_eq!(residuals.map(|r| fingerprint(&r)), reference_strided.3);

        // padded row-major tensors passed as mutable: used in place with ld > n
        let mut a_padded = make_mat(a_vec, m, n, &device, false, true);
        let mut b_padded = make_mat(b_vec, m, k, &device, false, true);
        let (x, s, rank, residuals) = DGELSD::default()
            .a(a_padded.view_mut())
            .b(b_padded.view_mut())
            .rcond(rcond)
            .build()
            .unwrap()
            .run()
            .unwrap();
        assert!((fingerprint(&x) - reference.0).abs() < 1e-8);
        assert!((fingerprint(&s) - reference.1).abs() < 1e-8);
        assert_eq!(rank, reference.2);
        assert_eq!(residuals.map(|r| fingerprint(&r)), reference.3);
    }
}

/// Small full-rank test matrix: a Vandermonde block scaled into ~[1, 40].
fn vander(m: usize, n: usize) -> Vec<f64> {
    (0..m).flat_map(|i| (0..n).map(move |j| 0.01 * ((i + 1) as f64).powi(j as i32))).collect()
}

fn pattern(n: usize, seed: usize) -> Vec<f64> {
    (0..n).map(|j| (((j * 7 + seed) % 11) as f64 - 5.0) / 3.0).collect()
}

#[test]
fn test_dgelsd_relations() {
    let device = DeviceBLAS::default();
    for (m, n, k) in [(8usize, 5usize, 1usize), (8, 5, 3), (6, 6, 2), (5, 8, 1), (5, 8, 3)] {
        let a_data = vander(m, n);
        let b_data = pattern(m * k, 5);
        let a = make_mat(&a_data, m, n, &device, false, false);
        let b = make_mat(&b_data, m, k, &device, false, false);
        let (x, s, rank, residuals) =
            DGELSD::default().a(a.view()).b(b.view()).rcond(f64::EPSILON).build().unwrap().run().unwrap();

        assert_eq!(rank, m.min(n));
        assert_eq!(x.shape(), &[n, k]);
        assert_eq!(s.shape(), &[m.min(n)]);

        let a2 = make_mat(&a_data, m, n, &device, false, false);
        let ax = rt::matmul(a2.view(), x.view());
        if m > n {
            let mut r = rt::zeros_f(([m, k], &device)).unwrap().into_dim::<Ix2>();
            let mut sums = vec![0.0; k];
            for i in 0..m {
                for j in 0..k {
                    let rij = b[[i, j]] - ax[[i, j]];
                    r[[i, j]] = rij;
                    sums[j] += rij * rij;
                }
            }
            // residuals are the column sums of squares of b - a x
            let res = residuals.unwrap();
            for j in 0..k {
                assert!((sums[j] - res[[j]]).abs() < 1e-10, "residual column {j}: {} vs {}", sums[j], res[[j]]);
            }
            // the residual is orthogonal to the column space of a
            let atr = rt::matmul(a2.view().into_reverse_axes(), r.view());
            assert!(atr.raw().iter().all(|v| v.abs() < 1e-9), "a^T (b - a x) must vanish");
        } else {
            assert!(residuals.is_none());
            assert!(rt::allclose(ax.view(), b.view(), Some(1e-10)));
        }

        // the singular values agree with the dedicated SVD driver
        let a3 = make_mat(&a_data, m, n, &device, false, false);
        let (s_ref, _, _) = DGESDD::default().a(a3.view()).compute_uv(false).build().unwrap().run().unwrap();
        assert!(rt::allclose(&s, &s_ref, Some(1e-12)));
    }
}

#[test]
fn test_dgelsd_rank() {
    let device = DeviceBLAS::default();
    // a diagonal matrix with graded singular values
    let scales = [1.0, 1e-1, 1e-2, 1e-3, 1e-4];
    let (m, n, k) = (8, 5, 2);
    let a_data: Vec<f64> = (0..m * n).map(|j| if j % n == j / n { scales[j % n] } else { 0.0 }).collect();
    let b_data = pattern(m * k, 3);

    for rcond in [1e-6, 3e-2, 1e-1, 0.0, 2.0] {
        let a = make_mat(&a_data, m, n, &device, false, false);
        let b = make_mat(&b_data, m, k, &device, false, false);
        let (x, s, rank, residuals) =
            DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();

        // LAPACK clamps rcond <= 0 and rcond >= 1 to machine epsilon
        let clamped = if rcond <= 0.0 || rcond >= 1.0 { f64::EPSILON } else { rcond };
        let expected = scales.iter().filter(|s| **s > clamped * scales[0]).count();
        assert_eq!(rank, expected, "rcond {rcond}");
        for (i, scale) in scales.iter().enumerate() {
            assert!((s[[i]] - scale).abs() < 1e-12, "rcond {rcond}, s[{i}] = {} vs {scale}", s[[i]]);
        }
        assert_eq!(x.shape(), &[n, k]);
        // the residual is only defined for a full-rank tall system
        assert_eq!(residuals.is_some(), expected == n);
    }
}

#[test]
fn test_dgelsd_nan() {
    let device = DeviceBLAS::default();
    let mut a_data = vander(6, 4);
    let b_data = pattern(6, 1);

    // NaN in a
    a_data[3] = f64::NAN;
    let a = make_mat(&a_data, 6, 4, &device, false, false);
    let b = make_mat(&b_data, 6, 1, &device, false, false);
    // `DLANGE('Max')` may swallow NaN on x86 (MAXSS), so the library either
    // reports info = -4 or fails to converge; both must surface as an error
    let err = DGELSD::default().a(a.view()).b(b.view()).rcond(f64::EPSILON).build().unwrap().run();
    assert!(err.is_err(), "NaN in a must not return a solution");

    // NaN rcond is rejected before the driver call
    let a = make_mat(&vander(6, 4), 6, 4, &device, false, false);
    let err = DGELSD::default().a(a.view()).b(b.view()).rcond(f64::NAN).build().unwrap().run();
    let msg = format!("{}", err.expect_err("NaN rcond must fail"));
    assert!(msg.contains("rcond must not be NaN"), "{msg}");
}

#[test]
fn test_dgelsd_zero_size() {
    let device = DeviceBLAS::default();
    let rcond = f64::EPSILON;

    // m == 0: the solution is a zero block, no singular values
    let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 5], &device)).unwrap().into_dim::<Ix2>();
    let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 2], &device)).unwrap().into_dim::<Ix2>();
    let (x, s, rank, residuals) =
        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
    assert_eq!(x.shape(), &[5, 2]);
    assert!(x.raw().iter().all(|v| *v == 0.0));
    assert_eq!(s.shape(), &[0]);
    assert_eq!(rank, 0);
    assert!(residuals.is_none());

    // n == 0: the residuals of the right-hand sides are synthesized
    let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([5usize, 0], &device)).unwrap().into_dim::<Ix2>();
    let b_data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = rt::asarray((&b_data[..], [5usize, 1].c(), &device)).into_dim::<Ix2>();
    let (x, s, rank, residuals) =
        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
    assert_eq!(x.shape(), &[0, 1]);
    assert_eq!(s.shape(), &[0]);
    assert_eq!(rank, 0);
    assert!((residuals.unwrap()[[0]] - 55.0).abs() < 1e-12);

    // k == 0: one zero column is padded for the driver, then trimmed
    let a = make_mat(&vander(6, 4), 6, 4, &device, false, false);
    let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([6usize, 0], &device)).unwrap().into_dim::<Ix2>();
    let (x, s, rank, residuals) =
        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
    assert_eq!(x.shape(), &[4, 0]);
    assert_eq!(s.shape(), &[4]);
    assert_eq!(rank, 4);
    assert_eq!(residuals.unwrap().shape(), &[0]);

    // n == 0 with k == 0: no residual column, the result is still well-formed
    let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([5usize, 0], &device)).unwrap().into_dim::<Ix2>();
    let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([5usize, 0], &device)).unwrap().into_dim::<Ix2>();
    let (x, s, rank, residuals) =
        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
    assert_eq!(x.shape(), &[0, 0]);
    assert_eq!(s.shape(), &[0]);
    assert_eq!(rank, 0);
    assert_eq!(residuals.unwrap().shape(), &[0]);

    // m == 0 and n == 0: empty residuals (the NumPy rule)
    let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
    let b: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 3], &device)).unwrap().into_dim::<Ix2>();
    let (x, s, rank, residuals) =
        DGELSD::default().a(a.view()).b(b.view()).rcond(rcond).build().unwrap().run().unwrap();
    assert_eq!(x.shape(), &[0, 3]);
    assert_eq!(s.shape(), &[0]);
    assert_eq!(rank, 0);
    assert!(residuals.is_none());
}
