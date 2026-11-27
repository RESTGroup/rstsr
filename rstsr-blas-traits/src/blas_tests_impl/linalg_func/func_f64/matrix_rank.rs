use super::*;

#[test]
fn test_matrix_rank() {
    let device = DeviceBLAS::default();
    // full-rank 2x2
    let a = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0], [2, 2].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(a.view()).to_scalar(), 2);
    // `()`, `None`, and the args default are the default tolerance too
    assert_eq!(rt::linalg::matrix_rank((a.view(), ())).to_scalar(), 2);
    assert_eq!(rt::linalg::matrix_rank((a.view(), None)).to_scalar(), 2);
    let default = rt::linalg::MatrixRankRtol::<f64, DeviceBLAS>::default();
    assert_eq!(rt::linalg::matrix_rank((a.view(), default)).to_scalar(), 2);
    // rank 1: the second row is twice the first
    let b = rt::asarray((vec![1.0f64, 2.0, 2.0, 4.0], [2, 2].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(b.view()).to_scalar(), 1);
    // zero matrix -> rank 0
    let z = rt::asarray((vec![0.0f64; 4], [2, 2].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(z.view()).to_scalar(), 0);
    // rectangular 3x2 of full column rank
    let c = rt::asarray((vec![1.0f64, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(c.view()).to_scalar(), 2);

    // a bare scalar rtol: no tensor, hence no device
    assert_eq!(rt::linalg::matrix_rank((a.view(), 1.0f64)).to_scalar(), 0);
    assert_eq!(rt::linalg::matrix_rank((a.view(), 1e-8f64)).to_scalar(), 2);
    // a 0-d rtol tensor: rtol = 1 -> tol = s_max -> nothing strictly exceeds it -> rank 0
    let big = rt::from_scalar(1.0f64, &device);
    assert_eq!(rt::linalg::matrix_rank((a.view(), big)).to_scalar(), 0);
    // a tiny rtol keeps the rank
    let tiny = rt::from_scalar(1e-8f64, &device);
    assert_eq!(rt::linalg::matrix_rank((a.view(), tiny)).to_scalar(), 2);
}

#[test]
fn test_matrix_rank_rtol_device_mismatch() {
    let mut row = DeviceBLAS::default();
    row.set_default_order(RowMajor);
    let mut col = DeviceBLAS::default();
    col.set_default_order(ColMajor);
    let a = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0], [2, 2].c(), &row));

    // the rtol array carries its own device; a mismatch against the input is caught
    // (`same_device` compares thread count and default order)
    let rtol = rt::from_scalar(1e-8f64, &col);
    assert!(rt::linalg::matrix_rank_f((a.view(), rtol)).is_err());
    // a bare scalar has no device, so nothing to mismatch
    assert_eq!(rt::linalg::matrix_rank((a.view(), 1e-8f64)).to_scalar(), 2);
}

// NumPy: numpy/linalg/tests/test_linalg.py TestMatrixRank::test_matrix_rank
#[test]
fn test_matrix_rank_numpy() {
    let device = DeviceBLAS::default();
    // full rank
    let eye = rt::asarray((
        vec![1.0f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [4, 4].c(),
        &device,
    ));
    assert_eq!(rt::linalg::matrix_rank(eye.view()).to_scalar(), 4);
    // rank deficient: eye(4) with the last diagonal entry zeroed
    let mut zeroed = vec![0.0f64; 16];
    for i in 0..3 {
        zeroed[i * 4 + i] = 1.0;
    }
    let deficient = rt::asarray((zeroed.clone(), [4, 4].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(deficient.view()).to_scalar(), 3);
    // all zeros -> zero rank
    let zeros = rt::asarray((vec![0.0f64; 16], [4, 4].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(zeros.view()).to_scalar(), 0);

    // >2 dims: stacked matrices -> [3, 4, 0]
    let mut stacked_vals = zeroed.clone();
    stacked_vals.extend([1.0f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    stacked_vals.extend(vec![0.0f64; 16]);
    let stacked = rt::asarray((stacked_vals, [3, 4, 4].c(), &device));
    let r = rt::linalg::matrix_rank(stacked.view());
    assert_eq!(r.shape(), &[3]);
    assert_eq!((r[[0]], r[[1]], r[[2]]), (3, 4, 0));

    // an exact linear dependency reduces the rank
    // rows r0 = [1,2,3,4], r1 = [5,6,7,8], r2 = r0 + r1 -> rank 2
    let dep = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.0, 8.0, 10.0, 12.0], [3, 4].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(dep.view()).to_scalar(), 2);
}

// NumPy: numpy/linalg/tests/test_linalg.py test_empty_matrix_rank
#[test]
fn test_matrix_rank_empty() {
    let device = DeviceBLAS::default();
    // a zero-size matrix axis gives rank 0
    assert_eq!(rt::linalg::matrix_rank(rt::asarray((Vec::<f64>::new(), [0, 0].c(), &device)).view()).to_scalar(), 0);
    assert_eq!(rt::linalg::matrix_rank(rt::asarray((Vec::<f64>::new(), [0, 5].c(), &device)).view()).to_scalar(), 0);
    assert_eq!(rt::linalg::matrix_rank(rt::asarray((Vec::<f64>::new(), [5, 0].c(), &device)).view()).to_scalar(), 0);

    // an empty batch stays empty
    let empty = rt::linalg::matrix_rank(rt::asarray((Vec::<f64>::new(), [0, 5, 5].c(), &device)).view());
    assert_eq!(empty.shape(), &[0]);
    // a zero matrix axis collapses to rank 0 per matrix
    let r = rt::linalg::matrix_rank(rt::asarray((Vec::<f64>::new(), [3, 0, 5].c(), &device)).view());
    assert_eq!(r.shape(), &[3]);
    assert_eq!((r[[0]], r[[1]], r[[2]]), (0, 0, 0));
    let r = rt::linalg::matrix_rank(rt::asarray((Vec::<f64>::new(), [2, 5, 0].c(), &device)).view());
    assert_eq!(r.shape(), &[2]);
    assert_eq!((r[[0]], r[[1]]), (0, 0));
    let r = rt::linalg::matrix_rank(rt::asarray((Vec::<f64>::new(), [2, 3, 0, 4].c(), &device)).view());
    assert_eq!(r.shape(), &[2, 3]);
}

#[test]
fn test_matrix_rank_empty_validates_rtol() {
    let device = DeviceBLAS::default();
    // a zero-size matrix axis: rank 0 for each matrix
    let empty = rt::asarray((Vec::<f64>::new(), [3, 0, 2].c(), &device));
    let r = rt::linalg::matrix_rank(empty.view());
    assert_eq!(r.shape(), &[3]);
    assert_eq!((r[[0]], r[[1]], r[[2]]), (0, 0, 0));

    // ...but `rtol` is validated even on the empty short-circuit: a
    // non-broadcastable array (batch is [3]) errors rather than being ignored
    let bad = rt::asarray((vec![1e-8f64; 4], [4].c(), &device));
    assert!(rt::linalg::matrix_rank_f((empty.view(), bad)).is_err());

    // ...and so is a device mismatch
    let mut col = DeviceBLAS::default();
    col.set_default_order(ColMajor);
    assert!(rt::linalg::matrix_rank_f((empty.view(), rt::from_scalar(1e-8f64, &col))).is_err());

    // a valid tolerance still yields zero rank
    let ok = rt::asarray((vec![1e-8f64, 1.0, 1e-8], [3].c(), &device));
    let r = rt::linalg::matrix_rank((empty.view(), ok));
    assert_eq!((r[[0]], r[[1]], r[[2]]), (0, 0, 0));
}

// NumPy: numpy/linalg/tests/test_linalg.py TestMatrixRank::test_symmetric_rank
// (the tolerance threshold, exercised through the SVD path — no Hermitian fast path)
#[test]
fn test_matrix_rank_tolerance() {
    let device = DeviceBLAS::default();
    // singular values 1, 1, 1, 1e-8: the last is kept iff rtol * 1 < 1e-8
    let mut vals = vec![0.0f64; 16];
    for i in 0..3 {
        vals[i * 4 + i] = 1.0;
    }
    vals[15] = 1e-8;
    let i = rt::asarray((vals, [4, 4].c(), &device));

    // default tol = 4 * eps, far below 1e-8 -> full rank
    assert_eq!(rt::linalg::matrix_rank(i.view()).to_scalar(), 4);
    let below = rt::from_scalar(0.99e-8f64, &device);
    assert_eq!(rt::linalg::matrix_rank((i.view(), below)).to_scalar(), 4);
    let above = rt::from_scalar(1.01e-8f64, &device);
    assert_eq!(rt::linalg::matrix_rank((i.view(), above)).to_scalar(), 3);
}

#[test]
fn test_matrix_rank_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    // stack of two 2x2: row-major fill -> [[1,2],[3,4]] (rank 2), [[1,2],[2,4]] (rank 1)
    let stacked = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 4.0], [2, 2, 2].c(), &device));

    let r = rt::linalg::matrix_rank(stacked.view());
    assert_eq!(r.shape(), &[2]);
    assert_eq!(r[[0]], 2);
    assert_eq!(r[[1]], 1);

    // per-matrix rtol: a tiny tolerance keeps the first, rtol = 1 collapses the second
    let rtol = rt::asarray((vec![1e-8f64, 1.0], [2].c(), &device));
    let r = rt::linalg::matrix_rank((stacked.view(), rtol));
    assert_eq!(r.shape(), &[2]);
    assert_eq!(r[[0]], 2);
    assert_eq!(r[[1]], 0);
}

#[test]
fn test_matrix_rank_colmajor() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    // column-major fill of a 2x2 -> [[1,2],[3,4]]
    let a = rt::asarray((vec![1.0f64, 3.0, 2.0, 4.0], [2, 2], &device));
    assert_eq!(rt::linalg::matrix_rank(a.view()).to_scalar(), 2);

    // the matrix axes lead, the batch trails; column-major fill of [2,2,2]
    let stacked = rt::asarray((vec![1.0f64, 3.0, 2.0, 4.0, 1.0, 2.0, 2.0, 4.0], [2, 2, 2], &device));
    let r = rt::linalg::matrix_rank(stacked.view());
    assert_eq!(r.shape(), &[2]);
    assert_eq!(r[[0]], 2);
    assert_eq!(r[[1]], 1);
}

// ranks agree on stacks stored in both storage orders and on a strided view
#[test]
fn test_matrix_rank_layouts() {
    // slot k's row-major 2x2 data: slot 0 = [[1,2],[2,4]] (rank 1), slot 1 = [[1,2],[3,4]] (rank 2)
    let slot = |k: usize| -> Vec<f64> {
        if k == 0 {
            vec![1.0, 2.0, 2.0, 4.0]
        } else {
            vec![1.0, 2.0, 3.0, 4.0]
        }
    };
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        // place the slots at the C-order positions of the stacked shape
        let mut v = Vec::new();
        match order {
            RowMajor => {
                for k in 0..2 {
                    v.extend(slot(k));
                }
            },
            ColMajor => {
                for r in 0..2 {
                    for c in 0..2 {
                        for k in 0..2 {
                            v.push(slot(k)[r * 2 + c]);
                        }
                    }
                }
            },
        }
        let stacked = stack_tensor(&v, &[2], &[2, 2], order, order, &device);

        let r = rt::linalg::matrix_rank(stacked.view());
        assert_eq!(r.shape(), &[2]);
        assert_eq!((r[[0]], r[[1]]), (1, 2), "{order:?}");

        // off-order storage gives the same ranks
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        let stacked_off = stack_tensor(&v, &[2], &[2, 2], order, other, &device);
        let r_off = rt::linalg::matrix_rank(stacked_off.view());
        assert_eq!((r_off[[0]], r_off[[1]]), (1, 2));

        // an array rtol broadcasts in device order: rtol = 1 collapses slot 0 only
        let rtol = rt::asarray((vec![1.0f64, 1e-8], [2].c(), &device));
        let r = rt::linalg::matrix_rank((stacked.view(), rtol));
        assert_eq!((r[[0]], r[[1]]), (0, 2), "{order:?}");

        // strided batch: every second slot of a 4-slot buffer (slots {0, 2}
        // hold slot 0 scaled by 1 and 3 — both rank 1; the data sits at the
        // C-order positions of shape [4, 2, 2] on a row-major device and
        // [2, 2, 4] on a column-major one)
        let mut wide = Vec::with_capacity(32);
        match order {
            RowMajor => {
                for k in 0..4 {
                    for j in 0..4 {
                        wide.push(slot(0)[j] * (k as f64 + 1.0));
                    }
                }
            },
            ColMajor => {
                for r in 0..2 {
                    for c in 0..2 {
                        for k in 0..4 {
                            wide.push(slot(0)[r * 2 + c] * (k as f64 + 1.0));
                        }
                    }
                }
            },
        }
        let wide_t = match order {
            RowMajor => rt::asarray((wide, [4, 2, 2].c(), &device)),
            ColMajor => rt::asarray((wide, [2, 2, 4].c(), &device)),
        };
        let picked = match order {
            RowMajor => wide_t.i((slice!(None, None, 2), .., ..)).into_dyn(),
            ColMajor => wide_t.i((.., .., slice!(None, None, 2))).into_dyn(),
        };
        let r = rt::linalg::matrix_rank(picked.view());
        assert_eq!(r.shape(), &[2]);
        assert_eq!((r[[0]], r[[1]]), (1, 1));
    }
}

// NumPy golden value from `func_validation_f64.py` (`# ### matrix_rank`)
#[test]
fn test_matrix_rank_fingerprint() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
    let r = rt::linalg::matrix_rank(a.view()).to_scalar();
    assert_eq!(r, 512);
}
