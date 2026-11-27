use super::*;

#[test]
fn test_matrix_rank() {
    let device = DeviceBLAS::default();
    // full rank: A = [[1+1j, 2], [3, 4-1j]]
    let a = rt::asarray((vec![c64!(1.0, 1.0), c64!(2.0), c64!(3.0), c64!(4.0, -1.0)], [2, 2].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(a.view()).to_scalar(), 2);

    // rank 1: the second row is 2 times the first
    let b = rt::asarray((vec![c64!(1.0), c64!(2.0, 1.0), c64!(2.0), c64!(4.0, 2.0)], [2, 2].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(b.view()).to_scalar(), 1);

    // the rtol is a real tensor (or a bare scalar) even for a complex input
    let tight = rt::from_scalar(1e-8f64, &device);
    assert_eq!(rt::linalg::matrix_rank((a.view(), tight)).to_scalar(), 2);
    assert_eq!(rt::linalg::matrix_rank((a.view(), 1.0f64)).to_scalar(), 0);
    let big = rt::from_scalar(1.0f64, &device);
    assert_eq!(rt::linalg::matrix_rank((a.view(), big)).to_scalar(), 0);

    // rectangular full column rank
    let c = rt::asarray((vec![c64!(1.0), c64!(0.0), c64!(0.0), c64!(1.0), c64!(1.0), c64!(1.0)], [3, 2].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(c.view()).to_scalar(), 2);

    // zero matrix -> rank 0; the default tolerance form
    let z = rt::asarray((vec![c64!(0.0); 4], [2, 2].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(z.view()).to_scalar(), 0);
    assert_eq!(rt::linalg::matrix_rank((a.view(), None)).to_scalar(), 2);
}

// ranks agree on stacks stored in both storage orders and on a strided batch
#[test]
fn test_matrix_rank_layouts() {
    let s0 = vec![c64!(1.0), c64!(2.0), c64!(2.0), c64!(4.0)]; // rank 1
    let s1 = vec![c64!(1.0), c64!(0.0), c64!(0.0), c64!(1.0)]; // rank 2

    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        // place the slots at the C-order positions of the stacked shape
        let mut v: Vec<c64> = Vec::new();
        match order {
            RowMajor => {
                v.extend(s0.clone());
                v.extend(s1.clone());
            },
            ColMajor => {
                for r in 0..2 {
                    for c in 0..2 {
                        for k in [0usize, 1] {
                            v.push(if k == 0 { s0[r * 2 + c] } else { s1[r * 2 + c] });
                        }
                    }
                }
            },
        }
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        for stored in [order, other] {
            let stacked = stack_tensor(&v, &[2], &[2, 2], order, stored, &device);
            let r = rt::linalg::matrix_rank(stacked.view());
            assert_eq!(r.shape(), &[2]);
            assert_eq!((r[[0]], r[[1]]), (1, 2), "{order:?}/{stored:?}");
        }

        // per-matrix rtol: rtol = 1 collapses slot 0 only
        let stacked = stack_tensor(&v, &[2], &[2, 2], order, order, &device);
        let rtol = rt::asarray((vec![1.0f64, 1e-8], [2].c(), &device));
        let r = rt::linalg::matrix_rank((stacked.view(), rtol));
        assert_eq!((r[[0]], r[[1]]), (0, 2), "{order:?}");

        // strided batch: every second slot of a 4-slot buffer (slots {0, 2}
        // hold slot 0 scaled by 1 and 3 — both rank 1; the data sits at the
        // C-order positions of shape [4, 2, 2] on a row-major device and
        // [2, 2, 4] on a column-major one)
        let mut wide: Vec<c64> = Vec::with_capacity(16);
        match order {
            RowMajor => {
                for k in 0..4 {
                    for x in &s0 {
                        wide.push(*x * (k as f64 + 1.0));
                    }
                }
            },
            ColMajor => {
                for r in 0..2 {
                    for c in 0..2 {
                        for k in 0..4 {
                            wide.push(s0[r * 2 + c] * (k as f64 + 1.0));
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

#[test]
fn test_matrix_rank_nd_colmajor() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    // the matrix axes lead, the batch trails; column-major fill of [2,2,2]
    let stacked = rt::asarray((
        vec![c64!(1.0), c64!(3.0), c64!(2.0), c64!(4.0), c64!(1.0), c64!(2.0), c64!(2.0), c64!(4.0)],
        [2, 2, 2],
        &device,
    ));
    let r = rt::linalg::matrix_rank(stacked.view());
    assert_eq!(r.shape(), &[2]);
    assert_eq!(r[[0]], 2);
    assert_eq!(r[[1]], 1);

    // an empty batch stays empty (the batch axis trails under ColMajor); a
    // zero-size matrix axis gives rank 0
    let empty = rt::asarray((Vec::<c64>::new(), [3, 3, 0].c(), &device));
    assert_eq!(rt::linalg::matrix_rank(empty.view()).shape(), &[0]);
    let za = rt::asarray((Vec::<c64>::new(), [0, 2, 3].c(), &device));
    let r = rt::linalg::matrix_rank(za.view());
    assert_eq!(r.shape(), &[3]);
    assert_eq!((r[[0]], r[[1]], r[[2]]), (0, 0, 0));

    // the rtol array is validated before the empty short-circuit
    let bad = rt::asarray((vec![1e-8f64; 4], [4].c(), &device));
    assert!(rt::linalg::matrix_rank_f((za.view(), bad)).is_err());
}

// NumPy golden value from `func_validation_c64.py` (`# ### matrix_rank`)
#[test]
fn test_matrix_rank_fingerprint() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
    let r = rt::linalg::matrix_rank(a.view()).to_scalar();
    assert_eq!(r, 512);
}
