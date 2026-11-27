use super::*;

#[test]
fn test_matrix_norm() {
    let device = DeviceBLAS::default();
    // A = [[1+1j, 2], [3, 4-1j]]
    let a = rt::asarray((vec![c64!(1.0, 1.0), c64!(2.0), c64!(3.0), c64!(4.0, -1.0)], [2, 2].c(), &device));

    // fro = sqrt(2 + 4 + 9 + 17) = sqrt(32), real output
    assert!((rt::linalg::matrix_norm((a.view(), "fro")).to_scalar() - 32f64.sqrt()).abs() < 1e-12);
    // 1-norm = max column sum of |x| = 2 + sqrt(17)
    assert!((rt::linalg::matrix_norm((a.view(), 1)).to_scalar() - (2.0 + 17f64.sqrt())).abs() < 1e-12);
    // inf-norm = max row sum of |x| = 3 + sqrt(17)
    assert!((rt::linalg::matrix_norm((a.view(), f64::INFINITY)).to_scalar() - (3.0 + 17f64.sqrt())).abs() < 1e-12);

    // keepdims and the invalid orders
    assert_eq!(rt::linalg::matrix_norm((a.view(), ("fro", true))).shape(), &[1, 1]);
    assert!(rt::linalg::matrix_norm_f((a.view(), 0)).is_err());
    assert!(rt::linalg::matrix_norm_f((a.view(), "nofro")).is_err());
}

// NumPy golden values from `func_validation_c64.py` (`# ### matrix_norm`);
// the reduction sums fold in a different order than NumPy's, so the large
// sums carry a 1e-6 absolute tolerance
#[test]
fn test_matrix_norm_fingerprint() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..1024 * 1024].to_vec();
    let a = rt::asarray((a_vec, [1024, 1024].c(), &device));

    let fro: f64 = rt::linalg::matrix_norm((a.view(), "fro")).to_scalar();
    assert!((fro - 1449.1955776423124).abs() < 1e-6);
    let p1: f64 = rt::linalg::matrix_norm((a.view(), 1)).to_scalar();
    assert!((p1 - 1359.9335864360062).abs() < 1e-6);
    let nm1: f64 = rt::linalg::matrix_norm((a.view(), -1)).to_scalar();
    assert!((nm1 - 1222.892551470915).abs() < 1e-6);
    let pinf: f64 = rt::linalg::matrix_norm((a.view(), f64::INFINITY)).to_scalar();
    assert!((pinf - 1365.5480489523034).abs() < 1e-6);
    let ninf: f64 = rt::linalg::matrix_norm((a.view(), f64::NEG_INFINITY)).to_scalar();
    assert!((ninf - 1219.794834983426).abs() < 1e-6);

    // SVD-based orders on a 256x256 leading block (one GESDD call each)
    let a_vec = get_vec::<c64>('a')[..256 * 256].to_vec();
    let a = rt::asarray((a_vec, [256, 256].c(), &device)).into_dim::<Ix2>();
    let nuc: f64 = rt::linalg::matrix_norm((a.view(), "nuc")).to_scalar();
    assert!((nuc - 4942.898431230364).abs() < 1e-6);
    let p2: f64 = rt::linalg::matrix_norm((a.view(), 2)).to_scalar();
    assert!((p2 - 44.84157812023554).abs() < 1e-8);
    let nm2: f64 = rt::linalg::matrix_norm((a.view(), -2)).to_scalar();
    assert!((nm2 - 0.023430892892404798).abs() < 1e-8);
}

// the stack [a; 2a] of the 2x2 case agrees with the per-slot calls in both
// storage orders, including a strided batch (the SVD path must gather)
#[test]
fn test_matrix_norm_layouts() {
    let s0 = vec![c64!(1.0, 1.0), c64!(2.0), c64!(3.0), c64!(4.0, -1.0)];
    let s1: Vec<c64> = s0.iter().map(|x| 2.0 * x).collect();

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
            let s0_t = stack_tensor(&s0, &[], &[2, 2], order, stored, &device);
            let s1_t = stack_tensor(&s1, &[], &[2, 2], order, stored, &device);
            let n = rt::linalg::matrix_norm((stacked.view(), "nuc"));
            let d0: f64 = rt::linalg::matrix_norm((s0_t.view(), "nuc")).to_scalar();
            let d1: f64 = rt::linalg::matrix_norm((s1_t.view(), "nuc")).to_scalar();
            assert_eq!(n.shape(), &[2]);
            assert!((n[[0]] - d0).abs() < 1e-12, "{order:?}/{stored:?} slot 0");
            assert!((n[[1]] - d1).abs() < 1e-12, "{order:?}/{stored:?} slot 1");
            // and the fro norms scale with the slot factor
            let f = rt::linalg::matrix_norm((stacked.view(), "fro"));
            assert!((f[[1]] - 2.0 * f[[0]]).abs() < 1e-12);
        }

        // strided batch: every second slot of a 4-slot buffer (slots {0, 2}
        // hold slot 0 scaled by 1 and 3; the data sits at the C-order positions
        // of shape [4, 2, 2] on a row-major device and [2, 2, 4] on a
        // column-major one)
        let mut wide: Vec<c64> = Vec::with_capacity(16);
        match order {
            RowMajor => {
                for k in 0..4 {
                    for x in &s0 {
                        wide.push(x * (k as f64 + 1.0));
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
        let nd = rt::linalg::matrix_norm((picked.view(), "fro"));
        let ref0 = rt::asarray((s0.clone(), [2, 2].c(), &device));
        let ref2 = rt::asarray((s0.iter().map(|x| 3.0 * x).collect::<Vec<c64>>(), [2, 2].c(), &device));
        let d0: f64 = rt::linalg::matrix_norm((ref0.view(), "fro")).to_scalar();
        let d1: f64 = rt::linalg::matrix_norm((ref2.view(), "fro")).to_scalar();
        assert!((nd[[0]] - d0).abs() <= 1e-12 + 1e-12 * d0.abs());
        assert!((nd[[1]] - d1).abs() <= 1e-12 + 1e-12 * d1.abs());
    }
}

#[test]
fn test_vector_norm() {
    let device = DeviceBLAS::default();
    let v = rt::asarray((vec![c64!(3.0, 4.0), c64!(0.0)], [2].c(), &device));

    // |3 + 4j| = 5
    assert!((rt::linalg::vector_norm(v.view()).to_scalar() - 5.0).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), f64::INFINITY)).to_scalar() - 5.0).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), 0)).to_scalar() - 1.0).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), 1)).to_scalar() - 5.0).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), 3)).to_scalar() - (125f64).powf(1.0 / 3.0)).abs() < 1e-12);
    assert!(rt::linalg::vector_norm_f((v.view(), "nuc")).is_err());
}

// NumPy golden values from `func_validation_c64.py` (`# ### vector_norm`)
#[test]
fn test_vector_norm_fingerprint() {
    let device = DeviceBLAS::default();
    let v_vec = get_vec::<c64>('a')[..1024 * 1024].to_vec();
    let v = rt::asarray((v_vec, [1024 * 1024].c(), &device)).into_dim::<Ix1>();

    let p2: f64 = rt::linalg::vector_norm(v.view()).to_scalar();
    assert!((p2 - 1449.1955776423124).abs() < 1e-6);
    let p1: f64 = rt::linalg::vector_norm((v.view(), 1)).to_scalar();
    assert!((p1 - 1315110.649779901).abs() < 1e-6);
    let pinf: f64 = rt::linalg::vector_norm((v.view(), f64::INFINITY)).to_scalar();
    assert!((pinf - 5.418092204281868).abs() < 1e-8);
    let ninf: f64 = rt::linalg::vector_norm((v.view(), f64::NEG_INFINITY)).to_scalar();
    assert!((ninf - 0.0008315180802683977).abs() < 1e-12);
    let p3: f64 = rt::linalg::vector_norm((v.view(), 3)).to_scalar();
    assert!((p3 - 158.08987792483444).abs() < 1e-6);
}

// an empty complex input is 0 under every order (the rstsr divergence)
#[test]
fn test_norm_empty() {
    let device = DeviceBLAS::default();
    let z = rt::asarray((vec![c64!(0.0); 0], [0, 2].c(), &device));
    for ord in ["fro", "nuc"] {
        assert_eq!(rt::linalg::matrix_norm((z.view(), ord)).to_scalar(), 0.0);
    }
    for ord in [1.0f64, 2.0, f64::INFINITY, -2.0, f64::NEG_INFINITY] {
        assert_eq!(rt::linalg::matrix_norm((z.view(), ord)).to_scalar(), 0.0);
    }
    let zv = rt::asarray((vec![c64!(0.0); 0], [0].c(), &device));
    assert_eq!(rt::linalg::vector_norm(zv.view()).to_scalar(), 0.0);
}
