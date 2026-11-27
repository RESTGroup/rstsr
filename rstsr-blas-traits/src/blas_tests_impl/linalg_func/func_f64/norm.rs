use super::*;

#[test]
fn test_matrix_norm() {
    let device = DeviceBLAS::default();
    // NumPy's 2x2 case (numpy/linalg/tests/test_linalg.py: test_matrix_2x2)
    let a = rt::asarray((vec![1.0f64, 3.0, 5.0, 7.0], [2, 2].c(), &device));

    // the default order is 'fro'
    assert!((rt::linalg::matrix_norm(a.view()).to_scalar() - 84f64.sqrt()).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), "fro")).to_scalar() - 84f64.sqrt()).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), "nuc")).to_scalar() - 10.0).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), 1)).to_scalar() - 10.0).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), -1)).to_scalar() - 6.0).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), 2)).to_scalar() - 9.123105625617661).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), -2)).to_scalar() - 0.8768943743823404).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), f64::INFINITY)).to_scalar() - 12.0).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), f64::NEG_INFINITY)).to_scalar() - 4.0).abs() < 1e-12);

    // keepdims
    let k = rt::linalg::matrix_norm((a.view(), ("fro", true)));
    assert_eq!(k.shape(), &[1, 1]);

    // orders a matrix has no norm for (NumPy raises for 0, 3 and an unknown string)
    assert!(rt::linalg::matrix_norm_f((a.view(), 0)).is_err());
    assert!(rt::linalg::matrix_norm_f((a.view(), 3)).is_err());
    assert!(rt::linalg::matrix_norm_f((a.view(), "nofro")).is_err());
}

// NumPy golden values from `func_validation_f64.py` (`# ### matrix_norm`);
// the reduction sums fold in a different order than NumPy's, so the large
// sums carry a 1e-6 absolute tolerance
#[test]
fn test_matrix_norm_fingerprint() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..1024 * 1024].to_vec();
    let a = rt::asarray((a_vec, [1024, 1024].c(), &device));

    let fro: f64 = rt::linalg::matrix_norm((a.view(), "fro")).to_scalar();
    assert!((fro - 1024.3940160067925).abs() < 1e-6);
    let p1: f64 = rt::linalg::matrix_norm((a.view(), 1)).to_scalar();
    assert!((p1 - 896.2903422153938).abs() < 1e-6);
    let nm1: f64 = rt::linalg::matrix_norm((a.view(), -1)).to_scalar();
    assert!((nm1 - 749.990955316318).abs() < 1e-6);
    let pinf: f64 = rt::linalg::matrix_norm((a.view(), f64::INFINITY)).to_scalar();
    assert!((pinf - 880.5882890483025).abs() < 1e-6);
    let ninf: f64 = rt::linalg::matrix_norm((a.view(), f64::NEG_INFINITY)).to_scalar();
    assert!((ninf - 751.5018181471687).abs() < 1e-6);

    // SVD-based orders on a 256x256 leading block (one GESDD call each)
    let a_vec = get_vec::<f64>('a')[..256 * 256].to_vec();
    let a = rt::asarray((a_vec, [256, 256].c(), &device)).into_dim::<Ix2>();
    let nuc: f64 = rt::linalg::matrix_norm((a.view(), "nuc")).to_scalar();
    assert!((nuc - 3489.4333709872717).abs() < 1e-6);
    let p2: f64 = rt::linalg::matrix_norm((a.view(), 2)).to_scalar();
    assert!((p2 - 31.60505689274653).abs() < 1e-8);
    let nm2: f64 = rt::linalg::matrix_norm((a.view(), -2)).to_scalar();
    assert!((nm2 - 0.041947979102626105).abs() < 1e-8);
}

#[test]
fn test_matrix_norm_nd() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(RowMajor);
    // stack [a; 2a] of the 2x2 case
    let stacked = rt::asarray((vec![1.0f64, 3.0, 5.0, 7.0, 2.0, 6.0, 10.0, 14.0], [2, 2, 2].c(), &device));

    let n = rt::linalg::matrix_norm((stacked.view(), "nuc"));
    assert_eq!(n.shape(), &[2]);
    assert!((n[[0]] - 10.0).abs() < 1e-12);
    assert!((n[[1]] - 20.0).abs() < 1e-12);

    let f = rt::linalg::matrix_norm((stacked.view(), "fro"));
    assert!((f[[0]] - 84f64.sqrt()).abs() < 1e-12);
    assert!((f[[1]] - 2.0 * 84f64.sqrt()).abs() < 1e-12);

    let k = rt::linalg::matrix_norm((stacked.view(), ("nuc", true)));
    assert_eq!(k.shape(), &[2, 1, 1]);
}

#[test]
fn test_matrix_norm_nd_colmajor() {
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);
    // logical A = [[1, 3], [5, 7]]; passing a shape (not `.c()`) makes
    // `asarray` follow the device order (F), so the F-ordered data fills the
    // intended logical matrix
    let a = rt::asarray((vec![1.0f64, 5.0, 3.0, 7.0], [2, 2], &device));
    assert!((rt::linalg::matrix_norm((a.view(), "fro")).to_scalar() - 84f64.sqrt()).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), "nuc")).to_scalar() - 10.0).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), 1)).to_scalar() - 10.0).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), f64::INFINITY)).to_scalar() - 12.0).abs() < 1e-12);

    // the matrix axes are the first two under ColMajor, the batch trails
    let stacked = rt::asarray((vec![1.0f64, 5.0, 3.0, 7.0, 2.0, 10.0, 6.0, 14.0], [2, 2, 2], &device));
    let n = rt::linalg::matrix_norm((stacked.view(), "nuc"));
    assert_eq!(n.shape(), &[2]);
    assert!((n[[0]] - 10.0).abs() < 1e-12);
    assert!((n[[1]] - 20.0).abs() < 1e-12);
}

// every norm order agrees with the per-slot 2-D call on stacks stored in both
// storage orders, including a strided view of a wider buffer (the SVD path
// must gather)
#[test]
fn test_matrix_norm_layouts() {
    // slot k's row-major 2x2 data: slot 0 = [[1,3],[5,7]], slot 1 = 2a
    let slot = |k: usize| -> Vec<f64> { vec![1.0, 3.0, 5.0, 7.0].into_iter().map(|x| x * (k as f64 + 1.0)).collect() };
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        let stored = order;
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

        let stacked = stack_tensor(&v, &[2], &[2, 2], order, stored, &device);
        let s0 = rt::asarray((slot(0), [2, 2].c(), &device));
        let s1 = rt::asarray((slot(1), [2, 2].c(), &device));
        let orders = [
            rt::linalg::NormOrd::Fro,
            rt::linalg::NormOrd::Nuc,
            rt::linalg::NormOrd::P(1.0),
            rt::linalg::NormOrd::P(-1.0),
            rt::linalg::NormOrd::P(2.0),
            rt::linalg::NormOrd::P(-2.0),
            rt::linalg::NormOrd::Inf,
            rt::linalg::NormOrd::NegInf,
        ];
        for nrm in orders {
            let nd = rt::linalg::matrix_norm((stacked.view(), nrm));
            let d0: f64 = rt::linalg::matrix_norm((s0.view(), nrm)).to_scalar();
            let d1: f64 = rt::linalg::matrix_norm((s1.view(), nrm)).to_scalar();
            assert!((nd[[0]] - d0).abs() <= 1e-12 + 1e-12 * d0.abs(), "{nrm:?} slot 0");
            assert!((nd[[1]] - d1).abs() <= 1e-12 + 1e-12 * d1.abs(), "{nrm:?} slot 1");
        }

        // off-order storage: same logical stack, the other memory order, every
        // order (the SVD-based ones must gather the off-order slots)
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        let stacked_off = stack_tensor(&v, &[2], &[2, 2], order, other, &device);
        for nrm in orders {
            let nd = rt::linalg::matrix_norm((stacked_off.view(), nrm));
            let d0: f64 = rt::linalg::matrix_norm((s0.view(), nrm)).to_scalar();
            let d1: f64 = rt::linalg::matrix_norm((s1.view(), nrm)).to_scalar();
            assert!((nd[[0]] - d0).abs() <= 1e-12 + 1e-12 * d0.abs(), "off-order {nrm:?} slot 0");
            assert!((nd[[1]] - d1).abs() <= 1e-12 + 1e-12 * d1.abs(), "off-order {nrm:?} slot 1");
        }

        // a strided stack: every second slot of a 4-slot buffer (slots {0, 2}
        // hold slot 0 scaled by 1 and 3; the data sits at the C-order positions
        // of shape [4, 2, 2] on a row-major device and [2, 2, 4] on a
        // column-major one)
        let mut wide = Vec::with_capacity(32);
        match order {
            RowMajor => {
                for k in 0..4 {
                    for x in &slot(0) {
                        wide.push(x * (k as f64 + 1.0));
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
        let nd = rt::linalg::matrix_norm((picked.view(), "nuc"));
        let ref0 = rt::asarray((slot(0).clone(), [2, 2].c(), &device));
        let ref2 = rt::asarray((slot(0).iter().map(|x| 3.0 * x).collect::<Vec<f64>>(), [2, 2].c(), &device));
        let d0: f64 = rt::linalg::matrix_norm((ref0.view(), "nuc")).to_scalar();
        let d1: f64 = rt::linalg::matrix_norm((ref2.view(), "nuc")).to_scalar();
        assert!((nd[[0]] - d0).abs() <= 1e-12 + 1e-12 * d0.abs());
        assert!((nd[[1]] - d1).abs() <= 1e-12 + 1e-12 * d1.abs());
    }
}

#[test]
fn test_vector_norm() {
    let device = DeviceBLAS::default();
    // NumPy's vector case (numpy/linalg/tests/test_linalg.py: test_vector)
    let v = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0], [4].c(), &device));

    assert!((rt::linalg::vector_norm(v.view()).to_scalar() - 30f64.sqrt()).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), 2)).to_scalar() - 30f64.sqrt()).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), 1)).to_scalar() - 10.0).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), 0)).to_scalar() - 4.0).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), -1)).to_scalar() - 12.0 / 25.0).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), -2)).to_scalar() - (205f64 / 144.0).powf(-0.5)).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), 3)).to_scalar() - 100f64.powf(1.0 / 3.0)).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), f64::INFINITY)).to_scalar() - 4.0).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), f64::NEG_INFINITY)).to_scalar() - 1.0).abs() < 1e-12);

    // 'fro'/'nuc' are matrix orders, rejected for vectors
    assert!(rt::linalg::vector_norm_f((v.view(), "fro")).is_err());
    assert!(rt::linalg::vector_norm_f((v.view(), "nuc")).is_err());
}

// NumPy golden values from `func_validation_f64.py` (`# ### vector_norm`)
#[test]
fn test_vector_norm_fingerprint() {
    let device = DeviceBLAS::default();
    let v_vec = get_vec::<f64>('a')[..1024 * 1024].to_vec();
    let v = rt::asarray((v_vec, [1024 * 1024].c(), &device)).into_dim::<Ix1>();

    let p2: f64 = rt::linalg::vector_norm(v.view()).to_scalar();
    assert!((p2 - 1024.3940160067925).abs() < 1e-6);
    let p1: f64 = rt::linalg::vector_norm((v.view(), 1)).to_scalar();
    assert!((p1 - 837218.0185799017).abs() < 1e-6);
    let pinf: f64 = rt::linalg::vector_norm((v.view(), f64::INFINITY)).to_scalar();
    assert!((pinf - 4.829436008511157).abs() < 1e-8);
    let ninf: f64 = rt::linalg::vector_norm((v.view(), f64::NEG_INFINITY)).to_scalar();
    assert!((ninf - 7.816432447305714e-07).abs() < 1e-12);
    let p3: f64 = rt::linalg::vector_norm((v.view(), 3)).to_scalar();
    assert!((p3 - 118.73403684817686).abs() < 1e-6);
}

#[test]
fn test_vector_norm_axis() {
    let device = DeviceBLAS::default();
    // NumPy: x = arange(9).reshape(3, 3)
    let x = rt::asarray((vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [3, 3].c(), &device));

    // axis = 0 -> column-wise 2-norms
    let n = rt::linalg::vector_norm((x.view(), (2, AxesIndex::Val(0))));
    assert_eq!(n.shape(), &[3]);
    assert!((n[[0]] - 45f64.sqrt()).abs() < 1e-12);
    assert!((n[[1]] - 66f64.sqrt()).abs() < 1e-12);
    assert!((n[[2]] - 93f64.sqrt()).abs() < 1e-12);

    // axis = None (the default) flattens; keepdims keeps both axes
    assert!((rt::linalg::vector_norm(x.view()).to_scalar() - 204f64.sqrt()).abs() < 1e-12);
    let k = rt::linalg::vector_norm((x.view(), (rt::linalg::NormOrd::None, AxesIndex::None, true)));
    assert_eq!(k.shape(), &[1, 1]);
    assert!((k.to_scalar() - 204f64.sqrt()).abs() < 1e-12);

    // a bad axis errors even on an otherwise valid input
    assert!(rt::linalg::vector_norm_f((x.view(), (2, AxesIndex::Val(5)))).is_err());
}

// vector norms agree with the contiguous call on strided and off-order inputs
#[test]
fn test_vector_norm_layouts() {
    for order in [RowMajor, ColMajor] {
        let mut device = DeviceBLAS::default();
        device.set_default_order(order);
        // a (3, 8) buffer sliced to (3, 4): rows are strided on a row-major device
        let v: Vec<f64> = (0..24).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
        let wide = rt::asarray((v.clone(), [3, 8].c(), &device));
        let sliced = wide.i((.., ..4)).into_dim::<Ix2>();

        let n_slice = rt::linalg::vector_norm(sliced.view());
        let dense: Vec<f64> = v.chunks(8).flat_map(|r| r[..4].to_vec()).collect();
        let dense_t = rt::asarray((dense, [3, 4].c(), &device));
        let n_dense = rt::linalg::vector_norm(dense_t.view());
        assert!((n_slice.to_scalar() - n_dense.to_scalar()).abs() < 1e-12);

        // an off-order stack: same values, stored in the other memory order
        let other = match order {
            RowMajor => ColMajor,
            ColMajor => RowMajor,
        };
        let off = stack_tensor(&v[..16], &[2], &[4, 2], order, other, &device);
        let n_off = rt::linalg::vector_norm(off.view());
        let in_order = stack_tensor(&v[..16], &[2], &[4, 2], order, order, &device);
        let n_in = rt::linalg::vector_norm(in_order.view());
        assert!((n_off.to_scalar() - n_in.to_scalar()).abs() < 1e-12);
    }
}

// rstsr returns 0 for every order of an empty matrix / vector, where NumPy
// raises or returns inf on its min-based orders (see the module doc)
#[test]
fn test_norm_empty() {
    let device = DeviceBLAS::default();
    let z = rt::asarray((Vec::<f64>::new(), [0, 2].c(), &device));
    for ord in ["fro", "nuc"] {
        assert_eq!(rt::linalg::matrix_norm((z.view(), ord)).to_scalar(), 0.0);
    }
    for ord in [1.0f64, 2.0, f64::INFINITY] {
        assert_eq!(rt::linalg::matrix_norm((z.view(), ord)).to_scalar(), 0.0);
    }
    for ord in [-1.0f64, -2.0, f64::NEG_INFINITY] {
        assert_eq!(rt::linalg::matrix_norm((z.view(), ord)).to_scalar(), 0.0);
    }

    let zv = rt::asarray((Vec::<f64>::new(), [0].c(), &device));
    for ord in [1.0f64, 2.0, f64::INFINITY] {
        assert_eq!(rt::linalg::vector_norm((zv.view(), ord)).to_scalar(), 0.0);
    }
    for ord in [-1.0f64, f64::NEG_INFINITY] {
        assert_eq!(rt::linalg::vector_norm((zv.view(), ord)).to_scalar(), 0.0);
    }

    // an empty batch is empty (NumPy: shape (0,)), not an error
    let eb = rt::asarray((Vec::<f64>::new(), [0, 1, 1].c(), &device));
    assert_eq!(rt::linalg::matrix_norm((eb.view(), f64::NEG_INFINITY)).shape(), &[0]);
    let eb2 = rt::asarray((Vec::<f64>::new(), [0, 3].c(), &device));
    assert_eq!(rt::linalg::vector_norm((eb2.view(), (2, AxesIndex::Val(1)))).shape(), &[0]);
}

// every overload form on small inputs; the result shapes follow keepdims/axes
#[test]
fn test_norm_overloads() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((vec![1.0f64, 3.0, 5.0, 7.0], [2, 2].c(), &device));
    let v = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0], [4].c(), &device));

    // matrix_norm: bare, string, isize, f64, inf, (ord, keepdims), bool, args
    assert!((rt::linalg::matrix_norm(&a).to_scalar() - 84f64.sqrt()).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), "fro")).to_scalar() - 84f64.sqrt()).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), 1)).to_scalar() - 10.0).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), 2.0)).to_scalar() - 9.123105625617661).abs() < 1e-12);
    assert!((rt::linalg::matrix_norm((a.view(), f64::INFINITY)).to_scalar() - 12.0).abs() < 1e-12);
    assert_eq!(rt::linalg::matrix_norm((a.view(), ("fro", true))).shape(), &[1, 1]);
    assert_eq!(rt::linalg::matrix_norm((a.view(), true)).shape(), &[1, 1]);
    let args = rt::linalg::MatrixNormArgs::default();
    assert!((rt::linalg::matrix_norm((a.view(), args)).to_scalar() - 84f64.sqrt()).abs() < 1e-12);

    // vector_norm: bare, isize, (ord, axes), (ord, axes, keepdims), bad axis
    assert!((rt::linalg::vector_norm(&v).to_scalar() - 30f64.sqrt()).abs() < 1e-12);
    assert!((rt::linalg::vector_norm((v.view(), 1)).to_scalar() - 10.0).abs() < 1e-12);
    assert_eq!(rt::linalg::vector_norm((v.view(), (2, AxesIndex::Val(0)))).shape(), &[] as &[usize]);
    let vv = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0], [2, 2].c(), &device));
    assert_eq!(rt::linalg::vector_norm((vv.view(), (2, AxesIndex::Val(1)))).shape(), &[2]);
    assert_eq!(rt::linalg::vector_norm((vv.view(), (2, AxesIndex::Val(1), true))).shape(), &[2, 1]);
    assert_eq!(rt::linalg::vector_norm((vv.view(), (rt::linalg::NormOrd::None, AxesIndex::None, true))).shape(), &[
        1, 1
    ]);
    assert!(rt::linalg::vector_norm_f((vv.view(), (2, AxesIndex::Val(5)))).is_err());
    assert!(rt::linalg::vector_norm_f((vv.view(), "fro")).is_err());

    // rank rules: a 1-D input has no matrix norm; a 0-D input is a one-element
    // vector, so its vector norm is |x| (as NumPy)
    let v1 = rt::asarray((vec![1.0f64, 2.0], [2].c(), &device));
    assert!(rt::linalg::matrix_norm_f(v1.view()).is_err());
    let x0 = rt::from_scalar(-3.0f64, &device);
    let n0 = rt::linalg::vector_norm(x0.view());
    assert_eq!(n0.shape(), &[] as &[usize]);
    assert!((n0.to_scalar() - 3.0).abs() < 1e-12);
}
