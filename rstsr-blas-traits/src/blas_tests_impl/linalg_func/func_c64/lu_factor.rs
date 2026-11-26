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
