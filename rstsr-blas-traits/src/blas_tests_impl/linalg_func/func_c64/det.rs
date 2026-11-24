use super::*;

#[test]
fn test_det() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..5 * 5].to_vec();
    let mut a = rt::asarray((a_vec, [5, 5].c(), &device));

    let det = rt::linalg::det(a.view_mut());
    assert!((det.to_scalar() - c64!(-24.808965756481086, 11.800248863799464)).norm() < 1e-8);
}
