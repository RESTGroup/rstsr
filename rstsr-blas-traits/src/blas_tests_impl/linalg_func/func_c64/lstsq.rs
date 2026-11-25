use super::*;

#[test]
fn test_lstsq() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let b_vec = get_vec::<c64>('b')[..1024 * 2].to_vec();

    // tall, K = 1
    let a = rt::asarray((&a_vec[..], [1024, 512].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((&b_vec[..1024], [1024, 1].c(), &device)).into_dim::<Ix2>();
    let (x, residuals, rank, s) = rt::linalg::lstsq((a.view(), b.view())).into();
    assert_eq!(x.shape(), &[512, 1]);
    assert_eq!(residuals.shape(), &[1]);
    assert_eq!(rank.raw()[0], 512);
    assert!((fingerprint(&x.abs()) - 0.46834637017620495).abs() < 1e-8);
    assert!((fingerprint(&s) - 46.603434059217975).abs() < 1e-8);
    assert!((fingerprint(&residuals) - 1062.17917639187).abs() < 1e-8);

    // tall, K = 2
    let b = rt::asarray((&b_vec[..], [1024, 2].c(), &device)).into_dim::<Ix2>();
    let (x, residuals, rank, s) = rt::linalg::lstsq((a.view(), b.view())).into();
    assert_eq!(x.shape(), &[512, 2]);
    assert_eq!(residuals.shape(), &[2]);
    assert_eq!(rank.raw()[0], 512);
    assert!((fingerprint(&x.abs()) - 0.10730347392277714).abs() < 1e-8);
    assert!((fingerprint(&s) - 46.603434059217975).abs() < 1e-8);
    assert!((fingerprint(&residuals) - 1613.436740523908).abs() < 1e-8);

    // wide, K = 1: minimum-norm solution, empty residuals
    let a_vec = get_vec::<c64>('a')[..512 * 1024].to_vec();
    let a = rt::asarray((&a_vec[..], [512, 1024].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((&b_vec[..512], [512, 1].c(), &device)).into_dim::<Ix2>();
    let (x, residuals, rank, s) = rt::linalg::lstsq((a.view(), b.view())).into();
    assert_eq!(x.shape(), &[1024, 1]);
    assert_eq!(residuals.shape(), &[0]);
    assert_eq!(rank.raw()[0], 512);
    assert!((fingerprint(&x.abs()) - 0.7685570101997597).abs() < 1e-8);
    assert!((fingerprint(&s) - 47.59927483588676).abs() < 1e-8);
}
