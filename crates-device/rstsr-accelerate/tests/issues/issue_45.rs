#[test]
fn issue_45() {
    use rstsr::prelude::*;
    use rstsr_accelerate::DeviceAccelerate;
    let device = DeviceAccelerate::default();
    let a: Tensor<f64, _> = rt::asarray((vec![], [1024, 0], &device));
    let b: Tensor<f64, _> = rt::asarray((vec![], [1000, 0], &device));
    let c = &a % b.t();
    println!("{:?}", c.shape());
    assert!(c.abs().sum() < 1e-10);

    let c = &a % a.t();
    assert!(c.abs().sum() < 1e-10);
}
