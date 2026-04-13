#[test]
pub fn issue() {
    use rstsr::prelude::*;
    let mut device = DeviceCpuSerial::default();
    device.set_default_order(RowMajor);

    let a = rt::arange((12, &device)).into_shape((3, 4));
    let b = a.i(1..);
    let c = b.to_contig(RowMajor);
    println!("=== b ===\n{:}", b);
    println!("=== c ===\n{:}", c);
    assert!(rt::allclose(&b, &c, None));
}
