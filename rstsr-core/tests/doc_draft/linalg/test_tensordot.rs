use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_tensordot {
    use super::*;
    static FUNC: &str = "doc_tensordot";

    #[test]
    fn test_tensordot() {
        crate::specify_test!("test_tensordot");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // outer product (axes = 0)
        let a = rt::tensor_from_nested!([1, 2], &device);
        let b = rt::tensor_from_nested!([3, 4], &device);
        let c = rt::tensordot(&a, &b, 0);
        assert_eq!(c.shape(), &[2, 2]);
        println!("{c}");

        // matrix product (axes = 1)
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let b = rt::tensor_from_nested!([[5, 6], [7, 8]], &device);
        let c = rt::tensordot(&a, &b, 1);
        assert_eq!(c.shape(), &[2, 2]);
        println!("{c}");
    }
}
