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

        // matrix product (axes = 1)
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let b = rt::tensor_from_nested!([[5, 6], [7, 8]], &device);
        let c = rt::tensordot(&a, &b, 1);
        assert_eq!(c.shape(), &[2, 2]);
        println!("{c}");
        // [[ 19 22]
        //  [ 43 50]]
        let expected = rt::tensor_from_nested!([[19, 22], [43, 50]], &device);
        assert!(rt::allclose(&c, &expected, None));
        assert_eq!(format!("{c}"), "[[ 19 22]\n [ 43 50]]");

        // outer product (axes = 0)
        let a = rt::tensor_from_nested!([1, 2], &device);
        let b = rt::tensor_from_nested!([3, 4], &device);
        let c = rt::tensordot(&a, &b, 0);
        println!("{c}");
        // [[ 3 4]
        //  [ 6 8]]
        let expected = rt::tensor_from_nested!([[3, 4], [6, 8]], &device);
        assert!(rt::allclose(&c, &expected, None));
        assert_eq!(format!("{c}"), "[[ 3 4]\n [ 6 8]]");
    }
}
