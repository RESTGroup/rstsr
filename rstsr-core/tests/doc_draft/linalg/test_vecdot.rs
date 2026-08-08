use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_vecdot {
    use super::*;
    static FUNC: &str = "doc_vecdot";

    #[test]
    fn test_vecdot() {
        crate::specify_test!("test_vecdot");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Basic vector dot product
        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let b = rt::tensor_from_nested!([4, 5, 6], &device);
        let result = rt::vecdot(&a, &b, None);
        println!("{result}");
        // 32
        assert_eq!(result.to_scalar(), 32);

        // 2-dim dot product
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let b = rt::tensor_from_nested!([[5, 6], [7, 8]], &device);
        let result = rt::vecdot(&a, &b, None);
        println!("{result}");
        // [ 17 53]
        let expected = rt::tensor_from_nested!([17, 53], &device);
        assert!(rt::allclose(&result, &expected, None));

        // 2-dim broadcasted dot product (row-major)
        let a = rt::tensor_from_nested!([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]], &device);
        let b = rt::tensor_from_nested!([0., 0.6, 0.8], &device);
        let result = rt::vecdot(&a, &b, None);
        println!("{result}");
        // [ 3 8 10]
        let expected = rt::tensor_from_nested!([3., 8., 10.], &device);
        assert!(rt::allclose(&result, &expected, None));

        // complex dot product (conjugates a)
        use num::complex::c64;
        let a = rt::tensor_from_nested!([c64(1., 0.), c64(2., 2.), c64(3., 0.)], &device);
        let b = rt::tensor_from_nested!([c64(1., 0.), c64(2., 0.), c64(3., 3.)], &device);
        // 1 * 1 + (2 - 2j) * 2 + 3 * (3 + 3j) = 14 + 5j
        //          conj               identity
        let result = rt::vecdot(&a, &b, None);
        println!("{result}");
        // 14+5i
        assert_eq!(result.to_scalar(), c64(14., 5.));
    }
}
