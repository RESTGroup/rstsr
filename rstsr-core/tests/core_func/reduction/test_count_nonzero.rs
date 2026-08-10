#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_count_nonzero {
    use super::*;
    static FUNC: &str = "numpy_count_nonzero";

    #[test]
    fn test_numeric() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestNonarrayArgs::test_count_nonzero (line 126)
        crate::specify_test!("test_numeric");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr = [[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]]
        // tgt = np.array([2, 3])
        // out = np.count_nonzero(arr, axis=1)
        // assert_equal(out, tgt)
        let arr = rt::tensor_from_nested!([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]], &device);
        let out = arr.count_nonzero_axes(1);
        let expected = rt::tensor_from_nested!([2, 3], &device);
        assert_equal(&out, &expected, None);
    }
}

#[cfg(test)]
mod custom_count_nonzero {
    use super::*;
    static FUNC: &str = "custom_count_nonzero";

    #[test]
    fn test_count_nonzero_all() {
        crate::specify_test!("test_count_nonzero_all");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // count_nonzero over all axes -> scalar count.
        let arr = rt::tensor_from_nested!([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]], &device);
        let out = arr.count_nonzero_axes(None);
        assert_eq!(out.to_scalar(), 5);
    }
}
