#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_zeros {
    use super::*;
    static FUNC: &str = "numpy_zeros";

    #[test]
    fn test_numeric() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestCreationFuncs::test_zeros (line 3384)
        // NumPy's check_function iterates dtype/order; rstsr checks shape + fill value.
        crate::specify_test!("test_numeric");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let z: Tensor<i32, _> = rt::zeros(([2, 3], &device));
        assert_eq!(z.shape(), &[2, 3]);
        assert_equal(&z, &rt::tensor_from_nested!([[0, 0, 0], [0, 0, 0]], &device), None);
    }
}

#[cfg(test)]
mod custom_zeros {
    use super::*;
    static FUNC: &str = "custom_zeros";

    #[test]
    fn test_zeros_like() {
        crate::specify_test!("test_zeros_like");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let z: Tensor<i32, _> = a.zeros_like();
        assert_eq!(z.shape(), &[2, 3]);
        assert_equal(&z, &rt::tensor_from_nested!([[0, 0, 0], [0, 0, 0]], &device), None);
    }
}
