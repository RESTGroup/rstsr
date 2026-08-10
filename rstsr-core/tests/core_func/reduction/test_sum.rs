#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_sum {
    use super::*;
    static FUNC: &str = "numpy_sum";

    #[test]
    fn test_numeric() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestNonarrayArgs::test_sum (line 320)
        crate::specify_test!("test_numeric");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        // tgt = [[6], [15], [24]]
        // out = np.sum(m, axis=1, keepdims=True)
        // assert_equal(tgt, out)
        let m = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6], [7, 8, 9]], &device);
        // rstsr `sum_axes` drops the reduced axis (no `keepdims`); the reduction
        // value is identical to NumPy's keepdims result squeezed back to 1-D.
        let out = m.sum_axes(1);
        let expected = rt::tensor_from_nested!([6, 15, 24], &device);
        assert_equal(&out, &expected, None);
    }
}

#[cfg(test)]
mod custom_sum {
    use super::*;
    static FUNC: &str = "custom_sum";

    #[test]
    fn test_sum_all() {
        crate::specify_test!("test_sum_all");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.sum([[1,2,3],[4,5,6],[7,8,9]]) == 45
        let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6], [7, 8, 9]], &device);
        assert_eq!(a.sum_all(), 45);
    }

    #[test]
    fn test_sum_axes_negative_multi() {
        crate::specify_test!("test_sum_axes_negative_multi");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.arange(24).reshape(2, 3, 4)
        // np.sum(a, axis=(-2, -1)) -> shape (2,); slab 0 (0..12)=66, slab 1 (12..24)=210
        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        let out = a.sum_axes([-2, -1]);
        let expected = rt::tensor_from_nested!([66, 210], &device);
        assert_equal(&out, &expected, None);
    }

    #[test]
    fn test_sum_axis_none() {
        crate::specify_test!("test_sum_axis_none");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // sum_axes(None) reduces over all axes -> 0-d (scalar) tensor.
        let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        let out = a.sum_axes(None);
        assert_eq!(out.to_scalar(), 21);
    }
}
