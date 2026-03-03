use crate::tests_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_flip {
    use super::*;
    static FUNC: &str = "numpy_flip";

    fn get_mat(n: usize) -> Tensor<usize, DeviceType> {
        // NumPy v2.4.2, lib/tests/test_function_base.py, get_mat (line 68)

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // data = np.arange(n)
        // data = np.add.outer(data, data)
        // return data
        let data = rt::arange((n, &device));
        data.i((.., None)) + data.i((None, ..))
    }

    #[test]
    fn test_axes() {
        // NumPy v2.4.2, lib/tests/test_function_base.py, TestFlip::test_axes (line 155)
        crate::specify_test!("test_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // assert_raises(AxisError, np.flip, np.ones(4), axis=1)
        let a: Tensor<usize, _> = rt::ones(([4], &device));
        assert!(a.flip_f(1).is_err());

        // assert_raises(AxisError, np.flip, np.ones((4, 4)), axis=2)
        let a: Tensor<usize, _> = rt::ones(([4, 4], &device));
        assert!(a.flip_f(2).is_err());

        // assert_raises(AxisError, np.flip, np.ones((4, 4)), axis=-3)
        assert!(a.flip_f(-3).is_err());

        // assert_raises(AxisError, np.flip, np.ones((4, 4)), axis=(0, 3))
        assert!(a.flip_f([0, 3]).is_err());
    }

    #[test]
    fn test_basic_lr() {
        // NumPy v2.4.2, lib/tests/test_function_base.py, TestFlip::test_basic_lr (line 161)
        crate::specify_test!("test_basic_lr");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = get_mat(4)
        // b = a[:, ::-1]
        // assert_equal(np.flip(a, 1), b)
        let a = get_mat(4);
        let b = a.i((.., slice!(None, None, -1)));
        assert_equal(a.flip(1), &b, None);

        // a = [[0, 1, 2], [3, 4, 5]]
        // b = [[2, 1, 0], [5, 4, 3]]
        // assert_equal(np.flip(a, 1), b)
        let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        let b = rt::tensor_from_nested!([[2, 1, 0], [5, 4, 3]], &device);
        assert_equal(a.flip(1), &b, None);
    }

    #[test]
    fn test_basic_ud() {
        // NumPy v2.4.2, lib/tests/test_function_base.py, TestFlip::test_basic_ud (line 171)
        crate::specify_test!("test_basic_ud");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = get_mat(4)
        // b = a[::-1, :]
        // assert_equal(np.flip(a, 0), b)
        let a = get_mat(4);
        let b = a.i((slice!(None, None, -1), ..));
        assert_equal(a.flip(0), &b, None);

        // a = [[0, 1, 2], [3, 4, 5]]
        // b = [[3, 4, 5], [0, 1, 2]]
        // assert_equal(np.flip(a, 0), b)
        let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        let b = rt::tensor_from_nested!([[3, 4, 5], [0, 1, 2]], &device);
        assert_equal(a.flip(0), &b, None);
    }

    #[test]
    fn test_3d_swap_axis0() {
        // NumPy v2.4.2, lib/tests/test_function_base.py, TestFlip::test_3d_swap_axis0 (line 181)
        crate::specify_test!("test_3d_swap_axis0");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        let a = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);

        // b = np.array([[[4, 5], [6, 7]], [[0, 1], [2, 3]]])
        let b = rt::tensor_from_nested!([[[4, 5], [6, 7]], [[0, 1], [2, 3]]], &device);

        // assert_equal(np.flip(a, 0), b)
        assert_equal(a.flip(0), &b, None);
    }

    #[test]
    fn test_3d_swap_axis1() {
        // NumPy v2.4.2, lib/tests/test_function_base.py, TestFlip::test_3d_swap_axis1 (line 194)
        crate::specify_test!("test_3d_swap_axis1");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
        let b = rt::tensor_from_nested!([[[2, 3], [0, 1]], [[6, 7], [4, 5]]], &device);
        assert_equal(a.flip(1), &b, None);
    }

    #[test]
    fn test_3d_swap_axis2() {
        // NumPy v2.4.2, lib/tests/test_function_base.py, TestFlip::test_3d_swap_axis2 (line 207)
        crate::specify_test!("test_3d_swap_axis2");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
        let b = rt::tensor_from_nested!([[[1, 0], [3, 2]], [[5, 4], [7, 6]]], &device);
        assert_equal(a.flip(2), &b, None);
    }

    #[test]
    fn test_default_axis() {
        // NumPy v2.4.2, lib/tests/test_function_base.py, TestFlip::test_default_axis (line 226)
        crate::specify_test!("test_default_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.array([[1, 2, 3], [4, 5, 6]])
        let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);

        // b = np.array([[6, 5, 4], [3, 2, 1]])
        let b = rt::tensor_from_nested!([[6, 5, 4], [3, 2, 1]], &device);

        // assert_equal(np.flip(a), b)
        assert_equal(a.flip(None), &b, None);
    }

    #[test]
    fn test_multiple_axes() {
        // NumPy v2.4.2, lib/tests/test_function_base.py, TestFlip::test_multiple_axes (line 233)
        crate::specify_test!("test_multiple_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);

        // assert_equal(np.flip(a, axis=()), a)
        // Note: In NumPy, axis=() means no axes to flip, so the result equals the original array.
        // In RSTSR, empty tuple () is treated the same as None (flip all axes).
        // This behavioral difference is due to RSTSR's implementation choice.
        // assert_equal(a.flip(()), &a, None);  // Not applicable for RSTSR

        // b = np.array([[[5, 4], [7, 6]], [[1, 0], [3, 2]]])
        let b = rt::tensor_from_nested!([[[5, 4], [7, 6]], [[1, 0], [3, 2]]], &device);

        // assert_equal(np.flip(a, axis=(0, 2)), b)
        assert_equal(a.flip([0, 2]), &b, None);

        // c = np.array([[[3, 2], [1, 0]], [[7, 6], [5, 4]]])
        let c = rt::tensor_from_nested!([[[3, 2], [1, 0]], [[7, 6], [5, 4]]], &device);

        // assert_equal(np.flip(a, axis=(1, 2)), c)
        assert_equal(a.flip([1, 2]), &c, None);
    }
}

#[cfg(test)]
mod docs_flip {
    use super::*;
    static FUNC: &str = "docs_flip";

    #[test]
    fn flip_single_axis() {
        crate::specify_test!("flip_single_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Flipping the first (0) axis
        let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
        let b = a.flip(0);
        let b_expected = rt::tensor_from_nested!([[[4, 5], [6, 7]], [[0, 1], [2, 3]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));

        // Flipping is equivalent to slicing with step -1
        let b_sliced = a.i(slice!(None, None, -1));
        assert!(rt::allclose(&b_sliced, &b_expected, None));

        // Flipping the second (1) axis
        let b = a.flip(1);
        let b_expected = rt::tensor_from_nested!([[[2, 3], [0, 1]], [[6, 7], [4, 5]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
    }

    #[test]
    fn flip_multiple_axes() {
        crate::specify_test!("flip_multiple_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Flipping the first (0) and last (-1) axes
        let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
        let b = a.flip([0, -1]);
        let b_expected = rt::tensor_from_nested!([[[5, 4], [7, 6]], [[1, 0], [3, 2]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
    }

    #[test]
    fn flip_all_axes() {
        crate::specify_test!("flip_all_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Flipping all axes with None
        let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
        let b = a.flip(None);
        let b_expected = rt::tensor_from_nested!([[[7, 6], [5, 4]], [[3, 2], [1, 0]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));

        // Flipping all axes with empty tuple
        let b = a.flip(());
        assert!(rt::allclose(&b, &b_expected, None));
    }
}
