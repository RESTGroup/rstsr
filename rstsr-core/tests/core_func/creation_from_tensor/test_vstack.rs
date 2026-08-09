#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

// Parity tests for `vstack` (stack vertically / row-wise).
//
// Source: NumPy v2.5.2, `_core/tests/test_shape_base.py::TestVstack`.
// rstsr `vstack` matches NumPy: each input is promoted with `atleast_2d` (0-D ->
// (1, 1), 1-D (N,) -> (1, N)) before concatenating along axis 0, so the result is
// always at least 2-D. See [`atleast_2d`] and [`vstack`]'s docstring.
//
// Not ported (N/A, Rust typed API): test_non_iterable, test_generator,
// test_casting_and_dtype[_type_error] (TypeError / dtype / casting kwargs).

#[cfg(test)]
mod numpy_vstack {
    use super::*;
    static FUNC: &str = "numpy_vstack";

    #[test]
    fn test_empty_input() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestVstack::test_empty_input (line 199)
        crate::specify_test!("test_empty_input");

        // assert_raises(ValueError, vstack, ())
        let empty: Vec<Tensor<i64, DeviceType>> = vec![];
        assert!(rt::vstack_f(empty).is_err());
    }

    #[test]
    fn test_0d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestVstack::test_0D_array (line 202)
        crate::specify_test!("test_0d_array");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = array(1); b = array(2); res = vstack([a, b]); desired = array([[1], [2]])
        // atleast_2d promotes 0-D -> (1, 1); concat axis 0 -> (2, 1).
        let a: Tensor<i32, _> = rt::asarray((1, &device));
        let b: Tensor<i32, _> = rt::asarray((2, &device));
        let res = rt::vstack([&a, &b]);
        let desired = rt::tensor_from_nested!([[1], [2]], &device);
        assert_equal(&res, &desired, None);
    }

    #[test]
    fn test_1d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestVstack::test_1D_array (line 209)
        crate::specify_test!("test_1d_array");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = array([1]); b = array([2]); res = vstack([a, b]); desired = array([[1], [2]])
        // atleast_2d promotes 1-D (1,) -> (1, 1); concat axis 0 -> (2, 1).
        let a = rt::tensor_from_nested!([1], &device);
        let b = rt::tensor_from_nested!([2], &device);
        let res = rt::vstack([&a, &b]);
        let desired = rt::tensor_from_nested!([[1], [2]], &device);
        assert_equal(&res, &desired, None);
    }

    #[test]
    fn test_2d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestVstack::test_2D_array (line 216)
        crate::specify_test!("test_2d_array");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = array([[1], [2]]); b = array([[1], [2]]); desired = array([[1], [2], [1], [2]])
        let a = rt::tensor_from_nested!([[1], [2]], &device);
        let b = rt::tensor_from_nested!([[1], [2]], &device);
        let res = rt::vstack([&a, &b]);
        let desired = rt::tensor_from_nested!([[1], [2], [1], [2]], &device);
        assert_equal(&res, &desired, None);
    }

    #[test]
    fn test_2d_array2() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestVstack::test_2D_array2 (line 223)
        crate::specify_test!("test_2d_array2");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = array([1, 2]); b = array([1, 2]); res = vstack([a, b])
        // desired = array([[1, 2], [1, 2]])  (atleast_2d promotes (2,) -> (1, 2))
        let a = rt::tensor_from_nested!([1, 2], &device);
        let b = rt::tensor_from_nested!([1, 2], &device);
        let res = rt::vstack([&a, &b]);
        let desired = rt::tensor_from_nested!([[1, 2], [1, 2]], &device);
        assert_equal(&res, &desired, None);
    }
}
