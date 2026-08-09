#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

// Parity tests for `hstack` (stack horizontally / column-wise).
//
// Source: NumPy v2.5.2, `_core/tests/test_shape_base.py::TestHstack`.
// rstsr `hstack` matches NumPy: each input is promoted with `atleast_1d` (0-D ->
// (1,)) before concatenating along axis 0 for 1-D inputs, else along axis 1. See
// [`atleast_1d`] and [`hstack`]'s docstring.
//
// Not ported (N/A, Rust typed API): test_non_iterable, test_generator,
// test_casting_and_dtype[_type_error] (TypeError / dtype / casting kwargs).

#[cfg(test)]
mod numpy_hstack {
    use super::*;
    static FUNC: &str = "numpy_hstack";

    #[test]
    fn test_empty_input() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestHstack::test_empty_input (line 151)
        crate::specify_test!("test_empty_input");

        let _device = TESTCFG.device.clone();
        // assert_raises(ValueError, hstack, ())
        let empty: Vec<Tensor<i64, DeviceType>> = vec![];
        assert!(rt::hstack_f(empty).is_err());
    }

    #[test]
    fn test_1d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestHstack::test_1D_array (line 161)
        crate::specify_test!("test_1d_array");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = array([1]); b = array([2])
        // res = hstack([a, b])
        // desired = array([1, 2])
        let a = rt::tensor_from_nested!([1], &device);
        let b = rt::tensor_from_nested!([2], &device);
        let res = rt::hstack([&a, &b]);
        let desired = rt::tensor_from_nested!([1, 2], &device);
        assert_equal(&res, &desired, None);
    }

    #[test]
    fn test_2d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestHstack::test_2D_array (line 168)
        crate::specify_test!("test_2d_array");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = array([[1], [2]]); b = array([[1], [2]])
        // res = hstack([a, b])
        // desired = array([[1, 1], [2, 2]])
        let a = rt::tensor_from_nested!([[1], [2]], &device);
        let b = rt::tensor_from_nested!([[1], [2]], &device);
        let res = rt::hstack([&a, &b]);
        let desired = rt::tensor_from_nested!([[1, 1], [2, 2]], &device);
        assert_equal(&res, &desired, None);
    }

    #[test]
    fn test_0d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestHstack::test_0D_array (line 154)
        crate::specify_test!("test_0d_array");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = array(1); b = array(2); res = hstack([a, b]); desired = array([1, 2])
        // atleast_1d promotes 0-D -> (1,); 1-D inputs concatenate along axis 0 -> (2,).
        let a: Tensor<i32, _> = rt::asarray((1, &device));
        let b: Tensor<i32, _> = rt::asarray((2, &device));
        let res = rt::hstack([&a, &b]);
        let desired = rt::tensor_from_nested!([1, 2], &device);
        assert_equal(&res, &desired, None);
    }
}
