#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

// Parity tests for `stack` (join a sequence of arrays along a new axis).
//
// Source: NumPy v2.5.2, `_core/tests/test_shape_base.py::test_stack` (module-level fn).
// rstsr `stack` matches NumPy, including 0-D inputs (they stack into a 1-D array).
//
// Not ported (N/A, Rust typed API): non-iterable input (TypeError), generator input
// (TypeError), casting/dtype test + type_error (`dtype=`/`casting=` kwargs), `out=`
// case (no `out=` parameter).

#[cfg(test)]
mod numpy_stack {
    use super::*;
    static FUNC: &str = "numpy_stack";

    #[test]
    fn test_1d_input() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, test_stack (line 463) - 1d input
        crate::specify_test!("test_1d_input");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.array([1, 2, 3]); b = np.array([4, 5, 6])
        // r1 = array([[1, 2, 3], [4, 5, 6]])
        // assert_array_equal(np.stack((a, b)), r1)
        // assert_array_equal(np.stack((a, b), axis=1), r1.T)
        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let b = rt::tensor_from_nested!([4, 5, 6], &device);
        let r1 = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        assert_equal(rt::stack([&a, &b]), &r1, None);
        let r1t = r1.t();
        assert_equal(rt::stack((vec![&a, &b], 1)), &r1t, None);
    }

    #[test]
    fn test_shapes() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, test_stack (line 463) - shapes
        crate::specify_test!("test_shapes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // all shapes for 1d input (10 arrays of shape (3,))
        // axes = [0, 1, -1, -2]; expected_shapes = [(10, 3), (3, 10), (3, 10), (10, 3)]
        let arrays1: Vec<Tensor<i64, _>> = (0..10).map(|_| rt::arange((3, &device))).collect();
        for (axis, expected_shape) in [(0isize, vec![10, 3]), (1, vec![3, 10]), (-1, vec![3, 10]), (-2, vec![10, 3])] {
            let r = rt::stack((&arrays1, axis));
            assert_eq!(r.shape(), expected_shape.as_slice(), "1d axis={axis}");
        }
        // assert_raises_regex(AxisError, 'out of bounds', stack, arrays, axis=2)
        // assert_raises_regex(AxisError, 'out of bounds', stack, arrays, axis=-3)
        assert!(rt::stack_f((&arrays1, 2isize)).is_err());
        assert!(rt::stack_f((&arrays1, -3isize)).is_err());

        // all shapes for 2d input (10 arrays of shape (3, 4))
        // axes = [0, 1, 2, -1, -2, -3]
        // expected_shapes = [(10, 3, 4), (3, 10, 4), (3, 4, 10), (3, 4, 10), (3, 10, 4), (10, 3, 4)]
        let arrays2: Vec<Tensor<i64, _>> = (0..10).map(|_| rt::arange((12, &device)).into_shape([3, 4])).collect();
        for (axis, expected_shape) in [
            (0isize, vec![10, 3, 4]),
            (1, vec![3, 10, 4]),
            (2, vec![3, 4, 10]),
            (-1, vec![3, 4, 10]),
            (-2, vec![3, 10, 4]),
            (-3, vec![10, 3, 4]),
        ] {
            let r = rt::stack((&arrays2, axis));
            assert_eq!(r.shape(), expected_shape.as_slice(), "2d axis={axis}");
        }
    }

    #[test]
    fn test_empty_arrays() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, test_stack (line 463) - empty arrays
        crate::specify_test!("test_empty_arrays");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // empty arrays
        // assert_(stack([[], [], []]).shape == (3, 0))
        // assert_(stack([[], [], []], axis=1).shape == (0, 3))
        let e: Tensor<i64, _> = rt::zeros(([0], &device));
        let r = rt::stack([&e, &e, &e]);
        assert_eq!(r.shape(), &[3, 0]);
        let r = rt::stack((vec![&e, &e, &e], 1isize));
        assert_eq!(r.shape(), &[0, 3]);
    }

    #[test]
    fn test_edge_cases() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, test_stack (line 463) - edge cases
        crate::specify_test!("test_edge_cases");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // edge cases
        // assert_raises_regex(ValueError, 'need at least one array', stack, [])
        let empty: Vec<Tensor<i64, DeviceType>> = vec![];
        assert!(rt::stack_f((&empty, 0isize)).is_err());

        // assert_raises_regex(ValueError, 'must have the same shape', stack, [1, np.arange(3)])
        // rstsr cannot mix a scalar and an array in one vec (homogeneous typed); the
        // equivalent - two arrays of different shapes - exercises the same check.
        let a: Tensor<i64, _> = rt::arange((3, &device));
        let b: Tensor<i64, _> = rt::arange((2, &device));
        assert!(rt::stack_f((vec![&a, &b], 0isize)).is_err());
        // ... also along axis=1
        assert!(rt::stack_f((vec![&a, &b], 1isize)).is_err());

        // must have the same shape: 2-D vs 1-D
        let a2: Tensor<i64, _> = rt::zeros(([3, 3], &device));
        assert!(rt::stack_f((vec![&a2, &a], 1isize)).is_err());

        // different lengths (2 vs 3)
        let c: Tensor<i64, _> = rt::arange((2, &device));
        let d: Tensor<i64, _> = rt::arange((3, &device));
        assert!(rt::stack_f((vec![&c, &d], 0isize)).is_err());
    }

    #[test]
    fn test_0d_input() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, test_stack (line 463) - 0d input
        crate::specify_test!("test_0d_input");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // 0d input: stack([array(1), array(2), array(3)]) == [1, 2, 3]
        // Each 0-D is expanded with a new axis -> (1,); concat axis 0 -> (3,).
        let a: Tensor<i32, _> = rt::asarray((1, &device));
        let b: Tensor<i32, _> = rt::asarray((2, &device));
        let c: Tensor<i32, _> = rt::asarray((3, &device));
        let res = rt::stack([&a, &b, &c]);
        let desired = rt::tensor_from_nested!([1, 2, 3], &device);
        assert_equal(&res, &desired, None);
    }
}
