#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_broadcast_to {
    use super::*;
    static FUNC: &str = "numpy_broadcast_to";

    #[test]
    fn test_broadcast_to_succeeds() {
        // NumPy v2.5.2, lib/tests/test_stride_tricks.py, test_broadcast_to_succeeds (line 242)
        crate::specify_test!("test_broadcast_to_succeeds");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // [np.array(0), (0,), np.array(0)]
        let input_array: Tensor<i32, _> = rt::asarray((0, &device));
        let result = rt::broadcast_to(&input_array, vec![0]);
        assert_eq!(result.shape(), &[0]);

        // [np.array(0), (1,), np.zeros(1)]
        let input_array: Tensor<i32, _> = rt::asarray((0, &device));
        let result = rt::broadcast_to(&input_array, vec![1]);
        let expected: Tensor<f64, _> = rt::zeros(([1], &device));
        assert_eq!(result.shape(), expected.shape());

        // [np.array(0), (3,), np.zeros(3)]
        let input_array: Tensor<i32, _> = rt::asarray((0, &device));
        let result = rt::broadcast_to(&input_array, vec![3]);
        assert_eq!(result.shape(), &[3]);

        // [np.ones(1), (1,), np.ones(1)]
        let input_array: Tensor<f64, _> = rt::ones(([1], &device));
        let result = rt::broadcast_to(&input_array, vec![1]);
        let expected: Tensor<f64, _> = rt::ones(([1], &device));
        assert_equal(&result, &expected, None);

        // [np.ones(1), (2,), np.ones(2)]
        let input_array: Tensor<f64, _> = rt::ones(([1], &device));
        let result = rt::broadcast_to(&input_array, vec![2]);
        let expected: Tensor<f64, _> = rt::ones(([2], &device));
        assert_equal(&result, &expected, None);

        // [np.ones(1), (1, 2, 3), np.ones((1, 2, 3))]
        let input_array: Tensor<f64, _> = rt::ones(([1], &device));
        let result = rt::broadcast_to(&input_array, vec![1, 2, 3]);
        let expected: Tensor<f64, _> = rt::ones(([1, 2, 3], &device));
        assert_equal(&result, &expected, None);

        // [np.arange(3), (3,), np.arange(3)]
        let input_array: Tensor<i32, _> = rt::arange((3, &device));
        let result = rt::broadcast_to(&input_array, vec![3]);
        let expected: Tensor<i32, _> = rt::arange((3, &device));
        assert_equal(&result, &expected, None);

        // [np.arange(3), (1, 3), np.arange(3).reshape(1, -1)]
        let input_array: Tensor<i32, _> = rt::arange((3, &device));
        let result = rt::broadcast_to(&input_array, vec![1, 3]);
        let expected: Tensor<i32, _> = rt::arange((3, &device)).into_shape([1, 3]);
        assert_equal(&result, &expected, None);

        // [np.arange(3), (2, 3), np.array([[0, 1, 2], [0, 1, 2]])]
        let input_array: Tensor<i32, _> = rt::arange((3, &device));
        let result = rt::broadcast_to(&input_array, vec![2, 3]);
        let expected = rt::tensor_from_nested!([[0, 1, 2], [0, 1, 2]], &device);
        assert_equal(&result, &expected, None);

        // [np.ones(0), 0, np.ones(0)] - size-0 (0,) array to integer shape 0
        let input_array: Tensor<f64, _> = rt::ones(([0], &device));
        let result = rt::broadcast_to(&input_array, vec![0]);
        assert_eq!(result.shape(), &[0]);

        // [np.ones(1), 1, np.ones(1)] - shape as integer, not tuple
        let input_array: Tensor<f64, _> = rt::ones(([1], &device));
        let result = rt::broadcast_to(&input_array, vec![1]);
        let expected: Tensor<f64, _> = rt::ones(([1], &device));
        assert_equal(&result, &expected, None);

        // [np.ones(1), 2, np.ones(2)] - shape as integer, not tuple
        let input_array: Tensor<f64, _> = rt::ones(([1], &device));
        let result = rt::broadcast_to(&input_array, vec![2]);
        let expected: Tensor<f64, _> = rt::ones(([2], &device));
        assert_equal(&result, &expected, None);

        // [np.ones(1), (0,), np.ones(0)]
        let input_array: Tensor<f64, _> = rt::ones(([1], &device));
        let result = rt::broadcast_to(&input_array, vec![0]);
        assert_eq!(result.shape(), &[0]);

        // [np.ones((1, 2)), (0, 2), np.ones((0, 2))]
        let input_array: Tensor<f64, _> = rt::ones(([1, 2], &device));
        let result = rt::broadcast_to(&input_array, vec![0, 2]);
        assert_eq!(result.shape(), &[0, 2]);

        // [np.ones((2, 1)), (2, 0), np.ones((2, 0))]
        let input_array: Tensor<f64, _> = rt::ones(([2, 1], &device));
        let result = rt::broadcast_to(&input_array, vec![2, 0]);
        assert_eq!(result.shape(), &[2, 0]);
    }

    #[test]
    fn test_broadcast_to_raises() {
        // NumPy v2.5.2, lib/tests/test_stride_tricks.py, test_broadcast_to_raises (line 268)
        crate::specify_test!("test_broadcast_to_raises");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // [(0,), ()]
        let arr: Tensor<f64, _> = rt::zeros(([0], &device));
        assert!(rt::broadcast_to_f(&arr, vec![]).is_err());

        // [(1,), ()]
        let arr: Tensor<f64, _> = rt::zeros(([1], &device));
        assert!(rt::broadcast_to_f(&arr, vec![]).is_err());

        // [(3,), ()]
        let arr: Tensor<f64, _> = rt::zeros(([3], &device));
        assert!(rt::broadcast_to_f(&arr, vec![]).is_err());

        // [(3,), (1,)]
        let arr: Tensor<f64, _> = rt::zeros(([3], &device));
        assert!(rt::broadcast_to_f(&arr, vec![1]).is_err());

        // [(3,), (2,)]
        let arr: Tensor<f64, _> = rt::zeros(([3], &device));
        assert!(rt::broadcast_to_f(&arr, vec![2]).is_err());

        // [(3,), (4,)]
        let arr: Tensor<f64, _> = rt::zeros(([3], &device));
        assert!(rt::broadcast_to_f(&arr, vec![4]).is_err());

        // [(1, 2), (2, 1)]
        let arr: Tensor<f64, _> = rt::zeros(([1, 2], &device));
        assert!(rt::broadcast_to_f(&arr, vec![2, 1]).is_err());

        // [(1, 1), (1,)]
        let arr: Tensor<f64, _> = rt::zeros(([1, 1], &device));
        assert!(rt::broadcast_to_f(&arr, vec![1]).is_err());

        // Note: RSTSR does not support negative shape values, skipping those cases
        // [(1,), -1]
        // [(1,), (-1,)]
        // [(1, 2), (-1, 2)]
    }
}

#[cfg(test)]
mod numpy_broadcast_arrays {
    use super::*;
    static FUNC: &str = "numpy_broadcast_arrays";

    #[test]
    fn test_broadcast_arrays_basic() {
        // NumPy v2.5.2, lib/tests/test_stride_tricks.py, test_broadcast_shape (line 287)
        crate::specify_test!("test_broadcast_arrays_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Test broadcasting multiple arrays
        // np.ones((1, 1)) and np.ones((3, 4)) -> (3, 4)
        let a: Tensor<f64, _> = rt::ones(([1, 1], &device));
        let b: Tensor<f64, _> = rt::ones(([3, 4], &device));
        let result = rt::broadcast_arrays(vec![a.view(), b.view()]);
        assert_eq!(result[0].shape(), &[3, 4]);
        assert_eq!(result[1].shape(), &[3, 4]);

        // Test with scalar
        // np.ones((1, 2)) * 32 times -> (1, 2)
        let a: Tensor<f64, _> = rt::ones(([1, 2], &device));
        let views: Vec<TensorView<f64, _>> = (0..32).map(|_| a.view()).collect();
        let result = rt::broadcast_arrays(views);
        for r in &result {
            assert_eq!(r.shape(), &[1, 2]);
        }
    }

    #[test]
    fn test_same() {
        // NumPy v2.5.2, lib/tests/test_stride_tricks.py, test_same (line 63)
        crate::specify_test!("test_same");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.arange(10)
        // y = np.arange(10)
        // bx, by = broadcast_arrays(x, y)
        // assert_array_equal(x, bx)
        // assert_array_equal(y, by)
        let x = rt::arange((10, &device));
        let result = rt::broadcast_arrays(vec![x.view(), x.view()]);
        assert_equal(&result[0], &x, None);
        assert_equal(&result[1], &x, None);
    }

    #[test]
    fn test_one_off() {
        // NumPy v2.5.2, lib/tests/test_stride_tricks.py, test_one_off (line 81)
        crate::specify_test!("test_one_off");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.array([[1, 2, 3]])
        // y = np.array([[1], [2], [3]])
        // bx, by = broadcast_arrays(x, y)
        // bx0 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        // by0 = bx0.T
        // assert_array_equal(bx0, bx)
        // assert_array_equal(by0, by)
        let x = rt::tensor_from_nested!([[1, 2, 3]], &device);
        let y = rt::tensor_from_nested!([[1], [2], [3]], &device);
        let result = rt::broadcast_arrays(vec![x.view(), y.view()]);
        assert_eq!(result[0].shape(), &[3, 3]);
        assert_eq!(result[1].shape(), &[3, 3]);
        let bx0 = rt::tensor_from_nested!([[1, 2, 3], [1, 2, 3], [1, 2, 3]], &device);
        let by0 = rt::tensor_from_nested!([[1, 1, 1], [2, 2, 2], [3, 3, 3]], &device);
        assert_equal(&result[0], &bx0, None);
        assert_equal(&result[1], &by0, None);
    }

    #[test]
    fn test_incompatible_shapes_raise() {
        // NumPy v2.5.2, lib/tests/test_stride_tricks.py, test_incompatible_shapes_raise_valueerror (line
        // 175)
        crate::specify_test!("test_incompatible_shapes_raise");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // [(3,), (4,)]
        let a: Tensor<f64, _> = rt::zeros(([3], &device));
        let b: Tensor<f64, _> = rt::zeros(([4], &device));
        assert!(rt::broadcast_arrays_f(vec![a.view(), b.view()]).is_err());

        // [(2, 3), (2,)]
        let a: Tensor<f64, _> = rt::zeros(([2, 3], &device));
        let b: Tensor<f64, _> = rt::zeros(([2], &device));
        assert!(rt::broadcast_arrays_f(vec![a.view(), b.view()]).is_err());

        // [(3,), (3,), (4,)]
        let a: Tensor<f64, _> = rt::zeros(([3], &device));
        let b: Tensor<f64, _> = rt::zeros(([3], &device));
        let c: Tensor<f64, _> = rt::zeros(([4], &device));
        assert!(rt::broadcast_arrays_f(vec![a.view(), b.view(), c.view()]).is_err());

        // [(1, 3, 4), (2, 3, 3)]
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 4], &device));
        let b: Tensor<f64, _> = rt::zeros(([2, 3, 3], &device));
        assert!(rt::broadcast_arrays_f(vec![a.view(), b.view()]).is_err());
    }
}
