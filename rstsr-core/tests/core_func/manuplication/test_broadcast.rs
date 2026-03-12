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
        // NumPy v2.4.2, lib/tests/test_stride_tricks.py, test_broadcast_to_succeeds (line 242)
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
        // NumPy v2.4.2, lib/tests/test_stride_tricks.py, test_broadcast_to_raises (line 268)
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
        // NumPy v2.4.2, lib/tests/test_stride_tricks.py, test_broadcast_shape (line 287)
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
}

#[cfg(test)]
mod docs_broadcast {
    use super::*;
    static FUNC: &str = "docs_broadcast";

    #[test]
    #[rustfmt::skip]
    fn doc_broadcast_to_row_major() {
        crate::specify_test!("doc_broadcast_to_row_major");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2, 3], &device);

        // broadcast (3, ) -> (2, 3) in row-major:
        let result = a.to_broadcast(vec![2, 3]);
        println!("{result}");
        // [[ 1 2 3]
        //  [ 1 2 3]]
        let expected = rt::tensor_from_nested!(
            [[1, 2, 3],
             [1, 2, 3]],
            &device);
        assert!(rt::allclose!(&result, &expected));
    }

    #[test]
    #[rustfmt::skip]
    fn doc_broadcast_to_col_major() {
        crate::specify_test!("doc_broadcast_to_col_major");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        // in col-major, broadcast (3, ) -> (2, 3) will fail:
        let result = a.to_broadcast_f(vec![2, 3]);
        assert!(result.is_err());

        // broadcast (3, ) -> (3, 2) in col-major:
        let result = a.to_broadcast(vec![3, 2]);
        println!("{result}");
        // [[ 1 1]
        //  [ 2 2]
        //  [ 3 3]]
        let expected = rt::tensor_from_nested!(
            [[1, 1],
             [2, 2],
             [3, 3]],
            &device);
        assert!(rt::allclose!(&result, &expected));
    }

    #[test]
    fn doc_broadcast_to_elaborated_row_major() {
        crate::specify_test!("doc_broadcast_to_elaborated_row_major");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // A      (4d tensor):  8 x 1 x 6 x 1
        // B      (3d tensor):      7 x 1 x 5
        // ----------------------------------
        // Result (4d tensor):  8 x 7 x 6 x 5
        let a = rt::arange((48, &device)).into_shape([8, 1, 6, 1]);
        let b = rt::arange((35, &device)).into_shape([7, 1, 5]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[8, 7, 6, 5]);

        // A      (2d tensor):  5 x 4
        // B      (1d tensor):      1
        // --------------------------
        // Result (2d tensor):  5 x 4
        let a = rt::arange((20, &device)).into_shape([5, 4]);
        let b = rt::arange((1, &device)).into_shape([1]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[5, 4]);

        // A      (2d tensor):  5 x 4
        // B      (1d tensor):      4
        // --------------------------
        // Result (2d tensor):  5 x 4
        let a = rt::arange((20, &device)).into_shape([5, 4]);
        let b = rt::arange((4, &device)).into_shape([4]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[5, 4]);

        // A      (3d tensor):  15 x 3 x 5
        // B      (3d tensor):  15 x 1 x 5
        // -------------------------------
        // Result (3d tensor):  15 x 3 x 5
        let a = rt::arange((225, &device)).into_shape([15, 3, 5]);
        let b = rt::arange((75, &device)).into_shape([15, 1, 5]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[15, 3, 5]);

        // A      (3d tensor):  15 x 3 x 5
        // B      (2d tensor):       3 x 5
        // -------------------------------
        // Result (3d tensor):  15 x 3 x 5
        let a = rt::arange((225, &device)).into_shape([15, 3, 5]);
        let b = rt::arange((15, &device)).into_shape([3, 5]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[15, 3, 5]);

        // A      (3d tensor):  15 x 3 x 5
        // B      (2d tensor):       3 x 1
        // -------------------------------
        // Result (3d tensor):  15 x 3 x 5
        let a = rt::arange((225, &device)).into_shape([15, 3, 5]);
        let b = rt::arange((3, &device)).into_shape([3, 1]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[15, 3, 5]);
    }

    #[test]
    fn doc_broadcast_to_elaborated_col_major() {
        crate::specify_test!("doc_broadcast_to_elaborated_col_major");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        // A      (4d tensor):  1 x 6 x 1 x 8
        // B      (3d tensor):  5 x 1 x 7
        // ----------------------------------
        // Result (4d tensor):  5 x 6 x 7 x 8
        let a = rt::arange((48, &device)).into_shape([1, 6, 1, 8]);
        let b = rt::arange((35, &device)).into_shape([5, 1, 7]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[5, 6, 7, 8]);

        // A      (2d tensor):  4 x 5
        // B      (1d tensor):  1
        // --------------------------
        // Result (2d tensor):  4 x 5
        let a = rt::arange((20, &device)).into_shape([4, 5]);
        let b = rt::arange((1, &device)).into_shape([1]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[4, 5]);

        // A      (2d tensor):  4 x 5
        // B      (1d tensor):  4
        // --------------------------
        // Result (2d tensor):  4 x 5
        let a = rt::arange((20, &device)).into_shape([4, 5]);
        let b = rt::arange((4, &device)).into_shape([4]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[4, 5]);

        // A      (3d tensor):  5 x 3 x 15
        // B      (3d tensor):  5 x 1 x 15
        // -------------------------------
        // Result (3d tensor):  5 x 3 x 15
        let a = rt::arange((225, &device)).into_shape([5, 3, 15]);
        let b = rt::arange((75, &device)).into_shape([5, 1, 15]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[5, 3, 15]);

        // A      (3d tensor):  5 x 3 x 15
        // B      (2d tensor):  5 x 3
        // -------------------------------
        // Result (3d tensor):  5 x 3 x 15
        let a = rt::arange((225, &device)).into_shape([5, 3, 15]);
        let b = rt::arange((15, &device)).into_shape([5, 3]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[5, 3, 15]);

        // A      (3d tensor):  5 x 3 x 15
        // B      (2d tensor):  1 x 3
        // -------------------------------
        // Result (3d tensor):  5 x 3 x 15
        let a = rt::arange((225, &device)).into_shape([5, 3, 15]);
        let b = rt::arange((3, &device)).into_shape([1, 3]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[5, 3, 15]);
    }

    #[test]
    #[rustfmt::skip]
    fn doc_broadcast_arrays_row_major() {
        crate::specify_test!("doc_broadcast_arrays_row_major");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::asarray((vec![1, 2, 3], &device)).into_shape([3]);
        println!("{a}");
        // [ 1 2 3]
        let b = rt::asarray((vec![4, 5], &device)).into_shape([2, 1]);
        println!("{b}");
        // [[ 4]
        //  [ 5]]

        let result = rt::broadcast_arrays(vec![a, b]);
        println!("broadcasted a:\n{:}", result[0]);
        // [[ 1 2 3]
        //  [ 1 2 3]]
        println!("broadcasted b:\n{:}", result[1]);
        // [[ 4 4 4]
        //  [ 5 5 5]]
        let expected_a = rt::tensor_from_nested!(
            [[1, 2, 3],
             [1, 2, 3]],
            &device);
        let expected_b = rt::tensor_from_nested!(
            [[4, 4, 4],
             [5, 5, 5]],
            &device);
        assert!(rt::allclose!(&result[0], &expected_a));
        assert!(rt::allclose!(&result[1], &expected_b));
    }

    #[test]
    #[rustfmt::skip]
    fn doc_broadcast_arrays_col_major() {
        crate::specify_test!("doc_broadcast_arrays_col_major");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let a = rt::asarray((vec![1, 2, 3], &device)).into_shape([1, 3]);
        let b = rt::asarray((vec![4, 5], &device)).into_shape([2, 1]);

        let result = rt::broadcast_arrays(vec![a, b]);
        let expected_a = rt::tensor_from_nested!(
            [[1, 2, 3],
             [1, 2, 3]],
            &device);
        let expected_b = rt::tensor_from_nested!(
            [[4, 4, 4],
             [5, 5, 5]],
            &device);
        assert!(rt::allclose!(&result[0], &expected_a));
        assert!(rt::allclose!(&result[1], &expected_b));
    }
}
