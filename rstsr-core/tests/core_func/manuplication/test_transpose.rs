use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_transpose {
    use super::*;
    static FUNC: &str = "numpy_transpose";

    #[test]
    fn test_multiarray() {
        // NumPy v2.4.2, _core/tests/test_multiarray.py, TestMethods::test_transpose (line 2221)
        crate::specify_test!("test_multiarray");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.array([[1, 2], [3, 4]])
        // assert_equal(a.transpose(), [[1, 3], [2, 4]])
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let expected = rt::tensor_from_nested!([[1, 3], [2, 4]], &device);
        assert_equal(rt::transpose(&a, None), &expected, None);
        assert_equal(a.t(), &expected, None);

        // assert_raises(ValueError, lambda: a.transpose(0))
        assert!(rt::transpose_f(&a, [0]).is_err());

        // assert_raises(ValueError, lambda: a.transpose(0, 0))
        assert!(rt::transpose_f(&a, [0, 0]).is_err());

        // assert_raises(ValueError, lambda: a.transpose(0, 1, 2))
        assert!(rt::transpose_f(&a, [0, 1, 2]).is_err());
    }

    #[test]
    fn test_numeric() {
        // NumPy v2.4.2, _core/tests/test_numeric.py, TestNonarrayArgs::test_transpose (line 354)
        crate::specify_test!("test_numeric");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr = [[1, 2], [3, 4], [5, 6]]
        // tgt = [[1, 3, 5], [2, 4, 6]]
        // assert_equal(np.transpose(arr, (1, 0)), tgt)
        let arr = rt::tensor_from_nested!([[1, 2], [3, 4], [5, 6]], &device);
        let tgt = rt::tensor_from_nested!([[1, 3, 5], [2, 4, 6]], &device);
        assert_equal(rt::transpose(&arr, [1, 0]), &tgt, None);

        // assert_equal(np.transpose(arr, (-1, -2)), tgt)
        assert_equal(rt::transpose(&arr, [-1, -2]), &tgt, None);
    }

    #[test]
    fn test_regression_arr_transpose() {
        // NumPy v2.4.2, _core/tests/test_regression.py, TestRegression::test_arr_transpose (line 778)
        // Ticket #516 - High dimensional transpose
        crate::specify_test!("test_regression_arr_transpose");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.random.rand(*(2,) * 16)
        // x.transpose(list(range(16)))  # Should succeed
        let shape: [usize; 16] = [2; 16];
        let x: Tensor<usize, _> = rt::arange((65536, &device)).into_shape(shape);
        let axes: [isize; 16] = core::array::from_fn(|i| i as isize);
        let _transposed = rt::transpose(&x, axes); // Should succeed
    }
}

mod doc_transpose {
    use super::*;
    static FUNC: &str = "doc_transpose";

    #[test]
    fn test_doc() {
        // Test that the documentation examples for transpose work correctly.
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // 2-D array
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let result = a.transpose(None);
        println!("{result}");
        // [[ 1 3]
        //  [ 2 4]]
        let target = rt::tensor_from_nested!([[1, 3], [2, 4]], &device);
        assert!(rt::allclose(&result, &target, None));

        // 1-D array
        let a = rt::tensor_from_nested!([1, 2, 3, 4], &device);
        let result = a.transpose(None);
        println!("{result}");
        // [ 1 2 3 4]
        let target = rt::tensor_from_nested!([1, 2, 3, 4], &device);
        assert!(rt::allclose(&result, &target, None));

        // 3-D with axes argument
        let a: Tensor<i32, _> = rt::ones(([1, 2, 3], &device));
        let result = a.transpose(None);
        println!("{:?}", result.shape());
        // [3, 2, 1]
        assert_eq!(result.shape(), &[3, 2, 1]);
        let result = a.transpose([1, 0, 2]);
        println!("{:?}", result.shape());
        // [2, 1, 3]
        assert_eq!(result.shape(), &[2, 1, 3]);

        // 4-D full reverse order
        let a: Tensor<i32, _> = rt::ones(([2, 3, 4, 5], &device));
        let result = a.transpose(None);
        println!("{:?}", result.shape());
        // [5, 4, 3, 2]
        assert_eq!(result.shape(), &[5, 4, 3, 2]);

        // negative axes
        let a: Tensor<i32, _> = rt::arange((3 * 4 * 5, &device)).into_shape([3, 4, 5]);
        let result = a.transpose([-1, 0, -2]);
        println!("{:?}", result.shape());
        // [5, 3, 4]
        assert_eq!(result.shape(), &[5, 3, 4]);
    }
}

#[cfg(test)]
mod numpy_swapaxes {
    use super::*;
    static FUNC: &str = "numpy_swapaxes";

    #[test]
    fn test_numeric() {
        // NumPy v2.4.2, _core/tests/test_numeric.py, TestNonarrayArgs::test_swapaxes (line 315)
        crate::specify_test!("test_numeric");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // tgt = [[[0, 4], [2, 6]], [[1, 5], [3, 7]]]
        // a = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        // out = np.swapaxes(a, 0, 2)
        // assert_equal(out, tgt)
        let a = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
        let tgt = rt::tensor_from_nested!([[[0, 4], [2, 6]], [[1, 5], [3, 7]]], &device);
        let out = rt::swapaxes(&a, 0, 2);
        assert_equal(&out, &tgt, None);
    }

    #[test]
    fn test_multiarray() {
        // NumPy v2.4.2, _core/tests/test_multiarray.py, TestMethods::test_swapaxes (line 3850)
        crate::specify_test!("test_multiarray");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4).copy()
        let a = rt::arange((24, &device)).into_shape([1, 2, 3, 4]);

        // check exceptions
        // assert_raises(AxisError, a.swapaxes, -5, 0)
        assert!(a.swapaxes_f(-5, 0).is_err());
        // assert_raises(AxisError, a.swapaxes, 4, 0)
        assert!(a.swapaxes_f(4, 0).is_err());
        // assert_raises(AxisError, a.swapaxes, 0, -5)
        assert!(a.swapaxes_f(0, -5).is_err());
        // assert_raises(AxisError, a.swapaxes, 0, 4)
        assert!(a.swapaxes_f(0, 4).is_err());

        // Test various axis combinations
        for i in -4..4 {
            for j in -4..4 {
                let c = a.swapaxes(i, j);
                // check shape
                let mut expected_shape: Vec<usize> = a.shape().to_vec();
                let i_usize = if i < 0 { (a.ndim() as isize + i) as usize } else { i as usize };
                let j_usize = if j < 0 { (a.ndim() as isize + j) as usize } else { j as usize };
                expected_shape.swap(i_usize, j_usize);
                assert_eq!(c.shape().to_vec(), expected_shape, "shape mismatch for swapaxes({}, {})", i, j);
            }
        }
    }
}

#[cfg(test)]
mod doc_swapaxes {
    use super::*;
    static FUNC: &str = "doc_swapaxes";

    #[test]
    fn test_doc() {
        // Test that the documentation examples for swapaxes work correctly.
        // Based on NumPy v2.4.2 swapaxes docstring examples.
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // 2-D array: swapping axes 0 and 1 is equivalent to transpose
        let x = rt::tensor_from_nested!([[1, 2, 3]], &device);
        let result = x.swapaxes(0, 1);
        println!("{result}");
        // [[ 1]
        //  [ 2]
        //  [ 3]]
        let target = rt::tensor_from_nested!([[1], [2], [3]], &device);
        assert!(rt::allclose(&result, &target, None));

        // 3-D array: swapping axes 0 and 2
        let x = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
        let result = x.swapaxes(0, 2);
        println!("{result}");
        // [[[ 0 4]
        //   [ 2 6]]
        //
        //  [[ 1 5]
        //   [ 3 7]]]
        let target = rt::tensor_from_nested!([[[0, 4], [2, 6]], [[1, 5], [3, 7]]], &device);
        assert!(rt::allclose(&result, &target, None));

        // Using negative indices to swap axes
        let x = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
        let result = x.swapaxes(-1, -3);
        println!("{:?}", result.shape());
        // [2, 2, 2]
        let result2 = x.swapaxes(2, 0);
        assert!(rt::allclose(&result, &result2, None));
    }
}

#[cfg(test)]
mod doc_reverse_axes {
    use super::*;
    static FUNC: &str = "doc_reverse_axes";

    #[test]
    fn test_doc() {
        // Test that the documentation examples for reverse_axes work correctly.
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // 2-D array: reverse_axes is equivalent to matrix transpose
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let result = a.reverse_axes();
        println!("{result}");
        // [[ 1  3]
        //  [ 2  4]]
        let target = rt::tensor_from_nested!([[1, 3], [2, 4]], &device);
        assert!(rt::allclose(&result, &target, None));

        // 1-D array: reverse_axes returns unchanged view
        let a = rt::tensor_from_nested!([1, 2, 3, 4], &device);
        let result = a.reverse_axes();
        println!("{result}");
        // [ 1  2  3  4]
        let target = rt::tensor_from_nested!([1, 2, 3, 4], &device);
        assert!(rt::allclose(&result, &target, None));

        // 3-D array: reverse_axes reverses all axis order
        let a = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
        println!("Original shape: {:?}", a.shape());
        // [2, 2, 2]
        let result = a.reverse_axes();
        println!("Reversed shape: {:?}", result.shape());
        // [2, 2, 2]
        // Note: For [2,2,2] shape, reverse doesn't change shape but changes axis order
        // Original axes [0, 1, 2], Reversed axes [2, 1, 0]
        let expected = rt::tensor_from_nested!([[[1, 5], [3, 7]], [[2, 6], [4, 8]]], &device);
        assert!(rt::allclose(&result, &expected, None));

        // 4-D array: reverse_axes shows clear shape change
        let a: Tensor<i32, _> = rt::ones(([2, 3, 4, 5], &device));
        let result = a.reverse_axes();
        println!("{:?}", result.shape());
        // [5, 4, 3, 2]
        assert_eq!(result.shape(), &[5, 4, 3, 2]);
    }
}
