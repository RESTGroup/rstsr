use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

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
