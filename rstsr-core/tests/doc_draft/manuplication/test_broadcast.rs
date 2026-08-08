#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_broadcast {
    use super::*;
    static FUNC: &str = "doc_broadcast";

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
