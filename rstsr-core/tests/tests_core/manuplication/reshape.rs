use crate::tests_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

mod numpy_reshape {
    use super::*;
    static FUNC: &str = "numpy_reshape";

    #[test]
    fn multiarray_methods_reshape() {
        // NumPy v2.4.2, _core/tests/test_multiarray.py, TestMethods::test_reshape
        crate::specify_test!("multiarray_methods_reshape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        let arr = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], &device);

        let tgt = rt::tensor_from_nested!([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], &device);
        assert_equal(arr.reshape([2, 6]), &tgt, None);

        let tgt = rt::tensor_from_nested!([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], &device);
        assert_equal(arr.reshape([3, 4]), &tgt, None);

        let tgt = rt::tensor_from_nested!([[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]], &device);
        let mut arr_col = arr.to_owned();
        arr_col.device_mut().set_default_order(ColMajor);
        let mut arr_col = arr_col.into_shape([3, 4]).into_owned();
        arr_col.device_mut().set_default_order(RowMajor);
        assert_equal(arr_col, &tgt, None);

        let tgt = rt::tensor_from_nested!([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]], &device);
        assert_equal(arr.t().reshape([3, 4]), &tgt, None);
    }
}

#[cfg(test)]
mod docs_reshape {
    use super::*;
    static FUNC: &str = "docs_reshape";

    #[test]
    fn quick_start() {
        crate::specify_test!("quick_start");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device));
        let a_reshaped = a.reshape([2, 3]);
        let a_expected = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        assert!(rt::allclose(&a_reshaped, &a_expected, None));

        // in this case, unspecified axes length is inferred as 6 / 3 = 2
        let a_reshaped = a.reshape([3, -1]);
        let a_expected = rt::tensor_from_nested!([[0, 1], [2, 3], [4, 5]], &device);
        assert!(rt::allclose(&a_reshaped, &a_expected, None));
    }

    #[test]
    fn elaborated_diff_row_col() {
        crate::specify_test!("elaborated_diff_row_col");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        let b = a.reshape([3, 2]);
        let a_vec = a.iter().collect::<Vec<_>>();
        let b_vec = b.iter().collect::<Vec<_>>();
        assert_eq!(a_vec, b_vec); // iterated sequence is the same

        let mut device = DeviceCpu::default();
        device.set_default_order(ColMajor);
        let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        let b = a.reshape([3, 2]);
        let a_c_vec = a.iter().collect::<Vec<_>>();
        let b_c_vec = b.iter().collect::<Vec<_>>();
        assert_eq!(a_c_vec, b_c_vec); // iterated sequence is the same
        assert_ne!(a_c_vec, a_vec); // iterated sequence is different from row-major

        // Row-major reshape
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);
        // a: [[0, 1, 2], [3, 4, 5]]
        // b: [[0, 1], [2, 3], [4, 5]]
        // iterated sequence: [0, 1, 2, 3, 4, 5]
        let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        let b = a.reshape([3, 2]);
        let b_expected = rt::tensor_from_nested!([[0, 1], [2, 3], [4, 5]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
        let a_vec = a.iter().cloned().collect::<Vec<_>>();
        let b_vec = b.iter().cloned().collect::<Vec<_>>();
        assert_eq!(a_vec, b_vec); // iterated sequence is the same
        assert_eq!(a_vec, vec![0, 1, 2, 3, 4, 5]);

        // Column-major reshape
        let mut device = DeviceCpu::default();
        device.set_default_order(ColMajor);
        // a: [[0, 1, 2], [3, 4, 5]]
        // b: [[0, 4], [3, 2], [1, 5]]
        // iterated sequence: [0, 3, 1, 4, 2, 5]
        let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        let b = a.reshape([3, 2]);
        let b_expected = rt::tensor_from_nested!([[0, 4], [3, 2], [1, 5]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
        let a_vec = a.iter().cloned().collect::<Vec<_>>();
        let b_vec = b.iter().cloned().collect::<Vec<_>>();
        assert_eq!(a_vec, b_vec); // iterated sequence is the same
        assert_eq!(a_vec, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn elaborated_clone_occasion() {
        crate::specify_test!("elaborated_clone_occasion");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // some strided tensor
        // shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
        // contiguous situation: (4, [6, 9]), or say the last two dimensions are contiguous
        let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
        assert_eq!(a.shape(), &[4, 6, 9]);
        assert_eq!(a.stride(), &[72, 9, 1]);
        assert!(!a.c_contig());

        // reshape that does not require clone (outputs tensor view)

        // split a single dimension into multiple dimensions
        assert!(!a.reshape([2, 2, 6, 9]).is_owned()); // (4, 6, 9) -> ([2, 2], 6, 9)
        assert!(!a.reshape([4, 3, 2, 9]).is_owned()); // (4, 6, 9) -> (4, [3, 2], 9)
        assert!(!a.reshape([4, 2, 3, 3, 3]).is_owned()); // (4, 6, 9) -> (4, [2, 3], [3, 3])

        // merge contiguous dimensions into a single dimension
        assert!(!a.reshape([4, 54]).is_owned()); // (4, 6, 9) -> (4, 6 * 9)

        // merge contiguous dimensions and then split
        assert!(!a.reshape([4, 3, 6, 3]).is_owned()); // (4, [6, 9]) -> (4, [3, 6, 3])

        // reshape that requires clone (outputs owned tensor)

        // merge non-contiguous dimensions
        assert!(a.reshape([24, 9]).is_owned()); // (4, 6, 9) -> (4 * 6, 9)
        assert!(a.reshape(-1).is_owned()); // (4, 6, 9) -> (4 * 6 * 9)
        assert!(a.reshape([12, 2, 9]).is_owned()); // (4, 6, 9) -> (4 * [3, 2], 9)
    }

    #[test]
    fn into_shape() {
        crate::specify_test!("into_shape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        println!("a: {:?}", a);

        // shape: (4, 6, 9), stride: (-54, 9, 1), not c-contiguous
        // contiguous situation: (4, [6, 9]); the first dimension is reversed
        let a = rt::arange((216, &device)).into_shape([4, 6, 9]).into_flip(0);
        let a_ptr = a.raw().as_ptr();
        let b = a.into_shape([4, 54]);
        let b_ptr = b.raw().as_ptr();
        assert_eq!(a_ptr, b_ptr); // contiguous dims merged, no data clone happened

        // shape: (4, 6, 9), stride: (-54, 9, 1), not c-contiguous
        // contiguous situation: (4, [6, 9]); the first dimension is reversed
        let a = rt::arange((216, &device)).into_shape([4, 6, 9]).into_flip(0);
        let a_ptr = a.raw().as_ptr();
        let b = a.into_shape([24, 9]);
        let b_ptr = b.raw().as_ptr();
        assert_ne!(a_ptr, b_ptr); // layout not compatible, data clone happened

        // shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
        // contiguous situation: (4, [6, 9]), or say the last two dimensions are contiguous
        let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
        let a_ptr = a.raw().as_ptr();
        let b = a.into_shape([4, 54]);
        let b_ptr = b.raw().as_ptr();
        assert_ne!(a_ptr, b_ptr); // layout-compatible, but input tensor is not compact (216 < 288)
    }
}
