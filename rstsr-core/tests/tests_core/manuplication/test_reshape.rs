use crate::tests_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_reshape {
    use super::*;
    static FUNC: &str = "numpy_reshape";

    #[test]
    fn multiarray() {
        // NumPy v2.4.2, _core/tests/test_multiarray.py, TestMethods::test_reshape (line 2167)
        crate::specify_test!("multiarray_reshape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        let arr = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], &device);

        // tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        // assert_equal(arr.reshape(2, 6), tgt)
        let tgt = rt::tensor_from_nested!([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], &device);
        assert_equal(arr.reshape([2, 6]), &tgt, None);

        // tgt = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        // assert_equal(arr.reshape(3, 4), tgt)
        let tgt = rt::tensor_from_nested!([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], &device);
        assert_equal(arr.reshape([3, 4]), &tgt, None);

        // tgt = [[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]]
        // assert_equal(arr.reshape((3, 4), order='F'), tgt)
        let tgt = rt::tensor_from_nested!([[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]], &device);
        assert_equal(arr.reshape_with_args([3, 4], ColMajor), &tgt, None);

        // tgt = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
        // assert_equal(arr.T.reshape((3, 4), order='C'), tgt)
        let tgt = rt::tensor_from_nested!([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]], &device);
        assert_equal(arr.t().reshape([3, 4]), &tgt, None);
    }

    #[test]
    fn regression() {
        // NumPy v2.4.2, _core/tests/test_regression.py, TestRegression::test_reshape*
        crate::specify_test!("regression_reshape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // CASE test_reshape_order (line 635)

        // a = np.arange(6).reshape(2, 3, order='F')
        // assert_equal(a, [[0, 2, 4], [1, 3, 5]])
        let a = rt::arange((6, &device)).into_shape_with_args([2, 3], ColMajor);
        let tgt = rt::tensor_from_nested!([[0, 2, 4], [1, 3, 5]], &device);
        assert_equal(&a, &tgt, None);

        // a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        // b = a[:, 1]
        // assert_equal(b.reshape(2, 2, order='F'), [[2, 6], [4, 8]])
        let a = rt::tensor_from_nested!([[1, 2], [3, 4], [5, 6], [7, 8]], &device);
        let b = a.i((.., 1));
        let tgt = rt::tensor_from_nested!([[2, 6], [4, 8]], &device);
        assert_equal(b.reshape_with_args([2, 2], ColMajor), &tgt, None);

        // CASE test_reshape_zero_strides (line 643)

        // a = np.ones(1)
        // a = as_strided(a, shape=(5,), strides=(0,))
        // assert_(a.reshape(5, 1).strides[0] == 0)
        let layout = unsafe { Layout::new_unchecked([5], [0], 0) };
        let a = rt::asarray((vec![1], layout, &device));
        assert!(a.reshape([5, 1]).stride()[0] == 0);

        // CASE test_reshape_zero_size (line 649)

        // a = np.ones((0, 2))
        // a.shape = (-1, 2)
        let a: Tensor<i32, _> = rt::ones(([0, 2], &device));
        let _a_reshaped = a.reshape([-1, 2]);

        // CASE test_reshape_trailing_ones_strides (line 654)

        // a = np.zeros(12, dtype=np.int32)[::2]  # not contiguous
        // strides_c = (16, 8, 8, 8)
        // strides_f = (8, 24, 48, 48)
        // assert_equal(a.reshape(3, 2, 1, 1).strides, strides_c)
        // assert_equal(a.reshape(3, 2, 1, 1, order='F').strides, strides_f)
        // assert_equal(np.array(0, dtype=np.int32).reshape(1, 1).strides, (4, 4))
        let a: Tensor<i32, _> = rt::zeros(([12], &device)).into_slice(slice!(None, None, 2));
        assert_eq!(a.reshape([3, 2, 1, 1]).stride(), &[4, 2, 2, 2]);
        assert_eq!(a.reshape_with_args([3, 2, 1, 1], ColMajor).stride(), &[2, 6, 12, 12]);

        // assert_equal(np.array(0, dtype=np.int32).reshape(1, 1).strides, (4, 4))
        let a: Tensor<i32, _> = rt::asarray((0, &device));
        assert_eq!(a.reshape([1, 1]).stride(), &[1, 1]);

        // CASE test_reshape_size_overflow (line 2278)
        // please note in this case, panic occurs on rust-side, not from RSTSR (i.e., not coverable)

        // a = np.ones(20)[::2]
        let a: Tensor<i32, _> = rt::ones(([20], &device)).into_slice(slice!(None, None, 2));
        // if IS_64BIT:
        //     # 64 bit. The following are the prime factors of 2**63 + 5,
        //     # plus a leading 2, so when multiplied together as int64,
        //     # the result overflows to a total size of 10.
        //     new_shape = (2, 13, 419, 691, 823, 2977518503)
        // else:
        //     # 32 bit. The following are the prime factors of 2**31 + 5,
        //     # plus a leading 2, so when multiplied together as int32,
        //     # the result overflows to a total size of 10.
        //     new_shape = (2, 7, 7, 43826197)
        let new_shape: Vec<usize> = if cfg!(target_pointer_width = "64") {
            vec![2, 13, 419, 691, 823, 2977518503]
        } else {
            vec![2, 7, 7, 43826197]
        };
        // assert_raises(ValueError, a.reshape, new_shape)
        let panics = std::panic::catch_unwind(|| a.reshape_f(new_shape));
        assert!(panics.is_err());
    }

    #[test]
    fn numeric() {
        // NumPy v2.4.2, _core/tests/test_numeric.py, TestNonarrayArgs::test_reshape*

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // CASE test_reshape (line 178)

        // arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        // tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        // assert_equal(np.reshape(arr, (2, 6)), tgt)
        let arr = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], &device);
        let tgt = rt::tensor_from_nested!([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], &device);
        assert_equal(arr.reshape([2, 6]), &tgt, None);

        // CASE test_reshape_shape_arg (line 183)

        // arr = np.arange(12)
        // shape = (3, 4)
        // expected = arr.reshape(shape)
        let arr = rt::arange((12, &device));
        let shape = [3, 4];
        let expected = rt::reshape(&arr, shape);
        // assert_equal(np.reshape(arr, shape), expected)
        // assert_equal(np.reshape(arr, shape, order="C"), expected)
        // assert_equal(np.reshape(arr, shape, "C"), expected)
        // assert_equal(np.reshape(arr, shape=shape), expected)
        // assert_equal(np.reshape(arr, shape=shape, order="C"), expected)
        assert_equal(arr.reshape(shape), &expected, None);
        assert_equal(arr.reshape_with_args(shape, RowMajor), &expected, None);
        assert_equal(arr.reshape_with_args(shape, ReshapeArgs { order: Some(RowMajor), copy: None }), &expected, None);

        // CASE test_reshape_copy_arg (line 201)

        // arr = np.arange(24).reshape(2, 3, 4)
        // arr_f_ord = np.array(arr, order="F")
        // shape = (12, 2)
        let arr = rt::arange((24, &device)).into_shape([2, 3, 4]);
        let arr_f_ord = rt::arange((24, &device)).into_layout([2, 3, 4].f());
        let shape = [12, 2];
        // assert np.shares_memory(np.reshape(arr, shape), arr)
        // assert np.shares_memory(np.reshape(arr, shape, order="C"), arr)
        // assert np.shares_memory(
        //     np.reshape(arr_f_ord, shape, order="F"), arr_f_ord)
        // assert np.shares_memory(np.reshape(arr, shape, copy=None), arr)
        // assert np.shares_memory(np.reshape(arr, shape, copy=False), arr)
        // assert np.shares_memory(arr.reshape(shape, copy=False), arr)
        // assert not np.shares_memory(np.reshape(arr, shape, copy=True), arr)
        // assert not np.shares_memory(
        //     np.reshape(arr, shape, order="C", copy=True), arr)
        // assert not np.shares_memory(
        //     np.reshape(arr, shape, order="F", copy=True), arr)
        // assert not np.shares_memory(
        //     np.reshape(arr, shape, order="F", copy=None), arr)
        assert!(core::ptr::eq(arr.reshape(shape).raw(), arr.raw()));
        assert!(core::ptr::eq(arr.reshape_with_args(shape, RowMajor).raw(), arr.raw()));
        assert!(core::ptr::eq(arr_f_ord.reshape_with_args(shape, ColMajor).raw(), arr_f_ord.raw()));
        assert!(core::ptr::eq(arr.reshape_with_args(shape, None).raw(), arr.raw()));
        assert!(core::ptr::eq(arr.reshape_with_args(shape, false).raw(), arr.raw()));
        assert!(!core::ptr::eq(arr.reshape_with_args(shape, true).raw(), arr.raw()));
        assert!(!core::ptr::eq(arr.reshape_with_args(shape, (RowMajor, true)).raw(), arr.raw()));
        assert!(!core::ptr::eq(arr.reshape_with_args(shape, (ColMajor, true)).raw(), arr.raw()));
        assert!(!core::ptr::eq(arr.reshape_with_args(shape, (ColMajor, None)).raw(), arr.raw()));
        // err_msg = "Unable to avoid creating a copy while reshaping."
        // with pytest.raises(ValueError, match=err_msg):
        //     np.reshape(arr, shape, order="F", copy=False)
        // with pytest.raises(ValueError, match=err_msg):
        //     np.reshape(arr_f_ord, shape, order="C", copy=False)
        assert!(arr.reshape_with_args_f(shape, (ColMajor, false)).is_err());
        assert!(arr_f_ord.reshape_with_args_f(shape, (RowMajor, false)).is_err());
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
    fn reshape_with_args() {
        crate::specify_test!("reshape_with_args");

        let mut device = TESTCFG.device.clone();

        // Row-major reshape
        // a: [[0, 1, 2], [3, 4, 5]]
        // b: [[0, 1], [2, 3], [4, 5]]
        // iterated sequence: [0, 1, 2, 3, 4, 5]
        let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        let a_row = rt::tensor_from_nested!([[0, 1], [2, 3], [4, 5]], &device);
        assert!(rt::allclose(a.reshape_with_args([3, 2], RowMajor), &a_row, None));

        // Column-major reshape
        // a: [[0, 1, 2], [3, 4, 5]]
        // b: [[0, 4], [3, 2], [1, 5]]
        // iterated sequence: [0, 3, 1, 4, 2, 5]
        let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        let a_col = rt::tensor_from_nested!([[0, 4], [3, 2], [1, 5]], &device);
        assert!(rt::allclose(a.reshape_with_args([3, 2], ColMajor), &a_col, None));

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
        assert!(a.reshape_with_args_f([2, 2, 6, 9], false).is_ok()); // (4, 6, 9) -> ([2, 2], 6, 9)
        assert!(a.reshape_with_args_f([4, 3, 2, 9], false).is_ok()); // (4, 6, 9) -> (4, [3, 2], 9)
        assert!(a.reshape_with_args_f([4, 2, 3, 3, 3], false).is_ok()); // (4, 6, 9) -> (4, [2, 3], [3, 3])

        // merge contiguous dimensions into a single dimension
        assert!(a.reshape_with_args_f([4, 54], false).is_ok()); // (4, 6, 9) -> (4, 6 * 9)

        // merge contiguous dimensions and then split
        assert!(a.reshape_with_args_f([4, 3, 6, 3], false).is_ok()); // (4, [6, 9]) -> (4, [3, 6, 3])

        // reshape that requires clone (outputs owned tensor)

        // merge non-contiguous dimensions
        assert!(a.reshape_with_args_f([24, 9], false).is_err()); // (4, 6, 9) -> (4 * 6, 9)
        assert!(a.reshape_with_args_f([-1], false).is_err()); // (4, 6, 9) -> (4 * 6 * 9)
        assert!(a.reshape_with_args_f([12, 2, 9], false).is_err()); // (4, 6, 9) -> (4 * [3, 2], 9)
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
