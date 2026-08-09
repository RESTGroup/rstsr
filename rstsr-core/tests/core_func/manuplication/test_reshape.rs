#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_reshape {
    use super::*;
    static FUNC: &str = "numpy_reshape";

    #[test]
    fn multiarray() {
        // NumPy v2.5.2, _core/tests/test_multiarray.py, TestMethods::test_reshape (line 2206)
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
        // NumPy v2.5.2, _core/tests/test_regression.py, TestRegression::test_reshape*
        crate::specify_test!("regression_reshape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // CASE test_reshape_order (line 642)

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

        // CASE test_reshape_zero_strides (line 650)

        // a = np.ones(1)
        // a = as_strided(a, shape=(5,), strides=(0,))
        // assert_(a.reshape(5, 1).strides[0] == 0)
        let layout = unsafe { Layout::new_unchecked([5], [0], 0) };
        let a = rt::asarray((vec![1], layout, &device));
        assert!(a.reshape([5, 1]).stride()[0] == 0);

        // CASE test_reshape_zero_size (line 656)

        // a = np.ones((0, 2))
        // a.shape = (-1, 2)
        let a: Tensor<i32, _> = rt::ones(([0, 2], &device));
        let _a_reshaped = a.reshape([-1, 2]);

        // CASE test_reshape_trailing_ones_strides (line 662)

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

        // CASE test_reshape_size_overflow (line 2275)
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
        let panics = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.reshape_f(new_shape)));
        assert!(panics.is_err());
    }

    #[test]
    fn numeric() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestNonarrayArgs::test_reshape*

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // CASE test_reshape (line 177)

        // arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        // tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        // assert_equal(np.reshape(arr, (2, 6)), tgt)
        let arr = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], &device);
        let tgt = rt::tensor_from_nested!([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], &device);
        assert_equal(arr.reshape([2, 6]), &tgt, None);

        // CASE test_reshape_shape_arg (line 182)

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

        // CASE test_reshape_copy_arg (line 200)

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

    #[test]
    fn test_ravel() {
        // NumPy v2.5.2, _core/tests/test_multiarray.py, TestMethods::test_ravel (line 4088)
        crate::specify_test!("test_ravel");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // rstsr has no dedicated ravel; ravel maps to reshape(-1):
        //   ravel(order='C') == reshape(-1), ravel(order='F') == reshape_with_args(-1, ColMajor)
        // N/A: order A/K unsupported (intentional, see numpy_differences.md)

        // a = np.array([[0, 1], [2, 3]])
        // assert_equal(a.ravel(), [0, 1, 2, 3])
        // assert_equal(a.ravel(order='F'), [0, 2, 1, 3])
        let a = rt::tensor_from_nested!([[0, 1], [2, 3]], &device);
        let ravel_c = rt::tensor_from_nested!([0, 1, 2, 3], &device);
        let ravel_f = rt::tensor_from_nested!([0, 2, 1, 3], &device);
        assert_equal(rt::reshape(&a, &[-1]), &ravel_c, None);
        assert_equal(rt::reshape_with_args(&a, &[-1], ColMajor), &ravel_f, None);

        // a = np.array([[0, 1], [2, 3]], order='F')  # F-contiguous
        // assert_equal(a.ravel(), [0, 1, 2, 3])
        // assert_equal(a.ravel(order='A'), [0, 2, 1, 3])   # 'A' == 'F' for F-contiguous input
        // N/A: order 'A' unsupported
        let a = rt::asarray((vec![0, 2, 1, 3], [2, 2].f(), &device));
        assert_equal(rt::reshape(&a, &[-1]), &ravel_c, None);
        assert_equal(rt::reshape_with_args(&a, &[-1], ColMajor), &ravel_f, None);

        // a = np.array([[0, 1], [2, 3]])[::-1, :]  # negative-stride (flipped) input
        // assert_equal(a.ravel(), [2, 3, 0, 1])
        // assert_equal(a.ravel(order='F'), [2, 0, 3, 1])
        let a = rt::tensor_from_nested!([[0, 1], [2, 3]], &device).into_flip(0);
        let ravel_c = rt::tensor_from_nested!([2, 3, 0, 1], &device);
        let ravel_f = rt::tensor_from_nested!([2, 0, 3, 1], &device);
        assert_equal(rt::reshape(&a, &[-1]), &ravel_c, None);
        assert_equal(rt::reshape_with_args(&a, &[-1], ColMajor), &ravel_f, None);
    }

    #[test]
    fn test_flatten() {
        // NumPy v2.5.2, _core/tests/test_multiarray.py, TestMethods::test_flatten (line 3717)
        crate::specify_test!("test_flatten");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // rstsr has no dedicated flatten; flatten == reshape(-1):
        //   flatten() == reshape(-1), flatten('F') == reshape_with_args(-1, ColMajor)
        // NOTE: numpy flatten always copies, while rstsr reshape(-1) returns a view when
        // possible (matching ravel, not flatten) - intentional difference, see numpy_differences.md

        // x0 = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        // y0 = np.array([1, 2, 3, 4, 5, 6], np.int32)
        // y0f = np.array([1, 4, 2, 5, 3, 6], np.int32)
        // assert_equal(x0.flatten(), y0)
        // assert_equal(x0.flatten('F'), y0f)
        // assert_equal(x0.flatten('F'), x0.T.flatten())
        let x0 = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        let y0 = rt::tensor_from_nested!([1, 2, 3, 4, 5, 6], &device);
        let y0f = rt::tensor_from_nested!([1, 4, 2, 5, 3, 6], &device);
        assert_equal(rt::reshape(&x0, &[-1]), &y0, None);
        assert_equal(rt::reshape_with_args(&x0, &[-1], ColMajor), &y0f, None);
        assert_equal(rt::reshape_with_args(&x0, &[-1], ColMajor), x0.t().reshape(&[-1]), None);

        // x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], np.int32)
        // y1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], np.int32)
        // y1f = np.array([1, 5, 3, 7, 2, 6, 4, 8], np.int32)
        // assert_equal(x1.flatten(), y1)
        // assert_equal(x1.flatten('F'), y1f)
        // assert_equal(x1.flatten('F'), x1.T.flatten())
        let x1 = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
        let y1 = rt::tensor_from_nested!([1, 2, 3, 4, 5, 6, 7, 8], &device);
        let y1f = rt::tensor_from_nested!([1, 5, 3, 7, 2, 6, 4, 8], &device);
        assert_equal(rt::reshape(&x1, &[-1]), &y1, None);
        assert_equal(rt::reshape_with_args(&x1, &[-1], ColMajor), &y1f, None);
        assert_equal(rt::reshape_with_args(&x1, &[-1], ColMajor), x1.t().reshape(&[-1]), None);
    }

    #[test]
    fn test_ravel_with_order() {
        // NumPy v2.5.2, _core/tests/test_regression.py, TestRegression::test_ravel_with_order (line 80)
        crate::specify_test!("test_ravel_with_order");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.ones(2)
        // assert_(not a.ravel('F').flags.owndata)   # F-ravel of a C-contiguous 1-D array is a view
        let a: Tensor<i32, _> = rt::ones(([2], &device));
        let r = rt::reshape_with_args(&a, &[-1], ColMajor);
        assert!(core::ptr::eq(a.as_ptr(), r.as_ptr()));
    }
}
