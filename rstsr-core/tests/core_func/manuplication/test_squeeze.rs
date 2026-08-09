use rstsr::prelude::*;

use super::CATEGORY;
use crate::test_utils::*;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_squeeze {
    use super::*;
    static FUNC: &str = "numpy_squeeze";

    #[test]
    fn test_basic() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestNonarrayArgs::test_squeeze (line 291)
        crate::specify_test!("test_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        // assert_equal(np.squeeze(A).shape, (3, 3))
        let a = rt::tensor_from_nested!([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]], &device);
        let b = a.squeeze(None);
        assert_eq!(b.shape(), &[3, 3]);

        // assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1], &device));
        let b = a.squeeze(None);
        assert_eq!(b.shape(), &[3]);

        // assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1], &device));
        let b = a.squeeze(0);
        assert_eq!(b.shape(), &[3, 1]);

        // assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1], &device));
        let b = a.squeeze(-1);
        assert_eq!(b.shape(), &[1, 3]);

        // assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1], &device));
        let b = a.squeeze(2);
        assert_eq!(b.shape(), &[1, 3]);

        // assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        let inner: Tensor<f64, _> = rt::zeros(([3, 1], &device));
        let a = rt::expand_dims(&inner, 0);
        let b = a.squeeze(None);
        assert_eq!(b.shape(), &[3]);

        // assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        let inner: Tensor<f64, _> = rt::zeros(([3, 1], &device));
        let a = rt::expand_dims(&inner, 0);
        let b = a.squeeze(0);
        assert_eq!(b.shape(), &[3, 1]);

        // assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        let inner: Tensor<f64, _> = rt::zeros(([3, 1], &device));
        let a = rt::expand_dims(&inner, 0);
        let b = a.squeeze(2);
        assert_eq!(b.shape(), &[1, 3]);

        // assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))
        let inner: Tensor<f64, _> = rt::zeros(([3, 1], &device));
        let a = rt::expand_dims(&inner, 0);
        let b = a.squeeze(-1);
        assert_eq!(b.shape(), &[1, 3]);
    }

    #[test]
    fn test_multiarray() {
        // NumPy v2.5.2, _core/tests/test_multiarray.py, TestMethods::test_squeeze (line 2253)
        crate::specify_test!("test_multiarray");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.array([[[1], [2], [3]]])
        // assert_equal(a.squeeze(), [1, 2, 3])
        let a = rt::tensor_from_nested!([[[1], [2], [3]]], &device);
        let expected = rt::tensor_from_nested!([1, 2, 3], &device);
        assert_equal(a.squeeze(None), &expected, None);

        // assert_equal(a.squeeze(axis=(0,)), [[1], [2], [3]])
        let expected = rt::tensor_from_nested!([[1], [2], [3]], &device);
        assert_equal(a.squeeze([0]), &expected, None);

        // assert_raises(ValueError, a.squeeze, axis=(1,))
        // (non-singleton axis; also covered by numpy_squeeze::test_squeeze_non_singleton)
        assert!(a.squeeze_f(1).is_err());

        // assert_equal(a.squeeze(axis=(2,)), [[1, 2, 3]])
        let expected = rt::tensor_from_nested!([[1, 2, 3]], &device);
        assert_equal(a.squeeze([2]), &expected, None);
    }

    #[test]
    fn test_basic_shape_base() {
        // NumPy v2.5.2, lib/tests/test_shape_base.py, TestSqueeze::test_basic (line 657)
        crate::specify_test!("test_basic_shape_base");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = rand(20, 10, 10, 1, 1)
        // assert_array_equal(np.squeeze(a), np.reshape(a, (20, 10, 10)))
        let a: Tensor<i32, _> = rt::arange((2000, &device)).into_shape([20, 10, 10, 1, 1]);
        assert_equal(a.squeeze(None), a.reshape([20, 10, 10]), None);

        // b = rand(20, 1, 10, 1, 20)
        // assert_array_equal(np.squeeze(b), np.reshape(b, (20, 10, 20)))
        let b: Tensor<i32, _> = rt::arange((4000, &device)).into_shape([20, 1, 10, 1, 20]);
        assert_equal(b.squeeze(None), b.reshape([20, 10, 20]), None);

        // c = rand(1, 1, 20, 10)
        // assert_array_equal(np.squeeze(c), np.reshape(c, (20, 10)))
        let c: Tensor<i32, _> = rt::arange((200, &device)).into_shape([1, 1, 20, 10]);
        assert_equal(c.squeeze(None), c.reshape([20, 10]), None);

        // Squeezing to 0-dim should still give an ndarray
        // a = [[[1.5]]]
        // res = np.squeeze(a)
        // assert_equal(res, 1.5)
        // assert_equal(res.ndim, 0)
        let a = rt::tensor_from_nested!([[[1.5]]], &device);
        let res = a.squeeze(None);
        assert_eq!(res.shape(), &[]);
        let expected: Tensor<f64, _> = rt::asarray((1.5, &device));
        assert_equal(res, &expected, None);
        // N/A: assert_equal(type(res), np.ndarray) - every rstsr tensor is a typed tensor
    }

    #[test]
    fn test_squeeze_type() {
        // NumPy v2.5.2, _core/tests/test_regression.py, TestRegression::test_squeeze_type (line 285)
        crate::specify_test!("test_squeeze_type");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Ticket #133
        // a = np.array([3])
        // assert_(type(a.squeeze()) is np.ndarray)
        let a = rt::tensor_from_nested!([3], &device);
        let res = a.squeeze(None);
        assert_eq!(res.shape(), &[]);
        assert_eq!(res.to_scalar(), 3);
        // N/A: assert_(type(res) is np.ndarray) - every rstsr tensor is a typed tensor

        // b = np.array(3)
        // assert_(type(b.squeeze()) is np.ndarray)
        let b: Tensor<i32, _> = rt::asarray((3, &device));
        let res = b.squeeze(None);
        assert_eq!(res.shape(), &[]);
        assert_eq!(res.to_scalar(), 3);
        // N/A: assert_(type(res) is np.ndarray) - every rstsr tensor is a typed tensor
    }

    #[test]
    fn test_squeeze_contiguous() {
        // NumPy v2.5.2, _core/tests/test_regression.py, TestRegression::test_squeeze_contiguous (line 1695)
        crate::specify_test!("test_squeeze_contiguous");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.zeros((1, 2)).squeeze()
        // assert_(a.flags.c_contiguous)
        // assert_(a.flags.f_contiguous)
        let a: Tensor<f64, _> = rt::zeros(([1, 2], &device));
        let a = a.squeeze(None);
        assert_eq!(a.shape(), &[2]);
        assert!(a.c_contig());
        assert!(a.f_contig());

        // b = np.zeros((2, 2, 2), order='F')[:, :, ::2].squeeze()
        // assert_(b.flags.f_contiguous)
        let b: Tensor<f64, _> = rt::zeros(([2, 2, 2], &device)).into_layout(vec![2, 2, 2].f());
        let b = b.i((.., .., slice!(None, None, 2)));
        let b = b.squeeze(None);
        assert_eq!(b.shape(), &[2, 2]);
        assert!(b.f_contig());
    }

    #[test]
    fn test_axis_out_of_range() {
        crate::specify_test!("test_axis_out_of_range");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Axis out of range should error
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1], &device));
        assert!(a.squeeze_f(3).is_err());
        assert!(a.squeeze_f(-4).is_err());
    }

    #[test]
    fn test_repeated_axis() {
        crate::specify_test!("test_repeated_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Repeated axes should error
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1], &device));
        assert!(a.squeeze_f([0, 0]).is_err());
    }

    #[test]
    fn test_squeeze_non_singleton() {
        crate::specify_test!("test_squeeze_non_singleton");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Squeezing a non-singleton axis should error
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1], &device));
        assert!(a.squeeze_f(1).is_err());
    }

    #[test]
    fn test_multiple_axes() {
        crate::specify_test!("test_multiple_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4, 1], &device));

        // Squeeze multiple axes
        let b = a.squeeze([0, 2, 4]);
        assert_eq!(b.shape(), &[3, 4]);

        // Squeeze with negative indices
        let b = a.squeeze([-1, -3, 0]);
        assert_eq!(b.shape(), &[3, 4]);
    }

    #[test]
    fn test_empty_axes() {
        crate::specify_test!("test_empty_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Empty axes should return the same shape
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4, 1], &device));
        let b = a.squeeze(());
        assert_eq!(b.shape(), &[1, 3, 1, 4, 1]);
    }
}
