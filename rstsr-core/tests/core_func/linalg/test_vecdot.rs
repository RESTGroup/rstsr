use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_vecdot {
    use super::*;
    static FUNC: &str = "numpy_vecdot";

    #[test]
    fn test_vecdot_basic() {
        // NumPy v2.4.2, _core/tests/test_ufunc.py, TestUfuncs::test_vecdot (line 814)
        crate::specify_test!("test_vecdot_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr1 = np.arange(6).reshape((2, 3))
        // arr2 = np.arange(3).reshape((1, 3))
        // actual = np.vecdot(arr1, arr2)
        // expected = np.array([5, 14])
        let arr1 = rt::arange((6, &device)).into_shape([2, 3]);
        let arr2 = rt::arange((3, &device)).into_shape([1, 3]);
        let actual = rt::vecdot(&arr1, &arr2, None);
        let expected = rt::tensor_from_nested!([5, 14], &device);
        assert_equal(&actual, &expected, None);

        // actual2 = np.vecdot(arr1.T, arr2.T, axis=-2)
        // assert_array_equal(actual2, expected)
        let actual2 = rt::vecdot(&arr1.t(), &arr2.t(), -2);
        assert_equal(&actual2, &expected, None);
    }

    #[test]
    fn test_vecdot_broadcast() {
        // NumPy v2.4.2, _core/tests/test_ufunc.py, TestUfuncs::test_broadcast (line 924)
        crate::specify_test!("test_vecdot_broadcast");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // msg = "broadcast"
        // a = np.arange(4).reshape((2, 1, 2))
        // b = np.arange(4).reshape((1, 2, 2))
        // assert_array_equal(np.vecdot(a, b), np.sum(a * b, axis=-1), err_msg=msg)
        let a = rt::arange((4, &device)).into_shape([2, 1, 2]);
        let b = rt::arange((4, &device)).into_shape([1, 2, 2]);
        let result = rt::vecdot(&a, &b, None);
        let expected = rt::sum_axes(&a * &b, -1);
        assert_equal(&result, &expected, None);

        // msg = "extend & broadcast loop dimensions"
        // b = np.arange(4).reshape((2, 2))
        // assert_array_equal(np.vecdot(a, b), np.sum(a * b, axis=-1), err_msg=msg)
        let b2 = rt::arange((4, &device)).into_shape([2, 2]);
        let result2 = rt::vecdot(&a, &b2, None);
        let expected2 = rt::sum_axes(&a * &b2, -1);
        assert_equal(&result2, &expected2, None);
    }

    #[test]
    fn test_vecdot_broadcast_fails() {
        // NumPy v2.4.2, _core/tests/test_ufunc.py, TestUfuncs::test_broadcast (line 932)
        crate::specify_test!("test_vecdot_broadcast_fails");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Broadcast in core dimensions should fail
        // a = np.arange(8).reshape((4, 2))
        // b = np.arange(4).reshape((4, 1))
        // assert_raises(ValueError, np.vecdot, a, b)
        let a = rt::arange((8, &device)).into_shape([4, 2]);
        let b = rt::arange((4, &device)).into_shape([4, 1]);
        assert!(rt::vecdot_f(&a, &b, None).is_err());

        // Extend core dimensions should fail
        // a = np.arange(8).reshape((4, 2))
        // b = np.array(7)
        // assert_raises(ValueError, np.vecdot, a, b)
        let b2: Tensor<i32, _> = rt::asarray((7, &device));
        assert!(rt::vecdot_f(&a, &b2, None).is_err());

        // Broadcast should fail
        // a = np.arange(2).reshape((2, 1, 1))
        // b = np.arange(3).reshape((3, 1, 1))
        // assert_raises(ValueError, np.vecdot, a, b)
        let a2 = rt::arange((2, &device)).into_shape([2, 1, 1]);
        let b3 = rt::arange((3, &device)).into_shape([3, 1, 1]);
        assert!(rt::vecdot_f(&a2, &b3, None).is_err());
    }

    #[test]
    fn test_vecdot_complex() {
        // NumPy v2.4.2, _core/tests/test_ufunc.py, TestUfuncs::test_vecdot_matvec_vecmat_complex (line 877)
        crate::specify_test!("test_vecdot_complex");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        use num::complex::{c64, Complex};

        // arr1 = np.array([1, 2j, 3])
        // arr2 = np.array([1, 2, 3])
        // For vecdot without conjugation: 1*1 + 2j*2 + 3*3 = 1 + 4j + 9 = 10 + 4j
        let arr1: Tensor<Complex<f64>, _> = rt::asarray((vec![c64(1.0, 0.0), c64(0.0, 2.0), c64(3.0, 0.0)], &device));
        let arr2: Tensor<Complex<f64>, _> = rt::asarray((vec![c64(1.0, 0.0), c64(2.0, 0.0), c64(3.0, 0.0)], &device));

        // Reshape to (3,) for basic test
        let a = arr1.clone();
        let b = arr2.clone();
        let result = rt::vecdot(&a, &b, None);
        let expected: Tensor<Complex<f64>, _> = rt::asarray((c64(10.0, -4.0), &device));
        assert!(rt::allclose(&result, &expected, None));

        // Test with different shapes
        // actual1 = ufunc(arr1.reshape(shape1), arr2.reshape(shape2))
        // where shape1 = (3,), shape2 = (1, 3)
        let a2 = arr1.clone();
        let b2 = arr2.clone().into_shape([1, 3]);
        let result2 = rt::vecdot(&a2, &b2, None);
        // Result shape should be (1,) broadcasted from scalar
        assert_eq!(result2.ndim(), 1);
    }
}

#[cfg(test)]
mod custom_vecdot {
    use super::*;
    static FUNC: &str = "custom_vecdot";

    #[test]
    fn test_vecdot_empty() {
        crate::specify_test!("test_vecdot_empty");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Empty arrays should produce empty result
        let a: Tensor<i32, _> = rt::zeros(([0], &device));
        let b: Tensor<i32, _> = rt::zeros(([0], &device));
        let result = rt::vecdot(&a, &b, None);

        // Sum of empty is 0
        assert_eq!(result.to_scalar(), 0);
    }

    #[test]
    fn test_vecdot_non_contiguous() {
        crate::specify_test!("test_vecdot_non_contiguous");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Non-contiguous arrays (strided views)
        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let b = rt::arange((8, 20, &device)).into_shape([3, 4]);

        // Take every other column
        let a_view = a.i((.., slice!(None, None, 2)));
        let b_view = b.i((.., slice!(None, None, 2)));

        let result = rt::vecdot(&a_view, &b_view, None);
        let expect = rt::asarray((vec![20, 132, 308], &device));
        assert!(rt::allclose(result, expect, None));
    }
}

#[cfg(test)]
mod doc_vecdot {
    use super::*;
    static FUNC: &str = "doc_vecdot";

    #[test]
    fn doc_vecdot() {
        crate::specify_test!("doc_vecdot");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Basic vector dot product
        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let b = rt::tensor_from_nested!([4, 5, 6], &device);
        let result = rt::vecdot(&a, &b, None);
        println!("{result}");
        // 32
        assert_eq!(result.to_scalar(), 32);

        // 2-dim dot product
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let b = rt::tensor_from_nested!([[5, 6], [7, 8]], &device);
        let result = rt::vecdot(&a, &b, None);
        println!("{result}");
        // [ 17 53]
        let expected = rt::tensor_from_nested!([17, 53], &device);
        assert!(rt::allclose(&result, &expected, None));

        // 2-dim broadcasted dot product (row-major)
        let a = rt::tensor_from_nested!([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]], &device);
        let b = rt::tensor_from_nested!([0., 0.6, 0.8], &device);
        let result = rt::vecdot(&a, &b, None);
        println!("{result}");
        // [ 3 8 10]
        let expected = rt::tensor_from_nested!([3., 8., 10.], &device);
        assert!(rt::allclose(&result, &expected, None));

        // complex dot product (conjugates a)
        use num::complex::c64;
        let a = rt::tensor_from_nested!([c64(1., 0.), c64(2., 2.), c64(3., 0.)], &device);
        let b = rt::tensor_from_nested!([c64(1., 0.), c64(2., 0.), c64(3., 3.)], &device);
        // 1 * 1 + (2 - 2j) * 2 + 3 * (3 + 3j) = 14 + 5j
        //          conj               identity
        let result = rt::vecdot(&a, &b, None);
        println!("{result}");
        // 14+5i
        assert_eq!(result.to_scalar(), c64(14., 5.));
    }
}
