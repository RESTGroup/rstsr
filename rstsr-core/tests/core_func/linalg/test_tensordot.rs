#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_tensordot {
    use super::*;
    static FUNC: &str = "numpy_tensordot";

    #[test]
    fn test_rejects_duplicate_axes() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestTensordot::test_rejects_duplicate_axes (L4234)
        crate::specify_test!("test_rejects_duplicate_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.ones((2, 3, 3))
        // b = np.ones((3, 3, 4))
        // with pytest.raises(ValueError):
        //     np.tensordot(a, b, axes=([1, 1], [0, 0]))
        let a: Tensor<f64, _> = rt::ones(([2, 3, 3], &device));
        let b: Tensor<f64, _> = rt::ones(([3, 3, 4], &device));
        assert!(rt::tensordot_f(&a, &b, (vec![1, 1], vec![0, 0])).is_err());
    }

    #[test]
    fn test_zero_dimension() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestTensordot::test_zero_dimension (L4240)
        crate::specify_test!("test_zero_dimension");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.ndarray((3, 0))
        // b = np.ndarray((0, 4))
        // td = np.tensordot(a, b, (1, 0))
        // assert_array_equal(td, np.dot(a, b))
        let a: Tensor<f64, _> = rt::zeros(([3, 0], &device));
        let b: Tensor<f64, _> = rt::zeros(([0, 4], &device));
        let td = rt::tensordot(&a, &b, (1, 0));
        let expected: Tensor<f64, _> = rt::zeros(([3, 4], &device));
        assert_equal(&td, &expected, None);
    }

    #[test]
    fn test_zero_dimensional() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestTensordot::test_zero_dimensional (L4248)
        crate::specify_test!("test_zero_dimensional");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr_0d = np.array(1)
        // # contracting no axes is well defined
        // ret = np.tensordot(arr_0d, arr_0d, ([], []))
        // assert_array_equal(ret, arr_0d)
        let arr = rt::asarray((1.0, &device));
        let ret = rt::tensordot(&arr, &arr, (Vec::<isize>::new(), Vec::<isize>::new()));
        assert_equal(&ret, &arr, None);
    }
}

#[cfg(test)]
mod custom_tensordot {
    use super::*;
    static FUNC: &str = "custom_tensordot";

    #[test]
    fn test_outer_and_matmul() {
        // cases whose expected values were generated with NumPy v2.5.2
        crate::specify_test!("test_outer_and_matmul");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.tensordot(np.array([1, 2]), np.array([3, 4, 5]), axes=0)
        //   -> array([[3, 4, 5], [6, 8, 10]])
        let a = rt::tensor_from_nested!([1, 2], &device);
        let b = rt::tensor_from_nested!([3, 4, 5], &device);
        let actual = rt::tensordot(&a, &b, 0);
        let expected = rt::tensor_from_nested!([[3, 4, 5], [6, 8, 10]], &device);
        assert_eq!(actual.shape(), &[2, 3]);
        assert_equal(&actual, &expected, None);

        // np.tensordot(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), axes=1)
        //   -> array([[19, 22], [43, 50]])
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let b = rt::tensor_from_nested!([[5, 6], [7, 8]], &device);
        let actual = rt::tensordot(&a, &b, 1);
        let expected = rt::tensor_from_nested!([[19, 22], [43, 50]], &device);
        assert_equal(&actual, &expected, None);
    }

    #[test]
    fn test_full_contraction_and_default() {
        crate::specify_test!("test_full_contraction_and_default");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let b = rt::tensor_from_nested!([[5, 6], [7, 8]], &device);

        // np.tensordot(a, b, axes=2) -> 0-d, sum(a * b) = 70
        let actual = rt::tensordot(&a, &b, 2);
        assert_eq!(actual.shape(), &[] as &[usize]);
        assert_eq!(actual.to_scalar(), 70);

        // array-API default is axes=2 -> `None` / `()` carry the same meaning
        let d1 = rt::tensordot(&a, &b, None);
        let d2 = rt::tensordot(&a, &b, ());
        assert_eq!(d1.to_scalar(), 70);
        assert_eq!(d2.to_scalar(), 70);
    }

    #[test]
    fn test_pair_axes_3d() {
        // example from the NumPy tensordot docstring (cross-checked): contract
        // a-axis [1, 0] with b-axis [0, 1]
        crate::specify_test!("test_pair_axes_3d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.arange(60.).reshape(3, 4, 5)
        // b = np.arange(24.).reshape(4, 3, 2)
        // np.tensordot(a, b, axes=([1, 0], [0, 1]))
        let a = rt::arange((60, &device)).into_shape((3, 4, 5));
        let b = rt::arange((24, &device)).into_shape((4, 3, 2));
        let actual = rt::tensordot(&a, &b, ([1, 0], [0, 1]));
        let expected = rt::tensor_from_nested!(
            [[4400, 4730], [4532, 4874], [4664, 5018], [4796, 5162], [4928, 5306]],
            &device
        );
        assert_eq!(actual.shape(), &[5, 2]);
        assert_equal(&actual, &expected, None);

        // negative axes fold to the same result
        let neg = rt::tensordot(&a, &b, (vec![-2, -3], vec![0, 1]));
        assert_equal(&neg, &expected, None);
    }

    #[test]
    fn test_err_n_too_large() {
        crate::specify_test!("test_err_n_too_large");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // array-API: an integer `axes` must be non-negative and <= both ndims
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        assert!(rt::tensordot_f(&a, &a, 5).is_err());
    }

    #[test]
    fn test_err_negative_n() {
        crate::specify_test!("test_err_negative_n");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        assert!(rt::tensordot_f(&a, &a, -1).is_err());
    }
}
