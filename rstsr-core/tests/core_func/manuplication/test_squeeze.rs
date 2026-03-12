use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_squeeze {
    use super::*;
    static FUNC: &str = "numpy_squeeze";

    #[test]
    fn test_basic() {
        // NumPy v2.4.2, _core/tests/test_numeric.py, TestBool::test_squeeze (line 292)
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

#[cfg(test)]
mod docs_squeeze {
    use super::*;
    static FUNC: &str = "docs_squeeze";

    #[test]
    fn squeeze_single_axis() {
        crate::specify_test!("squeeze_single_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Squeeze a tensor along axis 0
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4], &device));
        let b = a.squeeze(0);
        assert_eq!(b.shape(), &[3, 1, 4]);

        // Squeeze a tensor along axis 2 (third axis with size 1)
        let b = a.squeeze(2);
        assert_eq!(b.shape(), &[1, 3, 4]);

        // Squeeze using negative index (-2 refers to the third axis with size 1)
        let b = a.squeeze(-2);
        assert_eq!(b.shape(), &[1, 3, 4]);
    }

    #[test]
    fn squeeze_multiple_axes() {
        crate::specify_test!("squeeze_multiple_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Squeeze multiple axes at once
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4, 1], &device));
        let b = a.squeeze([0, 2]);
        assert_eq!(b.shape(), &[3, 4, 1]);

        // Use negative indices to squeeze from the back
        let b = a.squeeze([0, -1]);
        assert_eq!(b.shape(), &[3, 1, 4]);
    }

    #[test]
    fn squeeze_all_singletons() {
        crate::specify_test!("squeeze_all_singletons");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Use None to squeeze all axes with size 1
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4, 1], &device));
        let b = a.squeeze(None);
        assert_eq!(b.shape(), &[3, 4]);
    }

    #[test]
    fn squeeze_empty_axes() {
        crate::specify_test!("squeeze_empty_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Use an empty tuple () to squeeze no axes
        let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4, 1], &device));
        let b = a.squeeze(());
        assert_eq!(b.shape(), &[1, 3, 1, 4, 1]);
    }

    #[test]
    fn squeeze_roundtrip_with_expand_dims() {
        crate::specify_test!("squeeze_roundtrip_with_expand_dims");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Test that squeeze and expand_dims are inverse operations
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let b = rt::expand_dims(&a, [0, 2]);
        assert_eq!(b.shape(), &[1, 2, 1, 3]);

        let c = b.squeeze([0, 2]);
        assert_eq!(c.shape(), &[2, 3]);
        assert!(rt::allclose(&a, &c, None));
    }
}
