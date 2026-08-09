use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_squeeze {
    use super::*;
    static FUNC: &str = "doc_squeeze";

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
