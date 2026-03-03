use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_expand_dims {
    use super::*;
    static FUNC: &str = "numpy_expand_dims";

    #[test]
    fn test_functionality() {
        // NumPy v2.4.2, lib/tests/test_shape_base.py, TestExpandDims::test_functionality (line 310)
        crate::specify_test!("test_functionality");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // s = (2, 3, 4, 5)
        // a = np.empty(s)
        // for axis in range(-5, 4):
        //     b = expand_dims(a, axis)
        //     assert_(b.shape[axis] == 1)
        //     assert_(np.squeeze(b).shape == s)
        let s = [2, 3, 4, 5];
        let a: Tensor<f64, _> = unsafe { rt::empty((s, &device)) };
        for axis in -5..4 {
            let b = rt::expand_dims(&a, axis);
            let axis_usize = if axis < 0 { (b.ndim() as isize + axis) as usize } else { axis as usize };
            assert_eq!(b.shape()[axis_usize], 1);
            let squeezed = rt::squeeze(&b, axis_usize);
            assert_eq!(squeezed.shape(), &s);
        }
    }

    #[test]
    fn test_axis_tuple() {
        // NumPy v2.4.2, lib/tests/test_shape_base.py, TestExpandDims::test_axis_tuple (line 318)
        crate::specify_test!("test_axis_tuple");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::zeros(([3, 3, 3], &device));

        // np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        let b = rt::expand_dims(&a, [0, 1, 2]);
        assert_eq!(b.shape(), &[1, 1, 1, 3, 3, 3]);

        // np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        let b = rt::expand_dims(&a, [0, -1, -2]);
        assert_eq!(b.shape(), &[1, 3, 3, 3, 1, 1]);

        // np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        let b = rt::expand_dims(&a, [0, 3, 5]);
        assert_eq!(b.shape(), &[1, 3, 3, 1, 3, 1]);

        // np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)
        let b = rt::expand_dims(&a, [0, -3, -5]);
        assert_eq!(b.shape(), &[1, 1, 3, 1, 3, 3]);
    }

    #[test]
    fn test_axis_out_of_range() {
        // NumPy v2.4.2, lib/tests/test_shape_base.py, TestExpandDims::test_axis_out_of_range (line 325)
        crate::specify_test!("test_axis_out_of_range");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // s = (2, 3, 4, 5)
        // a = np.empty(s)
        // assert_raises(AxisError, expand_dims, a, -6)
        let a: Tensor<f64, _> = unsafe { rt::empty(([2, 3, 4, 5], &device)) };
        assert!(rt::expand_dims_f(&a, -6).is_err());

        // assert_raises(AxisError, expand_dims, a, 5)
        assert!(rt::expand_dims_f(&a, 5).is_err());

        // a = np.empty((3, 3, 3))
        let a: Tensor<f64, _> = unsafe { rt::empty(([3, 3, 3], &device)) };

        // assert_raises(AxisError, expand_dims, a, (0, -6))
        assert!(rt::expand_dims_f(&a, [0, -6]).is_err());

        // assert_raises(AxisError, expand_dims, a, (0, 5))
        assert!(rt::expand_dims_f(&a, [0, 5]).is_err());
    }

    #[test]
    fn test_repeated_axis() {
        // NumPy v2.4.2, lib/tests/test_shape_base.py, TestExpandDims::test_repeated_axis (line 335)
        crate::specify_test!("test_repeated_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.empty((3, 3, 3))
        // assert_raises(ValueError, expand_dims, a, axis=(1, 1))
        let a: Tensor<f64, _> = unsafe { rt::empty(([3, 3, 3], &device)) };
        assert!(rt::expand_dims_f(&a, [1, 1]).is_err());
    }
}
