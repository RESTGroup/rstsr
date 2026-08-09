#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

// Parity tests for `atleast_1d` / `atleast_2d` / `atleast_3d`.
//
// Source: NumPy v2.5.2, `_core/tests/test_shape_base.py::{TestAtleast1d, TestAtleast2d,
// TestAtleast3d}`. rstsr's `atleast_*` are view-only promotions (no copy): 0-D / 1-D /
// 2-D inputs are reshaped via `expand_dims` to reach the target rank; higher-rank
// inputs are returned unchanged.

#[cfg(test)]
mod numpy_atleast_1d {
    use super::*;
    static FUNC: &str = "numpy_atleast_1d";

    #[test]
    fn test_0d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast1d::test_0D_array (line 37)
        crate::specify_test!("test_0d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // a = array(1); b = array(2)
        // res = [atleast_1d(a), atleast_1d(b)]; desired = [array([1]), array([2])]
        let a: Tensor<i32, _> = rt::asarray((1, &device));
        let b: Tensor<i32, _> = rt::asarray((2, &device));
        assert_equal(rt::atleast_1d(&a), &rt::tensor_from_nested!([1], &device), None);
        assert_equal(rt::atleast_1d(&b), &rt::tensor_from_nested!([2], &device), None);
    }

    #[test]
    fn test_1d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast1d::test_1D_array (line 44)
        crate::specify_test!("test_1d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // a = array([1, 2]); b = array([2, 3]); atleast_1d leaves them unchanged
        let a = rt::tensor_from_nested!([1, 2], &device);
        let b = rt::tensor_from_nested!([2, 3], &device);
        assert_equal(rt::atleast_1d(&a), &a, None);
        assert_equal(rt::atleast_1d(&b), &b, None);
    }

    #[test]
    fn test_2d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast1d::test_2D_array (line 51)
        crate::specify_test!("test_2d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // a = array([[1, 2], [1, 2]]); b = array([[2, 3], [2, 3]]); unchanged
        let a = rt::tensor_from_nested!([[1, 2], [1, 2]], &device);
        let b = rt::tensor_from_nested!([[2, 3], [2, 3]], &device);
        assert_equal(rt::atleast_1d(&a), &a, None);
        assert_equal(rt::atleast_1d(&b), &b, None);
    }

    #[test]
    fn test_3d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast1d::test_3D_array (line 58)
        crate::specify_test!("test_3d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // a = array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]); unchanged
        let a = rt::tensor_from_nested!([[[1, 2], [1, 2]], [[1, 2], [1, 2]]], &device);
        assert_equal(rt::atleast_1d(&a), &a, None);
    }

    #[test]
    fn test_r1array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast1d::test_r1array (line 65)
        crate::specify_test!("test_r1array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // assert_(atleast_1d(3).shape == (1,))
        let s: Tensor<i32, _> = rt::asarray((3, &device));
        assert_eq!(rt::atleast_1d(&s).shape(), &[1]);
        // assert_(atleast_1d([[2, 3], [4, 5]]).shape == (2, 2))
        let a = rt::tensor_from_nested!([[2, 3], [4, 5]], &device);
        assert_eq!(rt::atleast_1d(&a).shape(), &[2, 2]);
    }
}

#[cfg(test)]
mod numpy_atleast_2d {
    use super::*;
    static FUNC: &str = "numpy_atleast_2d";

    #[test]
    fn test_0d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast2d::test_0D_array (line 77)
        crate::specify_test!("test_0d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // a = array(1); b = array(2); desired = [array([[1]]), array([[2]])]
        let a: Tensor<i32, _> = rt::asarray((1, &device));
        let b: Tensor<i32, _> = rt::asarray((2, &device));
        assert_eq!(rt::atleast_2d(&a).shape(), &[1, 1]);
        assert_equal(rt::atleast_2d(&a), &rt::tensor_from_nested!([[1]], &device), None);
        assert_equal(rt::atleast_2d(&b), &rt::tensor_from_nested!([[2]], &device), None);
    }

    #[test]
    fn test_1d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast2d::test_1D_array (line 84)
        crate::specify_test!("test_1d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // a = array([1, 2]); b = array([2, 3]); desired = [array([[1, 2]]), array([[2, 3]])]
        let a = rt::tensor_from_nested!([1, 2], &device);
        let b = rt::tensor_from_nested!([2, 3], &device);
        assert_eq!(rt::atleast_2d(&a).shape(), &[1, 2]);
        assert_equal(rt::atleast_2d(&a), &rt::tensor_from_nested!([[1, 2]], &device), None);
        assert_equal(rt::atleast_2d(&b), &rt::tensor_from_nested!([[2, 3]], &device), None);
    }

    #[test]
    fn test_2d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast2d::test_2D_array (line 91)
        crate::specify_test!("test_2d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        let a = rt::tensor_from_nested!([[1, 2], [1, 2]], &device);
        let b = rt::tensor_from_nested!([[2, 3], [2, 3]], &device);
        assert_equal(rt::atleast_2d(&a), &a, None);
        assert_equal(rt::atleast_2d(&b), &b, None);
    }

    #[test]
    fn test_3d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast2d::test_3D_array (line 98)
        crate::specify_test!("test_3d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        let a = rt::tensor_from_nested!([[[1, 2], [1, 2]], [[1, 2], [1, 2]]], &device);
        assert_equal(rt::atleast_2d(&a), &a, None);
    }

    #[test]
    fn test_r2array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast2d::test_r2array (line 105)
        crate::specify_test!("test_r2array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // assert_(atleast_2d(3).shape == (1, 1))
        let s: Tensor<i32, _> = rt::asarray((3, &device));
        assert_eq!(rt::atleast_2d(&s).shape(), &[1, 1]);
        // assert_(atleast_2d([3j, 1]).shape == (1, 2))  -> use real values; shape-only
        let a = rt::tensor_from_nested!([3.0, 1.0], &device);
        assert_eq!(rt::atleast_2d(&a).shape(), &[1, 2]);
        // assert_(atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape == (2, 2, 2))
        let a = rt::tensor_from_nested!([[[3, 1], [4, 5]], [[3, 5], [1, 2]]], &device);
        assert_eq!(rt::atleast_2d(&a).shape(), &[2, 2, 2]);
    }
}

#[cfg(test)]
mod numpy_atleast_3d {
    use super::*;
    static FUNC: &str = "numpy_atleast_3d";

    #[test]
    fn test_0d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast3d::test_0D_array (line 116)
        crate::specify_test!("test_0d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // a = array(1); b = array(2); desired = [array([[[1]]]), array([[[2]]])]
        let a: Tensor<i32, _> = rt::asarray((1, &device));
        let b: Tensor<i32, _> = rt::asarray((2, &device));
        assert_eq!(rt::atleast_3d(&a).shape(), &[1, 1, 1]);
        assert_equal(rt::atleast_3d(&a), &rt::tensor_from_nested!([[[1]]], &device), None);
        assert_equal(rt::atleast_3d(&b), &rt::tensor_from_nested!([[[2]]], &device), None);
    }

    #[test]
    fn test_1d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast3d::test_1D_array (line 123)
        crate::specify_test!("test_1d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // a = array([1, 2]); b = array([2, 3])
        // desired = [array([[[1], [2]]]), array([[[2], [3]]])]  (shape (1, N, 1))
        let a = rt::tensor_from_nested!([1, 2], &device);
        let b = rt::tensor_from_nested!([2, 3], &device);
        assert_eq!(rt::atleast_3d(&a).shape(), &[1, 2, 1]);
        assert_equal(rt::atleast_3d(&a), &rt::tensor_from_nested!([[[1], [2]]], &device), None);
        assert_equal(rt::atleast_3d(&b), &rt::tensor_from_nested!([[[2], [3]]], &device), None);
    }

    #[test]
    fn test_2d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast3d::test_2D_array (line 130)
        crate::specify_test!("test_2d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        // a = array([[1, 2], [1, 2]]); desired = a[:, :, newaxis] (shape (2, 2, 1))
        let a = rt::tensor_from_nested!([[1, 2], [1, 2]], &device);
        let b = rt::tensor_from_nested!([[2, 3], [2, 3]], &device);
        assert_eq!(rt::atleast_3d(&a).shape(), &[2, 2, 1]);
        assert_equal(rt::atleast_3d(&a), &rt::tensor_from_nested!([[[1], [2]], [[1], [2]]], &device), None);
        assert_equal(rt::atleast_3d(&b), &rt::tensor_from_nested!([[[2], [3]], [[2], [3]]], &device), None);
    }

    #[test]
    fn test_3d_array() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestAtleast3d::test_3D_array (line 137)
        crate::specify_test!("test_3d_array");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        let a = rt::tensor_from_nested!([[[1, 2], [1, 2]], [[1, 2], [1, 2]]], &device);
        assert_equal(rt::atleast_3d(&a), &a, None);
    }
}
