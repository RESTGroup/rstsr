#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_transpose {
    use super::*;
    static FUNC: &str = "numpy_transpose";

    #[test]
    fn test_multiarray() {
        // NumPy v2.5.2, _core/tests/test_multiarray.py, TestMethods::test_transpose (line 2260)
        crate::specify_test!("test_multiarray");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.array([[1, 2], [3, 4]])
        // assert_equal(a.transpose(), [[1, 3], [2, 4]])
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let expected = rt::tensor_from_nested!([[1, 3], [2, 4]], &device);
        assert_equal(rt::transpose(&a, None), &expected, None);
        assert_equal(a.t(), &expected, None);

        // assert_raises(ValueError, lambda: a.transpose(0))
        assert!(rt::transpose_f(&a, [0]).is_err());

        // assert_raises(ValueError, lambda: a.transpose(0, 0))
        assert!(rt::transpose_f(&a, [0, 0]).is_err());

        // assert_raises(ValueError, lambda: a.transpose(0, 1, 2))
        assert!(rt::transpose_f(&a, [0, 1, 2]).is_err());
    }

    #[test]
    fn test_numeric() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestNonarrayArgs::test_transpose (line 353)
        crate::specify_test!("test_numeric");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr = [[1, 2], [3, 4], [5, 6]]
        // tgt = [[1, 3, 5], [2, 4, 6]]
        // assert_equal(np.transpose(arr, (1, 0)), tgt)
        let arr = rt::tensor_from_nested!([[1, 2], [3, 4], [5, 6]], &device);
        let tgt = rt::tensor_from_nested!([[1, 3, 5], [2, 4, 6]], &device);
        assert_equal(rt::transpose(&arr, [1, 0]), &tgt, None);

        // assert_equal(np.transpose(arr, (-1, -2)), tgt)
        assert_equal(rt::transpose(&arr, [-1, -2]), &tgt, None);
    }

    #[test]
    fn test_regression_arr_transpose() {
        // NumPy v2.5.2, _core/tests/test_regression.py, TestRegression::test_arr_transpose (line 786)
        // Ticket #516 - High dimensional transpose
        crate::specify_test!("test_regression_arr_transpose");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.random.rand(*(2,) * 16)
        // x.transpose(list(range(16)))  # Should succeed
        let shape: [usize; 16] = [2; 16];
        let x: Tensor<usize, _> = rt::arange((65536, &device)).into_shape(shape);
        let axes: [isize; 16] = core::array::from_fn(|i| i as isize);
        let _transposed = rt::transpose(&x, axes); // Should succeed
    }
}

#[cfg(test)]
mod numpy_swapaxes {
    use super::*;
    static FUNC: &str = "numpy_swapaxes";

    #[test]
    fn test_numeric() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestNonarrayArgs::test_swapaxes (line 314)
        crate::specify_test!("test_numeric");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // tgt = [[[0, 4], [2, 6]], [[1, 5], [3, 7]]]
        // a = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        // out = np.swapaxes(a, 0, 2)
        // assert_equal(out, tgt)
        let a = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
        let tgt = rt::tensor_from_nested!([[[0, 4], [2, 6]], [[1, 5], [3, 7]]], &device);
        let out = rt::swapaxes(&a, 0, 2);
        assert_equal(&out, &tgt, None);
    }

    #[test]
    fn test_multiarray() {
        // NumPy v2.5.2, _core/tests/test_multiarray.py, TestMethods::test_swapaxes (line 4205)
        crate::specify_test!("test_multiarray");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4).copy()
        let a = rt::arange((24, &device)).into_shape([1, 2, 3, 4]);

        // check exceptions
        // assert_raises(AxisError, a.swapaxes, -5, 0)
        assert!(a.swapaxes_f(-5, 0).is_err());
        // assert_raises(AxisError, a.swapaxes, 4, 0)
        assert!(a.swapaxes_f(4, 0).is_err());
        // assert_raises(AxisError, a.swapaxes, 0, -5)
        assert!(a.swapaxes_f(0, -5).is_err());
        // assert_raises(AxisError, a.swapaxes, 0, 4)
        assert!(a.swapaxes_f(0, 4).is_err());

        // Test various axis combinations. NumPy loops `for i in range(-4, 4) for j in
        // range(-4, 4)` (64 pairs) over both a contiguous source `a` and a non-contiguous
        // `b`, checking shape, elementwise content, and that a view is always returned
        // (gh-5260: `not c.flags['OWNDATA']`). rstsr checks shape + full element grid +
        // view on the contiguous source `a` (the non-contiguous `b` chaining is not ported).
        for i in -4..4 {
            for j in -4..4 {
                let c = a.swapaxes(i, j);
                let i_usize = if i < 0 { (a.ndim() as isize + i) as usize } else { i as usize };
                let j_usize = if j < 0 { (a.ndim() as isize + j) as usize } else { j as usize };

                // check shape: shape[i], shape[j] swapped
                let mut expected_shape: Vec<usize> = a.shape().to_vec();
                expected_shape.swap(i_usize, j_usize);
                assert_eq!(c.shape().to_vec(), expected_shape, "shape mismatch for swapaxes({}, {})", i, j);

                // check array contents (full element grid):
                //   i0, i1, i2, i3 = [dim - 1 for dim in c.shape]
                //   j0, j1, j2, j3 = [dim - 1 for dim in src.shape]
                //   assert_equal(src[idx[j0], idx[j1], idx[j2], idx[j3]],
                //                c[idx[i0], idx[i1], idx[i2], idx[i3]], str((i, j, k)))
                // i.e. c = a.swapaxes(i, j)  =>  c[i0, i1, i2, i3] == a[j0, j1, j2, j3],
                // where the j-index tuple is the i-index tuple with axes i/j swapped back.
                for i0 in 0..c.shape()[0] {
                    for i1 in 0..c.shape()[1] {
                        for i2 in 0..c.shape()[2] {
                            for i3 in 0..c.shape()[3] {
                                let mut j_idx = [i0, i1, i2, i3];
                                j_idx.swap(i_usize, j_usize);
                                let (got, expected) = (
                                    c.i((i0, i1, i2, i3)).to_scalar(),
                                    a.i((j_idx[0], j_idx[1], j_idx[2], j_idx[3])).to_scalar(),
                                );
                                assert_eq!(got, expected, "content mismatch for swapaxes({i}, {j}) at index {j_idx:?}");
                            }
                        }
                    }
                }

                // check a view is always returned, gh-5260 (`not c.flags['OWNDATA']`):
                // swapaxes permutes strides only, so the result must share the source
                // data buffer (no copy) - assert pointer equality.
                assert!(core::ptr::eq(c.as_ptr(), a.as_ptr()), "swapaxes({}, {}) did not return a view", i, j);
            }
        }
    }
}
