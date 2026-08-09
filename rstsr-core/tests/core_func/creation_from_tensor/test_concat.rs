#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

// Parity tests for `concat` / `concatenate` (joining arrays along an existing axis).
//
// Source: NumPy v2.5.2, `_core/tests/test_shape_base.py::TestConcatenate`.
// rstsr's `concat` is exposed as `rt::concat` with a `rt::concatenate` alias; it
// always allocates a fresh tensor (copy semantics), and takes an explicit integer
// axis (no `axis=None` flatten mode, no `out=` / `dtype=` / `casting=` kwargs).
//
// Not ported (N/A, Rust typed API / no equivalent):
//   test_huge_list_error        - requires_memory, 64-bit-only stress
//   test_concatenate_axis_None   - rstsr concat has no `axis=None` flatten mode (see
//                                  numpy_differences.md)
//   test_large_concatenate_axis_None - same
//   test_concatenate_same_value - invalid `casting="same_value"` kwarg
//   test_operator_concat        - Python `operator.concat`
//   test_bad_out_shape          - `out=` parameter
//   test_out_and_dtype          - `out=` / `dtype=` / `casting`
//   test_dtype_with_promotion   - `dtype=` promotion
//   test_string_dtype_does_not_inspect - `dtype="S"/"U"`
//   test_subarray_error         - subarray dtype

#[cfg(test)]
mod numpy_concatenate {
    use super::*;
    static FUNC: &str = "numpy_concatenate";

    #[test]
    fn test_returns_copy() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestConcatenate::test_returns_copy (line 249)
        crate::specify_test!("test_returns_copy");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.eye(3)
        // b = np.concatenate([a])
        // b[0, 0] = 2
        // assert b[0, 0] != a[0, 0]
        //
        // rstsr concat always allocates (empty + assign), so the result is a copy
        // that never shares storage with the input. Pointer inequality is the rstsr
        // equivalent of numpy's mutate-and-check.
        let a: Tensor<f64, _> = rt::ones(([3, 3], &device));
        let b = rt::concatenate(vec![&a]);
        assert!(!core::ptr::eq(b.as_ptr(), a.as_ptr()));
    }

    #[test]
    fn test_exceptions() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestConcatenate::test_exceptions (line 255)
        crate::specify_test!("test_exceptions");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // test axis must be in bounds
        // for ndim in [1, 2, 3]:
        //     a = np.ones((1,) * ndim)
        //     np.concatenate((a, a), axis=0)  # OK
        //     assert_raises(AxisError, np.concatenate, (a, a), axis=ndim)
        //     assert_raises(AxisError, np.concatenate, (a, a), axis=-(ndim + 1))
        // Unrolled over ndim (rstsr shape args are array literals, not runtime vecs).
        // ndim = 1
        let a: Tensor<f64, _> = rt::ones(([1], &device));
        let _ok = rt::concatenate((vec![&a, &a], 0isize));
        assert!(rt::concatenate_f((vec![&a, &a], 1isize)).is_err());
        assert!(rt::concatenate_f((vec![&a, &a], -2isize)).is_err());
        // ndim = 2
        let a: Tensor<f64, _> = rt::ones(([1, 1], &device));
        let _ok = rt::concatenate((vec![&a, &a], 0isize));
        assert!(rt::concatenate_f((vec![&a, &a], 2isize)).is_err());
        assert!(rt::concatenate_f((vec![&a, &a], -3isize)).is_err());
        // ndim = 3
        let a: Tensor<f64, _> = rt::ones(([1, 1, 1], &device));
        let _ok = rt::concatenate((vec![&a, &a], 0isize));
        assert!(rt::concatenate_f((vec![&a, &a], 3isize)).is_err());
        assert!(rt::concatenate_f((vec![&a, &a], -4isize)).is_err());

        // Scalars cannot be concatenated
        // assert_raises(ValueError, concatenate, (0,))
        // assert_raises(ValueError, concatenate, (np.array(0),))
        let z: Tensor<i32, _> = rt::asarray((0, &device));
        assert!(rt::concatenate_f((vec![&z], 0isize)).is_err());

        // dimensionality must match
        // assert_raises(ValueError, np.concatenate, (np.zeros(1), np.zeros((1, 1))))
        let a1: Tensor<f64, _> = rt::zeros(([1], &device));
        let a2: Tensor<f64, _> = rt::zeros(([1, 1], &device));
        assert!(rt::concatenate_f((vec![&a1, &a2], 0isize)).is_err());

        // test shapes must match except for concatenation axis.
        // a = np.ones((1, 2, 3)); b = np.ones((2, 2, 3))
        // numpy rotates the mismatched axis via moveaxis; collapsed here to the
        // representative fixed cases: concat along the matching axis succeeds, along
        // a mismatched axis (dimension 0: 1 vs 2) errors.
        let a: Tensor<f64, _> = rt::ones(([1, 2, 3], &device));
        let b: Tensor<f64, _> = rt::ones(([2, 2, 3], &device));
        // axis 0 (sizes 1+2, others match) OK
        let _ok = rt::concatenate((vec![&a, &b], 0isize));
        // axis 1 / axis 2 -> dimension 0 (1 vs 2) mismatch -> err
        assert!(rt::concatenate_f((vec![&a, &b], 1isize)).is_err());
        assert!(rt::concatenate_f((vec![&a, &b], 2isize)).is_err());

        // No arrays to concatenate raises ValueError
        // assert_raises(ValueError, concatenate, ())
        let empty: Vec<Tensor<i64, DeviceType>> = vec![];
        assert!(rt::concatenate_f((empty, 0isize)).is_err());
    }

    #[test]
    fn test_concatenate() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, TestConcatenate::test_concatenate (line 344)
        crate::specify_test!("test_concatenate");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // One sequence returns unmodified (but as array)
        // r4 = list(range(4)); assert_array_equal(concatenate((r4,)), r4)
        let r4 = rt::tensor_from_nested!([0, 1, 2, 3], &device);
        assert_equal(rt::concatenate(vec![&r4]), &r4, None);

        // 1D default concatenation
        // r3 = list(range(3)); assert_array_equal(concatenate((r4, r3)), r4 + r3)
        let r3 = rt::tensor_from_nested!([0, 1, 2], &device);
        let expected = rt::tensor_from_nested!([0, 1, 2, 3, 0, 1, 2], &device);
        assert_equal(rt::concatenate(vec![&r4, &r3]), &expected, None);

        // Explicit axis specification, including negative
        // assert_array_equal(concatenate((r4, r3), 0), r4 + r3)
        // assert_array_equal(concatenate((r4, r3), -1), r4 + r3)
        assert_equal(rt::concatenate((vec![&r4, &r3], 0)), &expected, None);
        assert_equal(rt::concatenate((vec![&r4, &r3], -1)), &expected, None);

        // 2D
        // a23 = array([[10, 11, 12], [13, 14, 15]])
        // a13 = array([[0, 1, 2]])
        // res = array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        let a23 = rt::tensor_from_nested!([[10, 11, 12], [13, 14, 15]], &device);
        let a13 = rt::tensor_from_nested!([[0, 1, 2]], &device);
        let res = rt::tensor_from_nested!([[10, 11, 12], [13, 14, 15], [0, 1, 2]], &device);
        assert_equal(rt::concatenate(vec![&a23, &a13]), &res, None);
        assert_equal(rt::concatenate((vec![&a23, &a13], 0)), &res, None);
        // assert_array_equal(concatenate((a23.T, a13.T), 1), res.T)
        // assert_array_equal(concatenate((a23.T, a13.T), -1), res.T)
        let a23t = a23.t();
        let a13t = a13.t();
        let rest = res.t();
        assert_equal(rt::concatenate((vec![&a23t, &a13t], 1)), &rest, None);
        assert_equal(rt::concatenate((vec![&a23t, &a13t], -1)), &rest, None);
        // Arrays must match shape
        // assert_raises(ValueError, concatenate, (a23.T, a13.T), 0)
        assert!(rt::concatenate_f((vec![&a23t, &a13t], 0)).is_err());

        // 3D
        // res = arange(2 * 3 * 7).reshape((2, 3, 7))
        // a0 = res[..., :4]; a1 = res[..., 4:6]; a2 = res[..., 6:]
        // assert_array_equal(concatenate((a0, a1, a2), 2), res)
        // assert_array_equal(concatenate((a0, a1, a2), -1), res)
        // assert_array_equal(concatenate((a0.T, a1.T, a2.T), 0), res.T)
        let res3 = rt::arange((42, &device)).into_shape([2, 3, 7]);
        let a0 = res3.i((.., .., ..4));
        let a1 = res3.i((.., .., 4..6));
        let a2 = res3.i((.., .., 6..));
        assert_equal(rt::concatenate((vec![&a0, &a1, &a2], 2)), &res3, None);
        assert_equal(rt::concatenate((vec![&a0, &a1, &a2], -1)), &res3, None);
        let a0t = a0.t();
        let a1t = a1.t();
        let a2t = a2.t();
        let res3t = res3.t();
        assert_equal(rt::concatenate((vec![&a0t, &a1t, &a2t], 0)), &res3t, None);

        // NOTE: numpy's `out = res.copy(); rout = concatenate(..., out=out)` cases
        // have no rstsr equivalent (Rust typed API) and are not ported.
    }
}
