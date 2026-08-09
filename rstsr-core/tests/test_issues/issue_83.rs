// Issue #83: `rt::vstack([v1, v2])` for 1-D vectors stacked to 1-D instead of 2-D,
// diverging from NumPy `vstack` (which yields 2-D `(2, N)`). The reported workaround
// was `rt::stack(([v1, v2], -1))`; that is actually the *transpose* of `vstack`
// (a new last axis -> `(N, 2)`, i.e. a column-stack), not `vstack` - the
// vstack-equivalent workaround is `stack(([v1, v2], 0))` (new axis 0 -> `(2, N)`).
// Fixed: `vstack` now `atleast_2d`-promotes each input (1-D `(N,)` -> `(1, N)`)
// before `concat` axis 0, so `vstack` itself returns 2-D and neither `stack`
// workaround is needed.

#[test]
pub fn issue_83() {
    use crate::test_utils::*;

    let mut device = crate::TESTCFG.device.clone();
    device.set_default_order(RowMajor);

    let v1 = rt::tensor_from_nested!([1, 2, 3], &device);
    let v2 = rt::tensor_from_nested!([4, 5, 6], &device);

    // np.vstack([v1, v2]) == [[1, 2, 3], [4, 5, 6]]  (shape (2, 3))
    // Previously rstsr returned the 1-D [1, 2, 3, 4, 5, 6] (shape (6,)).
    let res = rt::vstack([&v1, &v2]);
    assert_eq!(res.shape(), &[2, 3]);
    let expected = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
    assert_equal(&res, &expected, None);

    // stack axis 0 == vstack (the correct workaround): new axis 0 -> (1, N), concat
    // axis 0 -> (2, N). Still works after the `stack` change.
    let stacked0 = rt::stack((vec![&v1, &v2], 0isize));
    assert_eq!(stacked0.shape(), &[2, 3]);
    assert_equal(&stacked0, &expected, None);

    // stack axis -1 is the transpose (column-stack): new last axis -> (N, 1), concat
    // axis 1 -> (N, 2). This is the result the issue's `-1` "workaround" actually
    // produced - 2-D, but NOT `vstack`.
    let stacked_neg = rt::stack((vec![&v1, &v2], -1isize));
    assert_eq!(stacked_neg.shape(), &[3, 2]);
    let expected_neg = rt::tensor_from_nested!([[1, 4], [2, 5], [3, 6]], &device);
    assert_equal(&stacked_neg, &expected_neg, None);
}
