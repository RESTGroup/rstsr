#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

// Parity tests for `unstack` (split an array into views along an axis).
//
// Source: NumPy v2.5.2, `_core/tests/test_shape_base.py::test_unstack` (module-level fn).
// rstsr `rt::unstack((&tensor, axis))` mirrors `np.unstack(a, axis=...)`: it returns a
// `Vec<TensorView>` of length `shape[axis]`, where the k-th view is `a` indexed at
// `axis == k` (all other axes kept). numpy returns a tuple; rstsr returns a Vec
// (API-shape difference, intentional - see numpy_differences.md). Defaults: axis=0.

#[cfg(test)]
mod numpy_unstack {
    use super::*;
    static FUNC: &str = "numpy_unstack";

    #[test]
    fn test_unstack() {
        // NumPy v2.5.2, _core/tests/test_shape_base.py, test_unstack (line 531)
        crate::specify_test!("test_unstack");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.arange(24).reshape((2, 3, 4))
        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);

        // unstack(a) / unstack(a, axis=0) / unstack(a, axis=-3): 2 views, stacks[k] == a[k]
        for &axis in &[0isize, -3] {
            let stacks = rt::unstack((&a, axis));
            assert_eq!(stacks.len(), 2, "axis={axis} should yield 2 views");
            assert_equal(&stacks[0], a.i((0, .., ..)), None);
            assert_equal(&stacks[1], a.i((1, .., ..)), None);
        }

        // unstack(a, axis=1) / unstack(a, axis=-2): 3 views, stacks[k] == a[:, k]
        for &axis in &[1isize, -2] {
            let stacks = rt::unstack((&a, axis));
            assert_eq!(stacks.len(), 3, "axis={axis} should yield 3 views");
            assert_equal(&stacks[0], a.i((.., 0, ..)), None);
            assert_equal(&stacks[1], a.i((.., 1, ..)), None);
            assert_equal(&stacks[2], a.i((.., 2, ..)), None);
        }

        // unstack(a, axis=2) / unstack(a, axis=-1): 4 views, stacks[k] == a[:, :, k]
        for &axis in &[2isize, -1] {
            let stacks = rt::unstack((&a, axis));
            assert_eq!(stacks.len(), 4, "axis={axis} should yield 4 views");
            assert_equal(&stacks[0], a.i((.., .., 0)), None);
            assert_equal(&stacks[1], a.i((.., .., 1)), None);
            assert_equal(&stacks[2], a.i((.., .., 2)), None);
            assert_equal(&stacks[3], a.i((.., .., 3)), None);
        }

        // assert_raises(ValueError, np.unstack, a, axis=3)
        // assert_raises(ValueError, np.unstack, a, axis=-4)
        assert!(rt::unstack_f((&a, 3isize)).is_err());
        assert!(rt::unstack_f((&a, -4isize)).is_err());

        // assert_raises(ValueError, np.unstack, np.array(0), axis=0)  # 0-D input
        let z: Tensor<i32, _> = rt::asarray((0, &device));
        assert!(rt::unstack_f((&z, 0isize)).is_err());
    }
}
