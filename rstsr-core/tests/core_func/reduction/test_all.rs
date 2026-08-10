#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_all {
    use super::*;
    static FUNC: &str = "numpy_all";

    #[test]
    fn test_basic() {
        // NumPy v2.5.2, lib/tests/test_function_base.py, TestAll::test_basic (line 283)
        // NumPy uses int 0/1 as truthy; rstsr uses bool tensors (semantically identical).
        crate::specify_test!("test_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // y1 = [0, 1, 1, 0]; y2 = [0, 0, 0, 0]; y3 = [1, 1, 1, 1]
        // assert_(not np.all(y1))
        // assert_(np.all(y3))
        // assert_(not np.all(y2))
        // assert_(np.all(~np.array(y2)))
        let y1 = rt::tensor_from_nested!([false, true, true, false], &device);
        let y2 = rt::tensor_from_nested!([false, false, false, false], &device);
        let y3 = rt::tensor_from_nested!([true, true, true, true], &device);
        assert!(!y1.all_all());
        assert!(y3.all_all());
        assert!(!y2.all_all());
        // np.all(~np.array(y2))  -> bitwise-not of all-false -> all-true
        assert!(rt::not(&y2).all_all());
    }

    #[test]
    fn test_nd() {
        // NumPy v2.5.2, lib/tests/test_function_base.py, TestAll::test_nd (line 292)
        crate::specify_test!("test_nd");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        // assert_(not np.all(y1))
        // assert_array_equal(np.all(y1, axis=0), [0, 0, 1])
        // assert_array_equal(np.all(y1, axis=1), [0, 0, 1])
        let y1 = rt::tensor_from_nested!([[false, false, true], [false, true, true], [true, true, true]], &device);
        assert!(!y1.all_all());
        // bool result tensors can't use assert_equal (bool: ExtNum unsatisfied);
        // compare via to_vec() instead.
        assert_eq!(y1.all_axes(0).to_vec(), vec![false, false, true]);
        assert_eq!(y1.all_axes(1).to_vec(), vec![false, false, true]);
    }
}
