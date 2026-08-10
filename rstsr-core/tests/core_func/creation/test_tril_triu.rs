#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_tril_triu {
    use super::*;
    static FUNC: &str = "numpy_tril_triu";

    #[test]
    fn test_ndim2() {
        // NumPy v2.5.2, lib/tests/test_twodim_base.py, test_tril_triu_ndim2 (line 335)
        crate::specify_test!("test_ndim2");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.ones((2, 2))
        // b = np.tril(a); c = np.triu(a)
        // assert_array_equal(b, [[1, 0], [1, 1]])
        // assert_array_equal(c, b.T)  i.e. [[1, 1], [0, 1]]
        let a: Tensor<i32, _> = rt::ones(([2, 2], &device));
        let b = rt::tril((&a, 0));
        assert_equal(&b, &rt::tensor_from_nested!([[1, 0], [1, 1]], &device), None);
        let c = rt::triu((&a, 0));
        assert_equal(&c, &rt::tensor_from_nested!([[1, 1], [0, 1]], &device), None);
    }

    #[test]
    fn test_ndim3() {
        // NumPy v2.5.2, lib/tests/test_twodim_base.py, test_tril_triu_ndim3 (line 347)
        crate::specify_test!("test_ndim3");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = [[[1,1],[1,1]], [[1,1],[1,0]], [[1,1],[0,0]]]
        let a = rt::tensor_from_nested!([[[1, 1], [1, 1]], [[1, 1], [1, 0]], [[1, 1], [0, 0]]], &device);
        // a_tril_desired = [[[1,0],[1,1]], [[1,0],[1,0]], [[1,0],[0,0]]]
        let tril_desired = rt::tensor_from_nested!([[[1, 0], [1, 1]], [[1, 0], [1, 0]], [[1, 0], [0, 0]]], &device);
        // a_triu_desired = [[[1,1],[0,1]], [[1,1],[0,0]], [[1,1],[0,0]]]
        let triu_desired = rt::tensor_from_nested!([[[1, 1], [0, 1]], [[1, 1], [0, 0]], [[1, 1], [0, 0]]], &device);
        assert_equal(&rt::tril((&a, 0)), &tril_desired, None);
        assert_equal(&rt::triu((&a, 0)), &triu_desired, None);
    }

    #[test]
    fn test_with_offset() {
        crate::specify_test!("test_with_offset");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.tril(ones((3,3)), k=1) keeps the first superdiagonal too:
        // [[1,1,0],[1,1,1],[1,1,1]] (k=1 lower-includes the superdiagonal)
        let a: Tensor<i32, _> = rt::ones(([3, 3], &device));
        let tril1 = rt::tril((&a, 1));
        assert_equal(&tril1, &rt::tensor_from_nested!([[1, 1, 0], [1, 1, 1], [1, 1, 1]], &device), None);
        // np.triu(ones((3,3)), k=-1):
        // [[1,1,1],[1,1,1],[0,1,1]]
        let triu_neg1 = rt::triu((&a, -1));
        assert_equal(&triu_neg1, &rt::tensor_from_nested!([[1, 1, 1], [1, 1, 1], [0, 1, 1]], &device), None);
    }
}
