#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_matrix_transpose {
    use super::*;
    static FUNC: &str = "numpy_matrix_transpose";

    #[test]
    fn test_raises_error_for_1d() {
        // NumPy v2.5.2, _core/tests/test_arrayobject.py, test_matrix_transpose_raises_error_for_1d (line
        // 10)
        crate::specify_test!("test_raises_error_for_1d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr = np.arange(48); arr.mT -> ValueError ("ndim < 2 is undefined")
        let arr = rt::arange((48, &device));
        assert!(arr.matrix_transpose_f().is_err());
    }

    #[test]
    fn test_equals_transpose_2d() {
        // NumPy v2.5.2, _core/tests/test_arrayobject.py, test_matrix_transpose_equals_transpose_2d (line
        // 17)
        crate::specify_test!("test_equals_transpose_2d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr = np.arange(48).reshape((6, 8)); assert_array_equal(arr.T, arr.mT)
        let arr = rt::arange((48, &device)).into_shape([6, 8]);
        assert_equal(&arr.t(), &arr.matrix_transpose(), None);
    }

    #[test]
    fn test_equals_swapaxes() {
        // NumPy v2.5.2, _core/tests/test_arrayobject.py, test_matrix_transpose_equals_swapaxes (line 30)
        // mT swaps the last two axes; equivalent to swapaxes(-2, -1). Parametrized over
        // (5,2), (5,2,3), (5,2,3,4) in NumPy.
        crate::specify_test!("test_equals_swapaxes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        for shape in [vec![5, 2], vec![5, 2, 3], vec![5, 2, 3, 4]] {
            let n: usize = shape.iter().product();
            let arr = rt::arange((n, &device)).into_shape(shape.clone());
            let mt = arr.matrix_transpose();
            let swp = arr.swapaxes(-2, -1);
            assert_equal(&mt, &swp, None);
        }
    }
}
