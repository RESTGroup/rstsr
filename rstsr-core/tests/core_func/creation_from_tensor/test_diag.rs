#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

// Parity tests for `diag` (extract a diagonal or construct a diagonal tensor).
//
// Source: NumPy v2.5.2, `lib/tests/test_twodim_base.py::TestDiag`.
// rstsr's `rt::diag((&tensor, offset))` mirrors `np.diag(a, k)`: 1-D -> 2-D diagonal
// matrix with the input on the k-th diagonal; 2-D -> 1-D extraction of the k-th
// diagonal. All five TestDiag cases transfer (no dtype/out kwargs to skip here).

#[cfg(test)]
mod numpy_diag {
    use super::*;
    static FUNC: &str = "numpy_diag";

    #[test]
    fn test_vector() {
        // NumPy v2.5.2, lib/tests/test_twodim_base.py, TestDiag::test_vector (line 131)
        crate::specify_test!("test_vector");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // vals = (100 * arange(5)).astype('l') = [0, 100, 200, 300, 400]
        let vals = rt::tensor_from_nested!([0, 100, 200, 300, 400], &device);

        // diag(vals) -> 5x5 with vals on the main diagonal
        let b = rt::diag(&vals);
        let expected = rt::tensor_from_nested!(
            [[0, 0, 0, 0, 0], [0, 100, 0, 0, 0], [0, 0, 200, 0, 0], [0, 0, 0, 300, 0], [0, 0, 0, 0, 400],],
            &device
        );
        assert_equal(&b, &expected, None);

        // diag(vals, k=2) -> 7x7 with vals on the +2 (super) diagonal: b[k, k+2] = vals[k]
        let b = rt::diag((&vals, 2));
        let expected = rt::tensor_from_nested!(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 100, 0, 0, 0],
                [0, 0, 0, 0, 200, 0, 0],
                [0, 0, 0, 0, 0, 300, 0],
                [0, 0, 0, 0, 0, 0, 400],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            &device
        );
        assert_equal(&b, &expected, None);

        // diag(vals, k=-2) -> 7x7 with vals on the -2 (sub) diagonal: c[k+2, k] = vals[k]
        let c = rt::diag((&vals, -2));
        let expected = rt::tensor_from_nested!(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 100, 0, 0, 0, 0, 0],
                [0, 0, 200, 0, 0, 0, 0],
                [0, 0, 0, 300, 0, 0, 0],
                [0, 0, 0, 0, 400, 0, 0],
            ],
            &device
        );
        assert_equal(&c, &expected, None);
    }

    #[test]
    fn test_matrix() {
        // NumPy v2.5.2, lib/tests/test_twodim_base.py, TestDiag::test_matrix (line 145)
        crate::specify_test!("test_matrix");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // vals = (100 * get_mat(5) + 1).astype('l'), where get_mat(n)[i, j] = i + j.
        // => vals[i, j] = 100 * (i + j) + 1.
        let vals = rt::tensor_from_nested!(
            [
                [1, 101, 201, 301, 401],
                [101, 201, 301, 401, 501],
                [201, 301, 401, 501, 601],
                [301, 401, 501, 601, 701],
                [401, 501, 601, 701, 801],
            ],
            &device
        );

        // diag(vals) -> main diagonal [1, 201, 401, 601, 801]
        let b = rt::diag(&vals);
        let expected = rt::tensor_from_nested!([1, 201, 401, 601, 801], &device);
        assert_equal(&b, &expected, None);

        // diag(vals, 2) -> +2 diagonal: vals[k, k+2] for k in 0..3 = [201, 401, 601]
        let b = rt::diag((&vals, 2));
        let expected = rt::tensor_from_nested!([201, 401, 601], &device);
        assert_equal(&b, &expected, None);

        // diag(vals, -2) -> -2 diagonal: vals[k+2, k] for k in 0..3 = [201, 401, 601]
        let c = rt::diag((&vals, -2));
        let expected = rt::tensor_from_nested!([201, 401, 601], &device);
        assert_equal(&c, &expected, None);
    }

    #[test]
    fn test_fortran_order() {
        // NumPy v2.5.2, lib/tests/test_twodim_base.py, TestDiag::test_fortran_order (line 160)
        crate::specify_test!("test_fortran_order");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // vals = array(...100*get_mat(5)+1..., order='F') - same values, F-contiguous.
        // diag must be layout-agnostic: the extracted diagonal is identical.
        let vals = rt::tensor_from_nested!(
            [
                [1, 101, 201, 301, 401],
                [101, 201, 301, 401, 501],
                [201, 301, 401, 501, 601],
                [301, 401, 501, 601, 701],
                [401, 501, 601, 701, 801],
            ],
            &device
        );
        let vals_f = vals.into_contig(ColMajor); // F-contiguous, same values
        assert!(vals_f.f_contig());
        let b = rt::diag(&vals_f);
        let expected = rt::tensor_from_nested!([1, 201, 401, 601, 801], &device);
        assert_equal(&b, &expected, None);
    }

    #[test]
    fn test_diag_bounds() {
        // NumPy v2.5.2, lib/tests/test_twodim_base.py, TestDiag::test_diag_bounds (line 164)
        crate::specify_test!("test_diag_bounds");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // A = [[1, 2], [3, 4], [5, 6]]  (3x2)
        let a = rt::tensor_from_nested!([[1, 2], [3, 4], [5, 6]], &device);

        // diag(A, k=2) == []  (out of bounds)
        assert_eq!(rt::diag((&a, 2)).shape(), &[0]);
        // diag(A, k=1) == [2]
        assert_equal(rt::diag((&a, 1)), &rt::tensor_from_nested!([2], &device), None);
        // diag(A, k=0) == [1, 4]
        assert_equal(rt::diag(&a), &rt::tensor_from_nested!([1, 4], &device), None);
        // diag(A, k=-1) == [3, 6]
        assert_equal(rt::diag((&a, -1)), &rt::tensor_from_nested!([3, 6], &device), None);
        // diag(A, k=-2) == [5]
        assert_equal(rt::diag((&a, -2)), &rt::tensor_from_nested!([5], &device), None);
        // diag(A, k=-3) == []
        assert_eq!(rt::diag((&a, -3)).shape(), &[0]);
    }

    #[test]
    fn test_failure() {
        // NumPy v2.5.2, lib/tests/test_twodim_base.py, TestDiag::test_failure (line 173)
        crate::specify_test!("test_failure");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // assert_raises(ValueError, diag, [[[1]]])  # 3-D input
        let a = rt::tensor_from_nested!([[[1]]], &device);
        assert!(rt::diag_f(&a).is_err());
    }
}
