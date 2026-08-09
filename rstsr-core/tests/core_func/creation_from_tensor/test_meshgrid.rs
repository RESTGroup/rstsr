#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

// Parity tests for `meshgrid` (coordinate matrices from coordinate vectors).
//
// Source: NumPy v2.5.2, `lib/tests/test_function_base.py::TestMeshgrid`.
// rstsr `rt::meshgrid((&vec_of_refs, indexing, copy))` mirrors `np.meshgrid` with
// `indexing` in {"xy", "ij"} (default "xy") and a `copy: bool`. It returns a
// `Vec<Tensor>` (not a Python tuple). It has NO `sparse=` parameter and is
// homogeneous-dtype (all inputs must share one `T`).
//
// Not ported (N/A, divergence - see numpy_differences.md):
//   - test_sparse              - rstsr meshgrid has no `sparse=` parameter
//   - test_always_tuple        - returns Vec, not tuple; coupled to sparse
//   - test_invalid_arguments   - Python kwargs (`indices='ij'` typo)
//   - test_return_type         - rstsr is homogeneous-dtype (numpy preserves a per-input dtype:
//     x=f32 -> X=f32, y=f64 -> Y=f64)
//   - test_writeback           - Python view-vs-copy mutation semantics; rstsr `copy` controls
//     contig-copy vs broadcast, both owned

#[cfg(test)]
mod numpy_meshgrid {
    use super::*;
    static FUNC: &str = "numpy_meshgrid";

    #[test]
    fn test_simple() {
        // NumPy v2.5.2, lib/tests/test_function_base.py, TestMeshgrid::test_simple (line 2769)
        crate::specify_test!("test_simple");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7])  # indexing='xy' default
        let x = rt::tensor_from_nested!([1, 2, 3], &device);
        let y = rt::tensor_from_nested!([4, 5, 6, 7], &device);
        let r = rt::meshgrid((&vec![&x, &y], "xy", true));
        let ex = rt::tensor_from_nested!([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], &device);
        let ey = rt::tensor_from_nested!([[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]], &device);
        assert_equal(&r[0], &ex, None);
        assert_equal(&r[1], &ey, None);
    }

    #[test]
    fn test_single_input() {
        // NumPy v2.5.2, lib/tests/test_function_base.py, TestMeshgrid::test_single_input (line 2780)
        crate::specify_test!("test_single_input");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // [X] = meshgrid([1, 2, 3, 4]); assert_array_equal(X, [1, 2, 3, 4])
        let x = rt::tensor_from_nested!([1, 2, 3, 4], &device);
        let r = rt::meshgrid((&vec![&x], "xy", true));
        assert_eq!(r.len(), 1);
        assert_equal(&r[0], &rt::tensor_from_nested!([1, 2, 3, 4], &device), None);
    }

    #[test]
    fn test_no_input() {
        // NumPy v2.5.2, lib/tests/test_function_base.py, TestMeshgrid::test_no_input (line 2784)
        crate::specify_test!("test_no_input");

        // args = []; assert_array_equal([], meshgrid(*args))
        // rstsr: empty input -> empty output Vec.
        let inputs: Vec<&Tensor<i64, DeviceType>> = vec![];
        let r = rt::meshgrid((&inputs, "xy", true));
        assert!(r.is_empty());
    }

    #[test]
    fn test_indexing() {
        // NumPy v2.5.2, lib/tests/test_function_base.py, TestMeshgrid::test_indexing (line 2789)
        crate::specify_test!("test_indexing");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = [1, 2, 3]; y = [4, 5, 6, 7]
        // [X, Y] = meshgrid(x, y, indexing='ij')
        let x = rt::tensor_from_nested!([1, 2, 3], &device);
        let y = rt::tensor_from_nested!([4, 5, 6, 7], &device);
        let r = rt::meshgrid((&vec![&x, &y], "ij", true));
        let ex = rt::tensor_from_nested!([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], &device);
        let ey = rt::tensor_from_nested!([[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]], &device);
        assert_equal(&r[0], &ex, None);
        assert_equal(&r[1], &ey, None);

        // Test expected shapes (xy default swaps the first two axes; ij does not).
        // z = [8, 9]
        let z = rt::tensor_from_nested!([8, 9], &device);
        // assert_(meshgrid(x, y)[0].shape == (4, 3))
        assert_eq!(rt::meshgrid((&vec![&x, &y], "xy", true))[0].shape(), &[4, 3]);
        // assert_(meshgrid(x, y, indexing='ij')[0].shape == (3, 4))
        assert_eq!(rt::meshgrid((&vec![&x, &y], "ij", true))[0].shape(), &[3, 4]);
        // assert_(meshgrid(x, y, z)[0].shape == (4, 3, 2))
        assert_eq!(rt::meshgrid((&vec![&x, &y, &z], "xy", true))[0].shape(), &[4, 3, 2]);
        // assert_(meshgrid(x, y, z, indexing='ij')[0].shape == (3, 4, 2))
        assert_eq!(rt::meshgrid((&vec![&x, &y, &z], "ij", true))[0].shape(), &[3, 4, 2]);

        // assert_raises(ValueError, meshgrid, x, y, indexing='notvalid')
        assert!(rt::meshgrid_f((&vec![&x, &y], "notvalid", true)).is_err());
    }

    #[test]
    fn test_nd_shape() {
        // NumPy v2.5.2, lib/tests/test_function_base.py, TestMeshgrid::test_nd_shape (line 2861)
        crate::specify_test!("test_nd_shape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a, b, c, d, e = np.meshgrid(*([0] * i for i in range(1, 6)))
        //   inputs of length 1, 2, 3, 4, 5; xy default swaps the first two -> (2, 1, 3, 4, 5)
        // expected_shape = (2, 1, 3, 4, 5)
        let t1 = rt::tensor_from_nested!([0], &device);
        let t2 = rt::tensor_from_nested!([0, 0], &device);
        let t3 = rt::tensor_from_nested!([0, 0, 0], &device);
        let t4 = rt::tensor_from_nested!([0, 0, 0, 0], &device);
        let t5 = rt::tensor_from_nested!([0, 0, 0, 0, 0], &device);
        let r = rt::meshgrid((&vec![&t1, &t2, &t3, &t4, &t5], "xy", true));
        for t in &r {
            assert_eq!(t.shape(), &[2, 1, 3, 4, 5]);
        }
    }

    #[test]
    fn test_nd_values() {
        // NumPy v2.5.2, lib/tests/test_function_base.py, TestMeshgrid::test_nd_values (line 2870)
        crate::specify_test!("test_nd_values");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5])  # xy default -> (2, 1, 3)
        let a = rt::tensor_from_nested!([0], &device);
        let b = rt::tensor_from_nested!([1, 2], &device);
        let c = rt::tensor_from_nested!([3, 4, 5], &device);
        let r = rt::meshgrid((&vec![&a, &b, &c], "xy", true));
        let ea = rt::tensor_from_nested!([[[0, 0, 0]], [[0, 0, 0]]], &device);
        let eb = rt::tensor_from_nested!([[[1, 1, 1]], [[2, 2, 2]]], &device);
        let ec = rt::tensor_from_nested!([[[3, 4, 5]], [[3, 4, 5]]], &device);
        assert_equal(&r[0], &ea, None);
        assert_equal(&r[1], &eb, None);
        assert_equal(&r[2], &ec, None);
    }

    #[test]
    fn test_nd_indexing() {
        // NumPy v2.5.2, lib/tests/test_function_base.py, TestMeshgrid::test_nd_indexing (line 2876)
        crate::specify_test!("test_nd_indexing");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5], indexing='ij')  -> (1, 2, 3)
        let a = rt::tensor_from_nested!([0], &device);
        let b = rt::tensor_from_nested!([1, 2], &device);
        let c = rt::tensor_from_nested!([3, 4, 5], &device);
        let r = rt::meshgrid((&vec![&a, &b, &c], "ij", true));
        let ea = rt::tensor_from_nested!([[[0, 0, 0], [0, 0, 0]]], &device);
        let eb = rt::tensor_from_nested!([[[1, 1, 1], [2, 2, 2]]], &device);
        let ec = rt::tensor_from_nested!([[[3, 4, 5], [3, 4, 5]]], &device);
        assert_equal(&r[0], &ea, None);
        assert_equal(&r[1], &eb, None);
        assert_equal(&r[2], &ec, None);
    }
}
