use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_broadcast_shapes {
    use super::*;
    static FUNC: &str = "numpy_broadcast_shapes";

    #[test]
    fn test_broadcast_shapes_succeeds() {
        // NumPy v2.4.2, lib/tests/test_stride_tricks.py, test_broadcast_shapes_succeeds (line 304)
        crate::specify_test!("test_broadcast_shapes_succeeds");

        // Data format: (input_shapes, target_shape)
        // input_shapes is a list of shapes, where each shape is a tuple
        let data: Vec<(Vec<Vec<usize>>, Vec<usize>)> = vec![
            // [[], ()]
            (vec![], vec![]),
            // [[()], ()]
            (vec![vec![]], vec![]),
            // [[(7,)], (7,)]
            (vec![vec![7]], vec![7]),
            // [[(1, 2), (2,)], (1, 2)]
            (vec![vec![1, 2], vec![2]], vec![1, 2]),
            // [[(1, 1)], (1, 1)]
            (vec![vec![1, 1]], vec![1, 1]),
            // [[(1, 1), (3, 4)], (3, 4)]
            (vec![vec![1, 1], vec![3, 4]], vec![3, 4]),
            // [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)]
            (vec![vec![6, 7], vec![5, 6, 1], vec![7], vec![5, 1, 7]], vec![5, 6, 7]),
            // [[(5, 6, 1)], (5, 6, 1)]
            (vec![vec![5, 6, 1]], vec![5, 6, 1]),
            // [[(1, 3), (3, 1)], (3, 3)]
            (vec![vec![1, 3], vec![3, 1]], vec![3, 3]),
            // [[(1, 0), (0, 0)], (0, 0)]
            (vec![vec![1, 0], vec![0, 0]], vec![0, 0]),
            // [[(0, 1), (0, 0)], (0, 0)]
            (vec![vec![0, 1], vec![0, 0]], vec![0, 0]),
            // [[(1, 0), (0, 1)], (0, 0)]
            (vec![vec![1, 0], vec![0, 1]], vec![0, 0]),
            // [[(1, 1), (0, 0)], (0, 0)]
            (vec![vec![1, 1], vec![0, 0]], vec![0, 0]),
            // [[(1, 1), (1, 0)], (1, 0)]
            (vec![vec![1, 1], vec![1, 0]], vec![1, 0]),
            // [[(1, 1), (0, 1)], (0, 1)]
            (vec![vec![1, 1], vec![0, 1]], vec![0, 1]),
            // [[(), (0,)], (0,)]
            (vec![vec![], vec![0]], vec![0]),
            // [[(0,), (0, 0)], (0, 0)]
            (vec![vec![0], vec![0, 0]], vec![0, 0]),
            // [[(0,), (0, 1)], (0, 0)]
            (vec![vec![0], vec![0, 1]], vec![0, 0]),
            // [[(1,), (0, 0)], (0, 0)]
            (vec![vec![1], vec![0, 0]], vec![0, 0]),
            // [[(), (0, 0)], (0, 0)]
            (vec![vec![], vec![0, 0]], vec![0, 0]),
            // [[(1, 1), (0,)], (1, 0)]
            (vec![vec![1, 1], vec![0]], vec![1, 0]),
            // [[(1,), (0, 1)], (0, 1)]
            (vec![vec![1], vec![0, 1]], vec![0, 1]),
            // [[(1,), (1, 0)], (1, 0)]
            (vec![vec![1], vec![1, 0]], vec![1, 0]),
            // [[(), (1, 0)], (1, 0)]
            (vec![vec![], vec![1, 0]], vec![1, 0]),
            // [[(), (0, 1)], (0, 1)]
            (vec![vec![], vec![0, 1]], vec![0, 1]),
            // [[(1,), (3,)], (3,)]
            (vec![vec![1], vec![3]], vec![3]),
            // [[2, (3, 2)], (3, 2)]
            (vec![vec![2], vec![3, 2]], vec![3, 2]),
        ];

        for (input_shapes, target_shape) in data {
            let result = rt::broadcast_shapes(&input_shapes, RowMajor);
            assert_eq!(
                result, target_shape,
                "Failed for input_shapes: {:?}, expected: {:?}, got: {:?}",
                input_shapes, target_shape, result
            );
        }

        // Test with many shapes: (1, 2) repeated 32 times -> (1, 2)
        let shapes: Vec<IxD> = (0..32).map(|_| vec![1, 2]).collect();
        let result = rt::broadcast_shapes(&shapes, RowMajor);
        assert_eq!(result, vec![1, 2]);

        // Test with many shapes: (1, 2) repeated 100 times -> (1, 2)
        let shapes: Vec<IxD> = (0..100).map(|_| vec![1, 2]).collect();
        let result = rt::broadcast_shapes(&shapes, RowMajor);
        assert_eq!(result, vec![1, 2]);

        // Regression tests for gh-5862: (2,) repeated 32 times -> (2,)
        let shapes: Vec<IxD> = (0..32).map(|_| vec![2]).collect();
        let result = rt::broadcast_shapes(&shapes, RowMajor);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn test_broadcast_shapes_raises() {
        // NumPy v2.4.2, lib/tests/test_stride_tricks.py, test_broadcast_shapes_raises (line 345)
        crate::specify_test!("test_broadcast_shapes_raises");

        // Data format: list of input shapes that should fail
        let data: Vec<Vec<IxD>> = vec![
            // [(3,), (4,)]
            vec![vec![3], vec![4]],
            // [(2, 3), (2,)]
            vec![vec![2, 3], vec![2]],
            // [(3,), (3,), (4,)]
            vec![vec![3], vec![3], vec![4]],
            // [(1, 3, 4), (2, 3, 3)]
            vec![vec![1, 3, 4], vec![2, 3, 3]],
            // [(1, 2), (3, 1), (3, 2), (10, 5)]
            vec![vec![1, 2], vec![3, 1], vec![3, 2], vec![10, 5]],
            // [2, (2, 3)]
            vec![vec![2], vec![2, 3]],
        ];

        for input_shapes in data {
            let result = rt::broadcast_shapes_f(&input_shapes, RowMajor);
            assert!(result.is_err(), "Expected error for input_shapes: {:?}, but got: {:?}", input_shapes, result);
        }

        // Test with incompatible shapes: (2,) * 32 + (3,) * 32 -> should fail
        let mut shapes: Vec<IxD> = (0..32).map(|_| vec![2]).collect();
        shapes.extend((0..32).map(|_| vec![3]));
        let result = rt::broadcast_shapes_f(&shapes, RowMajor);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_shape_internal() {
        // NumPy v2.4.2, lib/tests/test_stride_tricks.py, test_broadcast_shape (line 287)
        // Tests internal _broadcast_shape which is equivalent to broadcast_shapes
        crate::specify_test!("test_broadcast_shape_internal");

        // assert_equal(_broadcast_shape(), ())
        let result = rt::broadcast_shapes(&[], RowMajor);
        assert_eq!(result, vec![]);

        // assert_equal(_broadcast_shape([1, 2]), (2,)) - This is for arrays, not shapes
        // In broadcast_shapes, this would be broadcast_shapes((2,),) -> (2,)
        let result = rt::broadcast_shapes(&[vec![2]], RowMajor);
        assert_eq!(result, vec![2]);

        // assert_equal(_broadcast_shape(np.ones((1, 1))), (1, 1))
        let result = rt::broadcast_shapes(&[vec![1, 1]], RowMajor);
        assert_eq!(result, vec![1, 1]);

        // assert_equal(_broadcast_shape(np.ones((1, 1)), np.ones((3, 4))), (3, 4))
        let result = rt::broadcast_shapes(&[vec![1, 1], vec![3, 4]], RowMajor);
        assert_eq!(result, vec![3, 4]);

        // assert_equal(_broadcast_shape(*([np.ones((1, 2))] * 32)), (1, 2))
        let shapes: Vec<IxD> = (0..32).map(|_| vec![1, 2]).collect();
        let result = rt::broadcast_shapes(&shapes, RowMajor);
        assert_eq!(result, vec![1, 2]);

        // assert_equal(_broadcast_shape(*([np.ones((1, 2))] * 100)), (1, 2))
        let shapes: Vec<IxD> = (0..100).map(|_| vec![1, 2]).collect();
        let result = rt::broadcast_shapes(&shapes, RowMajor);
        assert_eq!(result, vec![1, 2]);

        // Regression tests for gh-5862
        // assert_equal(_broadcast_shape(*([np.ones(2)] * 32 + [1])), (2,))
        let mut shapes: Vec<IxD> = (0..32).map(|_| vec![2]).collect();
        shapes.push(vec![]); // scalar
        let result = rt::broadcast_shapes(&shapes, RowMajor);
        assert_eq!(result, vec![2]);

        // bad_args = [np.ones(2)] * 32 + [np.ones(3)] * 32
        let mut shapes: Vec<IxD> = (0..32).map(|_| vec![2]).collect();
        shapes.extend((0..32).map(|_| vec![3]));
        let result = rt::broadcast_shapes_f(&shapes, RowMajor);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_shapes_col_major() {
        // Test broadcasting in column-major order
        crate::specify_test!("test_broadcast_shapes_col_major");

        // In col-major, broadcasting is applied from left instead of right
        // A      (4d array):  1 x 6 x 1 x 8
        // B      (3d array):  5 x 1 x 7
        // ---------------------------------
        // Result (4d array):  5 x 6 x 7 x 8
        let shapes = vec![vec![1, 6, 1, 8], vec![5, 1, 7]];
        let result = rt::broadcast_shapes(&shapes, ColMajor);
        assert_eq!(result, vec![5, 6, 7, 8]);

        // A      (2d array):  4 x 5
        // B      (1d array):  4
        // --------------------------
        // Result (2d array):  4 x 5
        let shapes = vec![vec![4, 5], vec![4]];
        let result = rt::broadcast_shapes(&shapes, ColMajor);
        assert_eq!(result, vec![4, 5]);

        // A      (3d array):  5 x 3 x 15
        // B      (3d array):  5 x 1 x 15
        // -------------------------------
        // Result (3d array):  5 x 3 x 15
        let shapes = vec![vec![5, 3, 15], vec![5, 1, 15]];
        let result = rt::broadcast_shapes(&shapes, ColMajor);
        assert_eq!(result, vec![5, 3, 15]);

        // A      (3d array):  5 x 3 x 15
        // B      (2d array):  5 x 3
        // -------------------------------
        // Result (3d array):  5 x 3 x 15
        let shapes = vec![vec![5, 3, 15], vec![5, 3]];
        let result = rt::broadcast_shapes(&shapes, ColMajor);
        assert_eq!(result, vec![5, 3, 15]);

        // Col-major cases that would fail in row-major
        // (3,) and (2, 1) -> fails in row-major, works in col-major
        let shapes = vec![vec![3], vec![2, 1]];
        let result = rt::broadcast_shapes_f(&shapes, ColMajor);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod docs_broadcast_shapes {
    use super::*;
    static FUNC: &str = "docs_broadcast_shapes";

    #[test]
    fn doc_broadcast_shapes_basic() {
        crate::specify_test!("doc_broadcast_shapes_basic");

        // A      (4d array):  8 x 1 x 6 x 1
        // B      (3d array):      7 x 1 x 5
        // ---------------------------------
        // Result (4d array):  8 x 7 x 6 x 5
        let shape1 = vec![8, 1, 6, 1];
        let shape2 = vec![7, 1, 5];
        let result = rt::broadcast_shapes(&[shape1, shape2], RowMajor);
        println!("{:?}", result);
        // [8, 7, 6, 5]
        assert_eq!(result, vec![8, 7, 6, 5]);
    }

    #[test]
    fn doc_broadcast_shapes_col_major() {
        crate::specify_test!("doc_broadcast_shapes_col_major");

        // A      (4d array):  1 x 6 x 1 x 8
        // B      (3d array):  5 x 1 x 7
        // ---------------------------------
        // Result (4d array):  5 x 6 x 7 x 8
        let shape1 = vec![1, 6, 1, 8];
        let shape2 = vec![5, 1, 7];
        let result = rt::broadcast_shapes(&[shape1, shape2], ColMajor);
        println!("{:?}", result);
        // [5, 6, 7, 8]
        assert_eq!(result, vec![5, 6, 7, 8]);
    }

    #[test]
    fn doc_broadcast_shapes_multiple() {
        crate::specify_test!("doc_broadcast_shapes_multiple");

        // Three shapes: (1,), (3, 1), (3, 2) -> (3, 2)
        let shapes = vec![vec![1], vec![3, 1], vec![3, 2]];
        let result = rt::broadcast_shapes(&shapes, RowMajor);
        println!("{:?}", result);
        // [3, 2]
        assert_eq!(result, vec![3, 2]);
    }
}
