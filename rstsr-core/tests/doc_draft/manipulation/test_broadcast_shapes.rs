use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_broadcast_shapes {
    use super::*;
    static FUNC: &str = "doc_broadcast_shapes";

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
