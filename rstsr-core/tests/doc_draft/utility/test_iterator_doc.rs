//! Doc-draft twins of the iterator docstrings.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_iter {
    use super::*;
    static FUNC: &str = "doc_iter";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let collected: Vec<i32> = a.iter().cloned().collect();
        println!("{collected:?}");
        // [0, 1, 2, 3, 4, 5]
        assert_eq!(collected, vec![0, 1, 2, 3, 4, 5]);

        // indexed iteration yields logical indices
        let pairs: Vec<_> = a.indexed_iter().map(|(idx, v)| (idx.to_vec(), *v)).collect();
        println!("{pairs:?}");
        // [([0, 0], 0), ([0, 1], 1), ([0, 2], 2), ([1, 0], 3), ([1, 1], 4), ([1, 2], 5)]
        let expected =
            vec![(vec![0, 0], 0), (vec![0, 1], 1), (vec![0, 2], 2), (vec![1, 0], 3), (vec![1, 1], 4), (vec![1, 2], 5)];
        assert_eq!(pairs, expected);

        // iter_mut writes through
        let mut b: Tensor<i32, _> = rt::zeros(([2, 2], &device));
        for (i, x) in b.iter_mut().enumerate() {
            *x = i as i32;
        }
        println!("{b}");
        // [[ 0 1]
        //  [ 2 3]]
        assert_eq!(format!("{b}"), "[[ 0 1]\n [ 2 3]]");
    }
}

#[cfg(test)]
mod doc_axes_iter {
    use super::*;
    static FUNC: &str = "doc_axes_iter";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);

        // iterate along axis 0: two views of shape (3,)
        for v in a.axes_iter(0) {
            println!("{v}");
        }
        // [ 0 1 2]
        // [ 3 4 5]

        // iterate along the last axis: three views of shape (2,)
        for v in a.axes_iter(-1) {
            println!("{v}");
        }
        // [ 0 3]
        // [ 1 4]
        // [ 2 5]

        // collect to check
        let rows: Vec<_> = a.axes_iter(0).map(|v| v.to_vec()).collect();
        assert_eq!(rows, vec![vec![0, 1, 2], vec![3, 4, 5]]);
        let cols: Vec<_> = a.axes_iter(-1).map(|v| v.to_vec()).collect();
        assert_eq!(cols, vec![vec![0, 3], vec![1, 4], vec![2, 5]]);

        // axes_iter_mut writes through
        let mut b: Tensor<i32, _> = rt::zeros(([2, 3], &device));
        for mut v in b.axes_iter_mut(0) {
            v += 1;
        }
        println!("{b}");
        // [[ 1 1 1]
        //  [ 1 1 1]]
        assert_eq!(format!("{b}"), "[[ 1 1 1]\n [ 1 1 1]]");
    }
}
