//! Doc-draft twins of the assignment docstrings.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_assign_fill {
    use super::*;
    static FUNC: &str = "doc_assign_fill";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // assign casts the dtype as needed
        let mut a: Tensor<f32, _> = rt::zeros(([2, 3], &device));
        let b = rt::arange((6, &device)).into_shape([2, 3]);
        a.assign(&b);
        println!("{a}");
        // [[ 0 1 2]
        //  [ 3 4 5]]
        assert_eq!(format!("{a}"), "[[ 0 1 2]\n [ 3 4 5]]");

        // broadcastable sources are assigned element-wise
        let mut c: Tensor<f32, _> = rt::zeros(([2, 3], &device));
        let row = rt::tensor_from_nested!([[1.0, 2.0, 3.0]], &device);
        c.assign(row.view().into_shape([1, 3]));
        println!("{c}");
        // [[ 1 2 3]
        //  [ 1 2 3]]
        assert_eq!(format!("{c}"), "[[ 1 2 3]\n [ 1 2 3]]");

        // fill sets every element
        let mut d: Tensor<i32, _> = rt::zeros(([2, 2], &device));
        d.fill(7);
        println!("{d}");
        // [[ 7 7]
        //  [ 7 7]]
        assert_eq!(format!("{d}"), "[[ 7 7]\n [ 7 7]]");
    }
}
