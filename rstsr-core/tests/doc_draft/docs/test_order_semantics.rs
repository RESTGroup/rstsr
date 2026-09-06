//! Doc-draft twin of the crate page `src/docs/order_semantics.md`.
//!
//! Every example displayed on that page must originate from this test; shown
//! output is pasted from the actual run and pinned by string assertions.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_order_semantics {
    use super::*;
    static FUNC: &str = "doc_order_semantics";

    #[test]
    fn creation_default_order() {
        crate::specify_test!("creation_default_order");

        // row-major: last axis is contiguous in memory
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        println!("{a}");
        assert_eq!(format!("{a}"), "[[ 0 1 2]\n [ 3 4 5]]");
        println!("{:?}", a.layout());
        assert_eq!(
            format!("{:?}", a.layout()),
            "2-Dim (dyn), contiguous: Cc\nshape: [2, 3], stride: [3, 1], offset: 0"
        );

        // column-major: first axis is contiguous in memory
        let mut device = DeviceType::default();
        device.set_default_order(ColMajor);
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        println!("{a}");
        assert_eq!(format!("{a}"), "[[ 0 2 4]\n [ 1 3 5]]");
        println!("{:?}", a.layout());
        assert_eq!(
            format!("{:?}", a.layout()),
            "2-Dim (dyn), contiguous: Ff\nshape: [2, 3], stride: [1, 2], offset: 0"
        );
    }

    #[test]
    fn broadcast_asymmetry() {
        crate::specify_test!("broadcast_asymmetry");

        // row-major: shapes align from the last axis (NumPy rule)
        let shape1 = vec![8, 1, 6, 1];
        let shape2 = vec![7, 1, 5];
        let result = rt::broadcast_shapes(&[shape1, shape2], RowMajor);
        println!("{result:?}");
        assert_eq!(result, vec![8, 7, 6, 5]);
        assert_eq!(format!("{result:?}"), "[8, 7, 6, 5]");

        // column-major: shapes align from the first axis (Fortran/Julia rule)
        let shape1 = vec![1, 6, 1, 8];
        let shape2 = vec![5, 1, 7];
        let result = rt::broadcast_shapes(&[shape1, shape2], ColMajor);
        println!("{result:?}");
        assert_eq!(result, vec![5, 6, 7, 8]);
        assert_eq!(format!("{result:?}"), "[5, 6, 7, 8]");
    }

    #[test]
    fn default_order_setup() {
        crate::specify_test!("default_order_setup");

        // twin of the page's opening snippet
        let mut device = DeviceType::default();
        device.set_default_order(ColMajor); // or RowMajor; RowMajor is the default
        assert_eq!(device.default_order(), ColMajor);
    }

    #[test]
    fn iteration_invariance() {
        crate::specify_test!("iteration_invariance");

        // reshaping never changes which value sits at a logical index; the
        // reading order of the output follows the reading order of the input
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let b = a.reshape([3, 2]);
        println!("{}", a.reshape([6]));
        println!("{}", b.reshape([6]));
        assert_eq!(format!("{}", a.reshape([6])), "[ 0 1 2 3 4 5]");
        assert_eq!(format!("{}", b.reshape([6])), "[ 0 1 2 3 4 5]");

        let mut device = DeviceType::default();
        device.set_default_order(ColMajor);
        // col-major: contiguous situation is ([4, 6], 9), not (4, [6, 9])
        let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
        println!("{:?}", a.layout());
        assert_eq!(
            format!("{:?}", a.layout()),
            "3-Dim (dyn), contiguous: f\nshape: [4, 6, 9], stride: [1, 4, 32], offset: 0"
        );
        // merging the leading (contiguous) dimensions needs no copy
        assert!(!a.reshape([24, 9]).is_owned());
        // merging across the contiguity boundary requires a copy
        assert!(a.reshape([4, 54]).is_owned());
    }
}
