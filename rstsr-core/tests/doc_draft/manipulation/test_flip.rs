#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_flip {
    use super::*;
    static FUNC: &str = "doc_flip";

    #[test]
    fn flip_single_axis() {
        crate::specify_test!("flip_single_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Flipping the first (0) axis
        let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
        println!("{a}");
        // [[[ 0 1]
        //   [ 2 3]]
        //
        //  [[ 4 5]
        //   [ 6 7]]]

        // flip(0)
        let b = a.flip(0);
        assert!(rt::allclose(a.i(slice!(None, None, -1)), &b, None));
        println!("{b}");
        // [[[ 4 5]
        //   [ 6 7]]
        //
        //  [[ 0 1]
        //   [ 2 3]]]
        let target = rt::tensor_from_nested!([[[4, 5], [6, 7]], [[0, 1], [2, 3]]], &device);
        assert!(rt::allclose(&b, &target, None));

        // flip(1)
        let b = a.flip(1);
        assert!(rt::allclose(a.i((.., slice!(None, None, -1))), &b, None));
        println!("{b}");
        // [[[ 2 3]
        //   [ 0 1]]
        //
        //  [[ 6 7]
        //   [ 4 5]]]
        let b_expected = rt::tensor_from_nested!([[[2, 3], [0, 1]], [[6, 7], [4, 5]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
    }

    #[test]
    fn flip_multiple_axes() {
        crate::specify_test!("flip_multiple_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // flip([0, -1])
        let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
        let b = a.flip([0, -1]);
        println!("{b}");
        // [[[ 5 4]
        //   [ 7 6]]
        //
        //  [[ 1 0]
        //   [ 3 2]]]
        let b_expected = rt::tensor_from_nested!([[[5, 4], [7, 6]], [[1, 0], [3, 2]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
    }

    #[test]
    fn flip_all_axes() {
        crate::specify_test!("flip_all_axes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Flipping all axes with None
        let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
        let b = a.flip(None);
        println!("{b}");
        // [[[ 7 6]
        //   [ 5 4]]
        //
        //  [[ 3 2]
        //   [ 1 0]]]
        let b_expected = rt::tensor_from_nested!([[[7, 6], [5, 4]], [[3, 2], [1, 0]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));

        // No fliping any axis with empty tuple
        let b = a.flip(());
        println!("{b}");
        // [[[ 0 1]
        //   [ 2 3]]
        //
        //  [[ 4 5]
        //   [ 6 7]]]
        assert!(rt::allclose(&b, &a, None));
    }
}
