//! Doc-draft twins of the ownership_conversion docstrings.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_view {
    use super::*;
    static FUNC: &str = "doc_view";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // view_mut writes through to the original tensor
        let mut a: Tensor<i32, _> = rt::arange((6, &device)).into_shape([2, 3]);
        let mut v = a.view_mut();
        v += 10;
        drop(v);
        println!("{a}");
        // [[ 10 11 12]
        //  [ 13 14 15]]
        assert_eq!(format!("{a}"), "[[ 10 11 12]\n [ 13 14 15]]");

        // view only reads
        let view = a.view();
        println!("{view}");
        // [[ 10 11 12]
        //  [ 13 14 15]]
        assert_eq!(format!("{view}"), "[[ 10 11 12]\n [ 13 14 15]]");
    }
}

#[cfg(test)]
mod doc_into_owned {
    use super::*;
    static FUNC: &str = "doc_into_owned";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a compact tensor moves its buffer; a sliced view gathers the visible
        // elements into a fresh owned tensor (TensorIterOrder::K)
        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        let v = a.into_slice((.., .., 0..2));
        let o = v.into_owned();
        println!("{o}");
        // [[[ 0 1]
        //   [ 4 5]
        //   [ 8 9]]
        //
        //  [[ 12 13]
        //   [ 16 17]
        //   [ 20 21]]]
        assert_eq!(format!("{o}"), "[[[ 0 1]\n  [ 4 5]\n  [ 8 9]]\n\n [[ 12 13]\n  [ 16 17]\n  [ 20 21]]]");

        // into_shared is cheap; the data is shared until written
        let a = rt::arange((6, &device));
        let s = a.into_shared();
        let v1 = s.view();
        println!("{v1}");
        // [ 0 1 2 3 4 5]
        assert_eq!(format!("{v1}"), "[ 0 1 2 3 4 5]");
    }
}

#[cfg(test)]
mod doc_to_vec_scalar {
    use super::*;
    static FUNC: &str = "doc_to_vec_scalar";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // to_vec: 1-D only
        let a = rt::arange((6, &device));
        let v: Vec<i32> = a.to_vec();
        println!("{v:?}");
        // [0, 1, 2, 3, 4, 5]
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);

        // to_scalar reads the single element at the layout offset
        let a = rt::arange((10, &device));
        let scalar = a.i(9);
        println!("{}", scalar.to_scalar());
        // 9
        assert_eq!(scalar.to_scalar(), 9);
    }
}
