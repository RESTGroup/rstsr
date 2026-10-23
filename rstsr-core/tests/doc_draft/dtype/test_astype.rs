use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_astype {
    use super::*;
    static FUNC: &str = "doc_astype";

    #[test]
    fn test_doc_basic() {
        crate::specify_test!("test_doc_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6i32, &device)).into_shape([2, 3]);
        let b = a.astype::<f64>();
        println!("{b}");
        // [[ 0 1 2]
        //  [ 3 4 5]]
        assert!(b.is_owned());
        let expected = rt::tensor_from_nested!([[0f64, 1., 2.], [3., 4., 5.]], &device);
        crate::test_utils::assert_equal(&b, &expected, None);
        assert_eq!(format!("{b}"), "[[ 0 1 2]\n [ 3 4 5]]");

        // same dtype: a view, no allocation
        let c = a.astype::<i32>();
        assert!(!c.is_owned());
        assert_eq!(format!("{c}"), "[[ 0 1 2]\n [ 3 4 5]]");
    }
}
