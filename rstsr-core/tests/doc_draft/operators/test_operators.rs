use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

mod doc_arithmetic {
    use super::*;
    static FUNC: &str = "doc_arithmetic";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        let b = rt::tensor_from_nested!([[10, 20], [30, 40]], &device);
        println!("{}", &a + &b);
        // [[ 11 22]
        //  [ 33 44]]
        println!("{}", &a * &b);
        // [[ 10 40]
        //  [ 90 160]]
        let af = rt::tensor_from_nested!([[10.0, 20.0], [30.0, 40.0]], &device);
        let bf = rt::tensor_from_nested!([[2.0, 4.0], [5.0, 8.0]], &device);
        println!("{}", &af / &bf);
        // [[ 5 5]
        //  [ 6 5]]
    }
}

mod doc_comparison {
    use super::*;
    static FUNC: &str = "doc_comparison";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2, 3, 4], &device);
        let b = rt::tensor_from_nested!([2, 2, 2, 2], &device);
        println!("{}", rt::gt(&a, &b));
        // [ false false true true]
        println!("{}", rt::eq(&a, &b));
        // [ false true false false]
    }
}

mod doc_maximum_minimum {
    use super::*;
    static FUNC: &str = "doc_maximum_minimum";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 5, 3], &device);
        let b = rt::tensor_from_nested!([4, 2, 6], &device);
        println!("{}", rt::maximum(&a, &b));
        // [ 4 5 6]
        println!("{}", rt::minimum(&a, &b));
        // [ 1 2 3]
    }
}
