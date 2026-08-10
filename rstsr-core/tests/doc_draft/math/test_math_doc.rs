use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

mod doc_unary_math {
    use super::*;
    static FUNC: &str = "doc_unary_math";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        use std::f64::consts::PI;

        let a = rt::tensor_from_nested!([-1, 2, -3], &device);
        println!("{}", rt::abs(&a));
        // [ 1 2 3]
        let s = rt::tensor_from_nested!([0.0, 1.0, 4.0], &device);
        println!("{}", rt::sqrt(&s));
        // [ 0 1 2]
        let ang = rt::tensor_from_nested!([0.0, PI / 2.0], &device);
        println!("{}", rt::sin(&ang));
        // [ 0 1]
        let x = rt::tensor_from_nested!([0.0, f64::NAN, f64::INFINITY], &device);
        println!("{}", rt::is_nan(&x));
        // [ false true false]
    }
}
