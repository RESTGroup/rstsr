use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

mod doc_indexing {
    use super::*;
    static FUNC: &str = "doc_indexing";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let m = rt::arange((12, &device)).into_shape([3, 4]);
        println!("{}", m.i(1));
        // [ 4 5 6 7]
        println!("{}", m.i((.., slice!(1, 3))));
        // [[ 1 2]
        //  [ 5 6]
        //  [ 9 10]]
        println!("{:?}", m.i((.., None)).shape());
        // [3, 1, 4]
        let v = rt::arange((6, &device));
        println!("{:?}", rt::index_select(&v, 0, [1, 3, 5]));
        // [1, 3, 5]
    }
}
