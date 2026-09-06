//! Doc-draft twins of the map_elementwise docstrings.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_map {
    use super::*;
    static FUNC: &str = "doc_map";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);

        // map: element-wise function by reference
        let b = a.map(|x| x * 2);
        println!("{b}");
        // [[ 0 2 4]
        //  [ 6 8 10]]
        assert_eq!(format!("{b}"), "[[ 0 2 4]\n [ 6 8 10]]");

        // mapv: by value, arbitrary output dtype
        let c = a.mapv(|x| x as f64 / 2.0);
        println!("{c}");
        // [[ 0 0.5 1]
        //  [ 1.5 2 2.5]]
        assert_eq!(format!("{c}"), "[[ 0 0.5 1]\n [ 1.5 2 2.5]]");

        // mapi: in place by mutable reference
        let mut d = a.clone();
        d.mapi(|x| *x += 1);
        println!("{d}");
        // [[ 1 2 3]
        //  [ 4 5 6]]
        assert_eq!(format!("{d}"), "[[ 1 2 3]\n [ 4 5 6]]");

        // mapb: two tensors, broadcast against each other
        let e = rt::full(([2, 3], 10, &device));
        let f = a.mapb(&e, |x, y| x + y);
        println!("{f}");
        // [[ 10 11 12]
        //  [ 13 14 15]]
        assert_eq!(format!("{f}"), "[[ 10 11 12]\n [ 13 14 15]]");
    }
}
