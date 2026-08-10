use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

mod doc_arange {
    use super::*;
    static FUNC: &str = "doc_arange";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        println!("{}", rt::arange((5, &device)));
        // [ 0 1 2 3 4]
        println!("{}", rt::arange((2, 10, &device)));
        // [ 2 3 4 5 6 7 8 9]
        println!("{}", rt::arange((10, 0, -3, &device)));
        // [ 10 7 4 1]
        assert_eq!(rt::arange((5, &device)).shape()[0], 5);
    }
}

mod doc_linspace {
    use super::*;
    static FUNC: &str = "doc_linspace";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let y: Tensor<f64, _> = rt::linspace((0.0, 1.0, 5, &device));
        println!("{y}");
        // [ 0 0.25 0.5 0.75 1]
        assert_eq!(y.shape()[0], 5);
    }
}

mod doc_zeros_ones_full {
    use super::*;
    static FUNC: &str = "doc_zeros_ones_full";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let z: Tensor<i32, _> = rt::zeros(([2, 3], &device));
        println!("{z}");
        // [[ 0 0 0]
        //  [ 0 0 0]]
        let o: Tensor<i32, _> = rt::ones(([2, 2], &device));
        println!("{o}");
        // [[ 1 1]
        //  [ 1 1]]
        let f: Tensor<i32, _> = rt::full(([2, 2], 7, &device));
        println!("{f}");
        // [[ 7 7]
        //  [ 7 7]]
        assert_eq!(f.shape(), &[2, 2]);
    }
}

mod doc_eye {
    use super::*;
    static FUNC: &str = "doc_eye";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let e: Tensor<i32, _> = rt::eye((4, &device));
        println!("{e}");
        // [[ 1 0 0 0]
        //  [ 0 1 0 0]
        //  [ 0 0 1 0]
        //  [ 0 0 0 1]]
        assert_eq!(e.shape(), &[4, 4]);
    }
}

mod doc_tril_triu {
    use super::*;
    static FUNC: &str = "doc_tril_triu";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<i32, _> = rt::ones(([3, 3], &device));
        println!("{}", rt::tril((&a, 0)));
        // [[ 1 0 0]
        //  [ 1 1 0]
        //  [ 1 1 1]]
        println!("{}", rt::triu((&a, 0)));
        // [[ 1 1 1]
        //  [ 0 1 1]
        //  [ 0 0 1]]
    }
}
