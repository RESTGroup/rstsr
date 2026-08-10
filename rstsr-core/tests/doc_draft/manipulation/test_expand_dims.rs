use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

mod doc_expand_dims {
    use super::*;
    static FUNC: &str = "doc_expand_dims";

    #[test]
    fn test_doc_examples() {
        crate::specify_test!("test_doc_examples");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // expand dims at axis 0
        let x = rt::arange((2, &device));
        let y = x.expand_dims(0);
        println!("{y}");
        // [[ 0 1]]
        println!("y shape: {:?}", y.shape());
        // y shape: [1, 2]
        let y_expected = rt::tensor_from_nested!([[0, 1]], &device);
        assert!(rt::allclose(&y, &y_expected, None));
        assert_eq!(y.shape(), &[1, 2]);
        assert_eq!(x.i(None).shape(), y.shape());

        // expand dims at axis -1 (last axis)
        let x = rt::arange((2, &device));
        let y = x.expand_dims(-1);
        println!("{y}");
        // [[ 0]
        //  [ 1]]
        println!("y shape: {:?}", y.shape());
        // y shape: [2, 1]
        let y_expected = rt::tensor_from_nested!([[0], [1]], &device);
        assert!(rt::allclose(&y, &y_expected, None));
        assert_eq!(y.shape(), &[2, 1]);
        assert_eq!(x.i((Ellipsis, None)).shape(), &[2, 1]);

        // expand dims at axes 0 and 1
        let x = rt::arange((2, &device));
        let y = x.expand_dims([0, 1]);
        println!("{y}");
        // [[[ 0 1]]]
        println!("y shape: {:?}", y.shape());
        // y shape: [1, 1, 2]
        let y_expected = rt::tensor_from_nested!([[[0, 1]]], &device);
        assert!(rt::allclose(&y, &y_expected, None));
        assert_eq!(y.shape(), &[1, 1, 2]);
        assert_eq!(x.i((None, None)).shape(), &[1, 1, 2]);

        // Expand dims at axes 0 and 2
        let x = rt::arange((2, &device));
        let y = x.expand_dims([0, 2]);
        println!("{y}");
        // [[[ 0]]
        //  [[ 1]]]
        println!("y shape: {:?}", y.shape());
        // y shape: [1, 2, 1]
        let y_expected = rt::tensor_from_nested!([[[0], [1]]], &device);
        assert!(rt::allclose(&y, &y_expected, None));
        assert_eq!(y.shape(), &[1, 2, 1]);
        assert_eq!(x.i((None, Ellipsis, None)).shape(), &[1, 2, 1]);
    }
}
