use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_to_layout {
    use super::*;
    static FUNC: &str = "doc_to_layout";

    #[test]
    fn test_doc_basic() {
        // Basic usage of to_layout
        crate::specify_test!("test_doc_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Convert tensor to a different layout
        let a = rt::arange((12, &device)).into_shape([3, 4]);
        println!("a layout: {:?}", a.layout());
        // 2-Dim (dyn), contiguous: Cc
        // shape: [3, 4], stride: [4, 1], offset: 0

        // Convert to F-contiguous layout
        let layout_f = [3, 4].f();
        let b = a.to_layout(layout_f);
        println!("b layout: {:?}", b.layout());
        // 2-Dim (dyn), contiguous: Fc
        // shape: [3, 4], stride: [1, 3], offset: 0
        assert!(b.f_contig());
        assert_eq!(b.shape(), &[3, 4]);

        // Values are preserved
        assert!(rt::allclose(&a, &b, None));
    }

    #[test]
    fn test_doc_reshape_via_layout() {
        // Using to_layout to reshape tensor
        crate::specify_test!("test_doc_reshape_via_layout");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        println!("a shape: {:?}", a.shape());
        // a shape: [3, 4]

        // Flatten to 1D
        let layout_1d = [12].c();
        let b = a.to_layout(layout_1d);
        println!("b shape: {:?}", b.shape());
        // b shape: [12]
        assert_eq!(b.shape(), &[12]);

        // Reshape to different 2D
        let layout_2d = [2, 6].c();
        let c = b.to_layout(layout_2d);
        println!("c shape: {:?}", c.shape());
        // c shape: [2, 6]
        assert_eq!(c.shape(), &[2, 6]);
    }

    #[test]
    fn test_doc_custom_layout() {
        // Using custom layout with specific strides (transpose effect)
        crate::specify_test!("test_doc_custom_layout");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        println!("{a}");
        // [[ 0 1 2]
        //  [ 3 4 5]]

        // Create F-contiguous layout which effectively transposes the data
        let layout_f = [2, 3].f();
        let b = a.to_layout(layout_f);
        println!("{b}");
        // F-contiguous layout reorders the data
        assert_eq!(b.shape(), &[2, 3]);

        // For a custom transposed view, use transpose instead
        let c = a.t();
        println!("{c}");
        // [[ 0 3]
        //  [ 1 4]
        //  [ 2 5]]
        assert_eq!(c.shape(), &[3, 2]);

        let expected = rt::tensor_from_nested!([[0, 3], [1, 4], [2, 5]], &device);
        assert!(rt::allclose(&c, &expected, None));
    }
}
