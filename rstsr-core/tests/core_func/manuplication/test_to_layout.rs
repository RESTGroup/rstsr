use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod test_to_layout_behavior {
    use super::*;
    static FUNC: &str = "test_to_layout_behavior";

    #[test]
    fn test_same_layout_no_copy() {
        // Same layout should return view without copy
        crate::specify_test!("test_same_layout_no_copy");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let layout = a.layout().clone();

        let result = a.to_layout(layout);
        assert!(!result.is_owned());
        assert!(core::ptr::eq(result.raw().as_ptr(), a.raw().as_ptr()));
    }

    #[test]
    fn test_different_layout_copies() {
        // Different layout should copy data
        crate::specify_test!("test_different_layout_copies");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        // Create F-contiguous layout with same shape
        let layout_f = [3, 4].f();

        let result = a.to_layout(layout_f);
        assert!(result.is_owned());
        assert!(result.f_contig());
    }

    #[test]
    fn test_dimensionality_change() {
        // to_layout can change dimensionality
        crate::specify_test!("test_dimensionality_change");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        // Flatten to 1D
        let layout_1d = [12].c();

        let result = a.to_layout(layout_1d);
        assert_eq!(result.shape(), &[12]);
        assert!(result.c_contig());

        // Values should be preserved
        let expected: Vec<i32> = a.iter().copied().collect();
        let actual: Vec<i32> = result.iter().copied().collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_strided_tensor() {
        // Strided (non-contiguous) tensor to specific layout
        crate::specify_test!("test_strided_tensor");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let sliced = a.i((.., slice!(None, None, 2))); // shape [3, 2], stride [4, 2]

        // Convert to C-contiguous
        let layout_c = [3, 2].c();
        let result = sliced.to_layout(layout_c);

        assert!(result.is_owned());
        assert!(result.c_contig());
        assert_eq!(result.shape(), &[3, 2]);

        // Verify values
        let expected = rt::tensor_from_nested!([[0, 2], [4, 6], [8, 10]], &device);
        assert!(rt::allclose(&result, &expected, None));
    }

    #[test]
    fn test_size_mismatch_error() {
        // Layout with different size should error
        crate::specify_test!("test_size_mismatch_error");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        // Layout with wrong size
        let layout_wrong = [3, 3].c();

        assert!(a.to_layout_f(layout_wrong).is_err());
    }

    #[test]
    fn test_variants_equivalence() {
        // Test that different function variants produce equivalent results
        crate::specify_test!("test_variants_equivalence");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let layout_f = [3, 4].f();

        // Using rt::to_layout
        let result1 = rt::to_layout(&a, layout_f.clone());

        // Using TensorAny::to_layout
        let result2 = a.to_layout(layout_f.clone());

        // Using to_layout_f (fallible)
        let result3 = a.to_layout_f(layout_f.clone()).unwrap();

        // All should be equal
        assert!(rt::allclose(&result1, &result2, None));
        assert!(rt::allclose(&result1, &result3, None));
    }

    #[test]
    fn test_into_layout() {
        // into_layout consumes the tensor and returns owned Tensor
        crate::specify_test!("test_into_layout");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let layout_f = [3, 4].f();

        let result = a.into_layout(layout_f);
        // into_layout returns Tensor (owned), which is always F-contig here
        assert!(result.f_contig());
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_change_layout() {
        // change_layout consumes tensor and returns TensorCow
        // Note: change_layout on owned Tensor returns owned TensorCow,
        // since into_cow() on owned tensor returns DataCow::Owned
        crate::specify_test!("test_change_layout");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let layout = a.layout().clone();

        // change_layout consumes owned tensor, returns TensorCow::Owned
        let result = a.change_layout(layout);
        // The result has the same layout
        assert_eq!(result.shape(), &[3, 4]);
        assert!(result.c_contig());
    }
}

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
