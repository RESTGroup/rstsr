use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod docs_to_contig {
    use super::*;
    static FUNC: &str = "docs_to_contig";

    #[test]
    fn test_doc_basic() {
        // Test basic usage examples from documentation
        crate::specify_test!("test_doc_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Example: Convert non-contiguous tensor to contiguous
        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let sliced = a.i((.., slice!(None, None, 2))); // Every other column
        println!("layout of sliced tensor: {:?}", sliced.layout());
        // 2-Dim (dyn), contiguous: Custom
        // shape: [3, 2], stride: [4, 2], offset: 0
        assert_eq!(sliced.shape(), &[3, 2]);
        assert_eq!(sliced.stride(), &[4, 2]);

        // Convert to C-contiguous
        let contig = sliced.to_contig(RowMajor);
        println!("Contiguous layout: {:?}", contig.layout());
        // 2-Dim (dyn), contiguous: Cc
        // shape: [3, 2], stride: [2, 1], offset: 0
        println!("Contiguous shape: {:?}", contig.shape());
        println!("Contiguous stride: {:?}", contig.stride());
        assert_eq!(contig.shape(), &[3, 2]);
        assert_eq!(contig.stride(), &[2, 1]);
        assert!(contig.c_contig());
    }

    #[test]
    fn test_doc_strided() {
        // Example: Converting strided tensor to contiguous
        crate::specify_test!("test_doc_strided");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Create a strided (non-contiguous) tensor
        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let strided = a.i((slice!(None, None, 2), ..)); // Every other row
        println!("Strided stride: {:?}", strided.stride());
        // [8, 1]
        assert_eq!(strided.stride(), &[8, 1]);

        // Convert to contiguous
        let contig = strided.to_contig(RowMajor);
        println!("Contiguous stride: {:?}", contig.stride());
        // [4, 1]
        assert_eq!(contig.stride(), &[4, 1]);
        assert!(contig.c_contig());
    }
}

#[cfg(test)]
mod docs_to_prefer {
    use super::*;
    static FUNC: &str = "docs_to_prefer";

    #[test]
    fn test_doc_already_contig() {
        // Example: Already contiguous tensor stays as view
        crate::specify_test!("test_doc_already_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Already C-contiguous tensor - no copy
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let result = a.to_prefer(RowMajor);
        assert!(!result.is_owned()); // View returned, no copy

        // Non-contiguous tensor - requires copy
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let transposed = a.t();
        let result = transposed.to_prefer(RowMajor);
        assert!(result.is_owned()); // Owned tensor returned, data copied
    }

    #[test]
    fn test_doc_prefer_vs_contig() {
        // Example: Demonstrating difference between to_prefer and to_contig
        crate::specify_test!("test_doc_prefer_vs_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // C-contiguous tensor stays as view with to_prefer
        let a = rt::tensor_from_nested!([[1, 2], [3, 4], [5, 6]], &device);
        let result = rt::to_prefer(&a, RowMajor);
        assert!(!result.is_owned());

        // Transposed (non-contiguous) tensor gets copied
        let transposed = a.t();
        let result = rt::to_prefer(&transposed, RowMajor);
        assert!(result.is_owned());

        // to_contig always creates a new contiguous layout
        // (though it may reuse data if already contiguous)
        let result = rt::to_contig(&a, RowMajor);
        assert!(!result.is_owned()); // Already C-contig, so view returned
    }

    #[test]
    fn test_doc_to_prefer_assoc() {
        // Example: Using to_prefer as associated method
        crate::specify_test!("test_doc_to_prefer_assoc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Already contiguous - returns view
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let result = a.to_prefer(RowMajor);
        assert!(!result.is_owned());

        // Non-contiguous - returns owned
        let transposed = a.t();
        let result = transposed.to_prefer(RowMajor);
        assert!(result.is_owned());
    }
}

#[cfg(test)]
mod test_to_contig_behavior {
    use super::*;
    static FUNC: &str = "test_to_contig_behavior";

    #[test]
    fn test_already_c_contig() {
        // Already C-contiguous tensor should return view for RowMajor
        crate::specify_test!("test_already_c_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        assert!(a.c_contig());

        // to_contig with RowMajor should return view
        let result = a.to_contig(RowMajor);
        assert!(!result.is_owned());
        assert!(core::ptr::eq(result.raw().as_ptr(), a.raw().as_ptr()));

        // to_contig with ColMajor should copy
        let result = a.to_contig(ColMajor);
        assert!(result.is_owned());
        assert!(result.f_contig());
    }

    #[test]
    fn test_already_f_contig() {
        // Already F-contiguous tensor should return view for ColMajor
        crate::specify_test!("test_already_f_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        assert!(a.f_contig());

        // to_contig with ColMajor should return view
        let result = a.to_contig(ColMajor);
        assert!(!result.is_owned());
        assert!(core::ptr::eq(result.raw().as_ptr(), a.raw().as_ptr()));

        // to_contig with RowMajor should copy
        let result = a.to_contig(RowMajor);
        assert!(result.is_owned());
        assert!(result.c_contig());
    }

    #[test]
    fn test_transposed_tensor() {
        // Transposed tensor should be copied for both orders
        crate::specify_test!("test_transposed_tensor");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let transposed = a.t();

        assert!(!transposed.c_contig());
        assert!(transposed.f_contig());

        // to_contig RowMajor
        let result_c = transposed.to_contig(RowMajor);
        assert!(result_c.is_owned());
        assert!(result_c.c_contig());
        assert_eq!(result_c.shape(), &[4, 3]);

        // to_contig ColMajor
        let result_f = transposed.to_contig(ColMajor);
        assert!(!result_f.is_owned());
        assert!(result_f.f_contig());
        assert_eq!(result_f.shape(), &[4, 3]);

        // Values should match
        let expected_c = rt::tensor_from_nested!([[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], &device);
        assert!(rt::allclose(&result_c, &expected_c, None));
    }

    #[test]
    fn test_sliced_tensor() {
        // Sliced tensor should be copied
        crate::specify_test!("test_sliced_tensor");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((24, &device)).into_shape([4, 6]);
        let sliced = a.i((slice!(None, None, 2), slice!(None, None, 2))); // Every other row and column

        println!("Sliced shape: {:?}", sliced.shape());
        println!("Sliced stride: {:?}", sliced.stride());
        // [2, 3]
        // [12, 2]
        assert_eq!(sliced.shape(), &[2, 3]);
        assert_eq!(sliced.stride(), &[12, 2]);
        assert!(!sliced.c_contig());

        let contig = sliced.to_contig(RowMajor);
        assert!(contig.is_owned());
        assert!(contig.c_contig());
        assert_eq!(contig.stride(), &[3, 1]);

        // Verify values
        let expected = rt::tensor_from_nested!([[0, 2, 4], [12, 14, 16]], &device);
        assert!(rt::allclose(&contig, &expected, None));
    }

    #[test]
    fn test_f_order_sliced() {
        // Slicing in F-order and converting
        crate::specify_test!("test_f_order_sliced");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let a = rt::arange((24, &device)).into_shape([4, 6]);
        let sliced = a.i((slice!(None, None, 2), slice!(None, None, 2)));

        let contig_c = sliced.to_contig(RowMajor);
        let contig_f = sliced.to_contig(ColMajor);

        assert!(contig_c.c_contig());
        assert!(contig_f.f_contig());

        // Both should have same values when iterated
        let vec_c: Vec<_> = contig_c.iter().collect();
        let vec_f: Vec<_> = contig_f.iter().collect();
        assert_eq!(vec_c, vec_f);
    }

    #[test]
    fn test_scalar_tensor() {
        // Scalar (0-D) tensor handling
        crate::specify_test!("test_scalar_tensor");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<i32, _> = rt::asarray((42, &device));

        let contig_c = a.to_contig(RowMajor);
        let contig_f = a.to_contig(ColMajor);

        // Scalar is always contiguous
        assert!(contig_c.c_contig());
        assert!(contig_f.f_contig());

        // Should return view (same data)
        assert!(!contig_c.is_owned());
        assert!(!contig_f.is_owned());
    }

    #[test]
    fn test_1d_tensor() {
        // 1-D tensor handling
        crate::specify_test!("test_1d_tensor");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((10, &device));
        assert!(a.c_contig());
        assert!(a.f_contig());

        let contig = a.to_contig(RowMajor);
        assert!(!contig.is_owned()); // Already contiguous

        // With stride
        let strided = a.i(slice!(None, None, 2));
        assert!(!strided.c_contig());
        assert!(!strided.f_contig());

        let contig = strided.to_contig(RowMajor);
        assert!(contig.is_owned());
        assert_eq!(contig.stride(), &[1]);
    }

    #[test]
    fn test_variants_equivalence() {
        // Test that different function variants produce equivalent results
        crate::specify_test!("test_variants_equivalence");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((24, &device)).into_shape([4, 6]);
        let sliced = a.i((slice!(None, None, 2), slice!(None, None, 2)));

        // Using rt::to_contig
        let result1 = rt::to_contig(&sliced, RowMajor);

        // Using TensorAny::to_contig
        let result2 = sliced.to_contig(RowMajor);

        // Using to_contig_f (fallible)
        let result3 = sliced.to_contig_f(RowMajor).unwrap();

        // All should be equal
        assert!(rt::allclose(&result1, &result2, None));
        assert!(rt::allclose(&result1, &result3, None));

        // Using change_contig (consuming, returns TensorCow)
        let sliced_clone = a.to_owned().into_slice((slice!(None, None, 2), slice!(None, None, 2)));
        let result4 = sliced_clone.change_contig(RowMajor);
        assert!(rt::allclose(&result1, &result4, None));

        // Using into_contig (consuming, returns Tensor)
        let sliced_clone = a.i((slice!(None, None, 2), slice!(None, None, 2)));
        let result5 = sliced_clone.into_contig(RowMajor);
        assert!(rt::allclose(&result1, &result5, None));
    }

    #[test]
    fn test_preserves_values() {
        // Ensure values are preserved during conversion
        crate::specify_test!("test_preserves_values");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Create tensor with specific values
        let data: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5];
        let a = rt::asarray((data.clone(), [3, 4], &device));

        // Slice to make non-contiguous
        let sliced = a.i((1.., slice!(None, None, 2))); // rows 1-2, every other column

        // Convert to both C and F order
        let contig_c = sliced.to_contig(RowMajor);
        let contig_f = sliced.to_contig(ColMajor);

        // Collect values
        let original_values: Vec<f64> = sliced.iter().copied().collect();
        let c_values: Vec<f64> = contig_c.iter().copied().collect();
        let f_values: Vec<f64> = contig_f.iter().copied().collect();

        assert_eq!(original_values, c_values);
        assert_eq!(original_values, f_values);
    }
}

#[cfg(test)]
mod test_to_prefer_behavior {
    use super::*;
    static FUNC: &str = "test_to_prefer_behavior";

    #[test]
    fn test_prefer_c_on_c_contig() {
        // C-contig tensor with RowMajor preference should return view
        crate::specify_test!("test_prefer_c_on_c_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        assert!(a.c_contig());

        let result = a.to_prefer(RowMajor);
        assert!(!result.is_owned()); // Should be view
        assert!(core::ptr::eq(result.raw().as_ptr(), a.raw().as_ptr()));
    }

    #[test]
    fn test_prefer_f_on_f_contig() {
        // F-contig tensor with ColMajor preference should return view
        crate::specify_test!("test_prefer_f_on_f_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        assert!(a.f_contig());

        let result = a.to_prefer(ColMajor);
        assert!(!result.is_owned()); // Should be view
    }

    #[test]
    fn test_prefer_c_on_f_contig() {
        // F-contig tensor with RowMajor preference should copy
        crate::specify_test!("test_prefer_c_on_f_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        assert!(a.f_contig());

        let result = a.to_prefer(RowMajor);
        assert!(result.is_owned()); // Should copy
        assert!(result.c_contig());
    }

    #[test]
    fn test_prefer_f_on_c_contig() {
        // C-contig tensor with ColMajor preference should copy
        crate::specify_test!("test_prefer_f_on_c_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        assert!(a.c_contig());

        let result = a.to_prefer(ColMajor);
        assert!(result.is_owned()); // Should copy
        assert!(result.f_contig());
    }

    #[test]
    fn test_prefer_on_non_contig() {
        // Non-contig tensor should copy regardless of preference
        crate::specify_test!("test_prefer_on_non_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((24, &device)).into_shape([4, 6]);
        let sliced = a.i((slice!(None, None, 2), slice!(None, None, 2)));
        assert!(!sliced.c_contig());
        assert!(!sliced.f_contig());

        let result_c = sliced.to_prefer(RowMajor);
        let result_f = sliced.to_prefer(ColMajor);

        assert!(result_c.is_owned());
        assert!(result_f.is_owned());
    }

    #[test]
    fn test_prefer_vs_contig() {
        // Demonstrate difference between to_prefer and to_contig
        crate::specify_test!("test_prefer_vs_contig");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);

        // to_prefer on already C-contig should return view
        let prefer_result = a.to_prefer(RowMajor);
        assert!(!prefer_result.is_owned());

        // to_contig on already C-contig should also return view
        let contig_result = a.to_contig(RowMajor);
        assert!(!contig_result.is_owned());

        // Both should be equivalent
        assert!(rt::allclose(&prefer_result, &contig_result, None));
    }
}

#[cfg(test)]
mod test_edge_cases {
    use super::*;
    static FUNC: &str = "test_edge_cases";

    #[test]
    fn test_empty_tensor() {
        // Empty tensor handling
        crate::specify_test!("test_empty_tensor");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::zeros(([0, 5], &device));
        let contig = a.to_contig(RowMajor);
        assert_eq!(contig.shape(), &[0, 5]);
        assert!(contig.c_contig());
    }

    #[test]
    fn test_single_element() {
        // Single element tensor
        crate::specify_test!("test_single_element");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::asarray((vec![42.0], [1, 1], &device));
        let contig = a.to_contig(RowMajor);
        assert_eq!(contig.shape(), &[1, 1]);
        assert!(contig.c_contig());
    }

    #[test]
    fn test_high_dim() {
        // Higher dimensional tensors
        crate::specify_test!("test_high_dim");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((120, &device)).into_shape([2, 3, 4, 5]);
        let sliced = a.i((.., slice!(None, None, 2), .., slice!(None, None, 2)));

        let contig_c = sliced.to_contig(RowMajor);
        let contig_f = sliced.to_contig(ColMajor);

        assert!(contig_c.c_contig());
        assert!(contig_f.f_contig());

        // Verify shapes
        assert_eq!(contig_c.shape(), sliced.shape());
        assert_eq!(contig_f.shape(), sliced.shape());
    }

    #[test]
    fn test_reverse_stride() {
        // Tensor with negative strides (from flip)
        crate::specify_test!("test_reverse_stride");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let flipped = a.flip(0); // Flip along axis 0

        println!("Flipped shape: {:?}", flipped.shape());
        println!("Flipped stride: {:?}", flipped.stride());

        let contig = flipped.to_contig(RowMajor);
        assert!(contig.c_contig());
        assert!(contig.stride()[0] > 0); // Positive stride after conversion

        // Verify values are correct
        let expected = rt::tensor_from_nested!([[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]], &device);
        assert!(rt::allclose(&contig, &expected, None));
    }
}
