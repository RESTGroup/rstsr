use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_to_contig {
    use super::*;
    static FUNC: &str = "doc_to_contig";

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
mod doc_to_prefer {
    use super::*;
    static FUNC: &str = "doc_to_prefer";

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
