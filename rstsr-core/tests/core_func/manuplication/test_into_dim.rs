use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod docs_into_dim {
    use super::*;
    static FUNC: &str = "docs_into_dim";

    #[test]
    fn doc_to_dim() {
        crate::specify_test!("doc_to_dim");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]); // shape: (2, 3), IxD

        // you can debug print tensor or it's layout to verify this
        println!("a: {:?}", a);
        // you can also call `const_ndim` to verify the dimension type
        // `None` here indicates dynamic dimension
        assert_eq!(a.shape().const_ndim(), None);

        let b = a.to_dim::<Ix2>(); // shape: (2, 3), Ix2
        println!("b: {:?}", b);
        assert_eq!(b.shape().const_ndim(), Some(2));

        // use `.to_dim::<IxD>()` or `.to_dyn()` to convert back to dynamic dimension
        let c = b.to_dyn(); // shape: (2, 3), IxD
        println!("c: {:?}", c);
        assert_eq!(c.shape().const_ndim(), None);
    }

    #[test]
    #[should_panic]
    fn doc_to_dim_panic() {
        crate::specify_test!("doc_to_dim_panic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]); // shape: (2, 3), IxD
        let b = a.to_dim::<Ix3>(); // shape: (2, 3), Ix3, panics
        println!("b: {:?}", b);
    }
}
