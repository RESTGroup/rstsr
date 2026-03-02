use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod docs_into_compatible_shape {
    use super::*;
    static FUNC: &str = "docs_into_compatible_shape";

    #[test]
    fn compatible_reshape() {
        crate::specify_test!("compatible_reshape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // shape: (4, 6), stride: (6, 1), c-contiguous
        let a = rt::arange((24, &device)).into_shape([4, 6]);
        // Split a dimension: (4, 6) -> (2, 2, 6) - layout compatible
        let b = a.into_compatible_shape([2, 2, 6], RowMajor);
        assert_eq!(b.shape(), &[2, 2, 6]);
    }

    #[test]
    fn incompatible_reshape() {
        crate::specify_test!("incompatible_reshape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
        let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
        // layout compatible
        assert!(a.to_compatible_shape_f([4, 6 * 9], RowMajor).is_ok());
        // layout incompatible
        assert!(a.to_compatible_shape_f([4 * 6, 9], RowMajor).is_err());
    }
}
