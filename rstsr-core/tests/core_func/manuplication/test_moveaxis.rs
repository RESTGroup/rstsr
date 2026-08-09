use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_moveaxis {
    use super::*;
    static FUNC: &str = "numpy_moveaxis";

    #[test]
    fn test_basic() {
        // NumPy v2.5.2, _core/numeric.py, moveaxis docstring examples
        crate::specify_test!("test_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.zeros((3, 4, 5))
        // np.moveaxis(x, 0, -1).shape -> (4, 5, 3)
        let x: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));
        let result = x.moveaxis(0, -1);
        println!("{:?}", result.shape());
        // [4, 5, 3]
        assert_eq!(result.shape(), &[4, 5, 3]);

        // np.moveaxis(x, -1, 0).shape -> (5, 3, 4)
        let result = x.moveaxis(-1, 0);
        println!("{:?}", result.shape());
        // [5, 3, 4]
        assert_eq!(result.shape(), &[5, 3, 4]);
    }

    #[test]
    fn test_equivalent_operations() {
        // NumPy v2.5.2, _core/numeric.py, moveaxis docstring examples
        // These all achieve the same result for a (3, 4, 5) tensor -> (5, 4, 3)
        crate::specify_test!("test_equivalent_operations");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));

        // np.transpose(x).shape -> (5, 4, 3)
        let result1 = x.transpose(None);
        println!("{:?}", result1.shape());
        // [5, 4, 3]
        assert_eq!(result1.shape(), &[5, 4, 3]);

        // np.swapaxes(x, 0, -1).shape -> (5, 4, 3)
        let result2 = x.swapaxes(0, -1);
        println!("{:?}", result2.shape());
        // [5, 4, 3]
        assert_eq!(result2.shape(), &[5, 4, 3]);

        // np.moveaxis(x, [0, 1], [-1, -2]).shape -> (5, 4, 3)
        let result3 = x.moveaxis([0, 1], [-1, -2]);
        println!("{:?}", result3.shape());
        // [5, 4, 3]
        assert_eq!(result3.shape(), &[5, 4, 3]);

        // np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape -> (5, 4, 3)
        let result4 = x.moveaxis([0, 1, 2], [-1, -2, -3]);
        println!("{:?}", result4.shape());
        // [5, 4, 3]
        assert_eq!(result4.shape(), &[5, 4, 3]);
    }

    #[test]
    fn test_errors() {
        // Test error conditions
        crate::specify_test!("test_errors");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));

        // source and destination must have the same length
        assert!(x.moveaxis_f([0, 1], [0]).is_err());

        // duplicate source axes
        assert!(x.moveaxis_f([0, 0], [1, 2]).is_err());

        // duplicate destination axes
        assert!(x.moveaxis_f([0, 1], [2, 2]).is_err());

        // out of bounds source
        assert!(x.moveaxis_f(5, 0).is_err());
        assert!(x.moveaxis_f(-6, 0).is_err());

        // out of bounds destination
        assert!(x.moveaxis_f(0, 5).is_err());
        assert!(x.moveaxis_f(0, -6).is_err());
    }
}
