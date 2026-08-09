use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_moveaxis {
    use super::*;
    static FUNC: &str = "numpy_moveaxis";

    #[test]
    fn test_basic() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestMoveaxis::test_move_to_end (line 3893)
        crate::specify_test!("test_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.random.randn(5, 6, 7)
        // for source, expected in [(0, (6, 7, 5)),
        //                          (1, (5, 7, 6)),
        //                          (2, (5, 6, 7)),
        //                          (-1, (5, 6, 7))]:
        //     actual = np.moveaxis(x, source, -1).shape
        //     assert_(actual, expected)
        let x: Tensor<f64, _> = rt::zeros(([5, 6, 7], &device));
        for (source, expected) in [(0isize, [6, 7, 5]), (1isize, [5, 7, 6]), (2isize, [5, 6, 7]), (-1isize, [5, 6, 7])]
        {
            let result = x.moveaxis(source, -1);
            assert_eq!(result.shape(), &expected);
        }
    }

    #[test]
    fn test_move_new_position() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestMoveaxis::test_move_new_position (line 3902)
        crate::specify_test!("test_move_new_position");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.random.randn(1, 2, 3, 4)
        // for source, destination, expected in [
        //         (0, 1, (2, 1, 3, 4)),
        //         (1, 2, (1, 3, 2, 4)),
        //         (1, -1, (1, 3, 4, 2)),
        //         ]:
        //     actual = np.moveaxis(x, source, destination).shape
        //     assert_(actual, expected)
        let x: Tensor<f64, _> = rt::zeros(([1, 2, 3, 4], &device));
        for (source, destination, expected) in
            [(0isize, 1isize, [2, 1, 3, 4]), (1isize, 2isize, [1, 3, 2, 4]), (1isize, -1isize, [1, 3, 4, 2])]
        {
            let result = x.moveaxis(source, destination);
            assert_eq!(result.shape(), &expected);
        }
    }

    #[test]
    fn test_preserve_order() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestMoveaxis::test_preserve_order (line 3912)
        crate::specify_test!("test_preserve_order");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.zeros((1, 2, 3, 4))
        // for source, destination in [
        //         (0, 0),
        //         (3, -1),
        //         (-1, 3),
        //         ([0, -1], [0, -1]),
        //         ([2, 0], [2, 0]),
        //         (range(4), range(4)),
        //         ]:
        //     actual = np.moveaxis(x, source, destination).shape
        //     assert_(actual, (1, 2, 3, 4))
        let x: Tensor<f64, _> = rt::zeros(([1, 2, 3, 4], &device));

        // (0, 0)
        let result = x.moveaxis(0, 0);
        assert_eq!(result.shape(), &[1, 2, 3, 4]);

        // (3, -1)
        let result = x.moveaxis(3, -1);
        assert_eq!(result.shape(), &[1, 2, 3, 4]);

        // (-1, 3)
        let result = x.moveaxis(-1, 3);
        assert_eq!(result.shape(), &[1, 2, 3, 4]);

        // ([0, -1], [0, -1])
        let result = x.moveaxis([0, -1], [0, -1]);
        assert_eq!(result.shape(), &[1, 2, 3, 4]);

        // ([2, 0], [2, 0])
        let result = x.moveaxis([2, 0], [2, 0]);
        assert_eq!(result.shape(), &[1, 2, 3, 4]);

        // (range(4), range(4)); numpy's `range` is not accepted by rstsr, use a Vec instead
        let axes: Vec<isize> = vec![0, 1, 2, 3];
        let result = x.moveaxis(&axes, &axes);
        assert_eq!(result.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_equivalent_operations() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestMoveaxis::test_move_multiples (line 3925)
        crate::specify_test!("test_equivalent_operations");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Retained equivalence checks: these all achieve the same result for a (3, 4, 5)
        // tensor -> (5, 4, 3)
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

        // x = np.zeros((0, 1, 2, 3))
        // for source, destination, expected in [
        //         ([0, 1], [2, 3], (2, 3, 0, 1)),
        //         ([2, 3], [0, 1], (2, 3, 0, 1)),
        //         ([0, 1, 2], [2, 3, 0], (2, 3, 0, 1)),
        //         ([3, 0], [1, 0], (0, 3, 1, 2)),
        //         ([0, 3], [0, 1], (0, 3, 1, 2)),
        //         ]:
        //     actual = np.moveaxis(x, source, destination).shape
        //     assert_(actual, expected)
        let x: Tensor<f64, _> = rt::zeros(([0, 1, 2, 3], &device));

        // ([0, 1], [2, 3]) -> (2, 3, 0, 1)
        let result = x.moveaxis([0, 1], [2, 3]);
        assert_eq!(result.shape(), &[2, 3, 0, 1]);

        // ([2, 3], [0, 1]) -> (2, 3, 0, 1)
        let result = x.moveaxis([2, 3], [0, 1]);
        assert_eq!(result.shape(), &[2, 3, 0, 1]);

        // ([0, 1, 2], [2, 3, 0]) -> (2, 3, 0, 1)
        let result = x.moveaxis([0, 1, 2], [2, 3, 0]);
        assert_eq!(result.shape(), &[2, 3, 0, 1]);

        // ([3, 0], [1, 0]) -> (0, 3, 1, 2)
        let result = x.moveaxis([3, 0], [1, 0]);
        assert_eq!(result.shape(), &[0, 3, 1, 2]);

        // ([0, 3], [0, 1]) -> (0, 3, 1, 2)
        let result = x.moveaxis([0, 3], [0, 1]);
        assert_eq!(result.shape(), &[0, 3, 1, 2]);
    }

    #[test]
    fn test_errors() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestMoveaxis::test_errors (line 3937)
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
