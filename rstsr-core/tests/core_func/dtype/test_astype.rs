use rstsr::prelude::*;

use super::CATEGORY;
use crate::test_utils::assert_equal;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_astype {
    use super::*;
    static FUNC: &str = "numpy_astype";

    #[test]
    fn test_astype() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestAsType::test_astype (L4258)
        crate::specify_test!("test_astype");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // data = [[1, 2], [3, 4]]
        // actual = np.astype(np.array(data, dtype=np.int64), np.uint32)
        // expected = np.array(data, dtype=np.uint32)
        // assert_array_equal(actual, expected)
        // assert_equal(actual.dtype, expected.dtype)
        let data = rt::tensor_from_nested!([[1i64, 2], [3, 4]], &device);
        let actual = data.astype::<u32>();
        let expected = rt::tensor_from_nested!([[1u32, 2], [3, 4]], &device);
        assert_equal(&actual, &expected, None);

        // assert np.shares_memory(actual, np.astype(actual, actual.dtype, copy=False))
        let actual_same = actual.astype::<u32>();
        assert!(!actual_same.is_owned());
        assert!(core::ptr::eq(actual.raw().as_ptr(), actual_same.raw().as_ptr()));

        // actual = np.astype(np.int64(10), np.float64)
        // expected = np.float64(10)
        // assert_equal(actual, expected)
        let scalar = rt::from_scalar(10i64, &device);
        let actual_scalar = scalar.astype::<f64>();
        let expected_scalar = rt::from_scalar(10f64, &device);
        assert_equal(&actual_scalar, &expected_scalar, None);
    }

    #[test]
    fn test_astype_copyflag() {
        // numpy: v2.5.2 | _core/tests/test_api.py::test_astype_copyflag (L653)
        crate::specify_test!("test_astype_copyflag");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr = np.arange(10, dtype=np.intp)
        let arr = rt::arange((10i64, &device));

        // res_true = arr.astype(np.intp, copy=True)
        // assert not np.shares_memory(arr, res_true)
        let res_true = arr.astype::<i64>().into_owned();
        assert!(!core::ptr::eq(arr.raw().as_ptr(), res_true.raw().as_ptr()));

        // res_false = arr.astype(np.intp, copy=False)
        // assert np.shares_memory(arr, res_false)
        let res_false = arr.astype::<i64>();
        assert!(!res_false.is_owned());
        assert!(core::ptr::eq(arr.raw().as_ptr(), res_false.raw().as_ptr()));

        // res_false_float = arr.astype(np.float64, copy=False)
        // assert not np.shares_memory(arr, res_false_float)
        let res_false_float = arr.astype::<f64>();
        assert!(res_false_float.is_owned());
        assert!(!core::ptr::addr_eq(arr.raw().as_ptr(), res_false_float.raw().as_ptr()));
    }
}

#[cfg(test)]
mod custom_astype {
    use super::*;
    use num::Complex;
    static FUNC: &str = "custom_astype";

    #[test]
    fn test_same_dtype_is_view() {
        crate::specify_test!("test_same_dtype_is_view");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // non-trivial layout: every other column starting at column 1
        let a = rt::arange((12i64, &device)).into_shape([3, 4]);
        let v = a.i((.., slice!(1, None, 2)));
        assert_eq!(v.shape(), &[3, 2]);
        assert_eq!(v.stride(), &[4, 2]);
        assert_eq!(v.layout().offset(), 1);

        let c = v.astype::<i64>();
        assert!(!c.is_owned());
        assert_eq!(c.shape(), v.shape());
        assert_eq!(c.stride(), v.stride());
        assert_eq!(c.layout().offset(), v.layout().offset());
        assert!(core::ptr::eq(c.raw().as_ptr(), v.raw().as_ptr()));
        assert_equal(&c, &v, None);
    }

    #[test]
    fn test_dtype_change_materializes_dense() {
        crate::specify_test!("test_dtype_change_materializes_dense");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // strided input: the result is dense and keeps the memory order
        let a = rt::arange((12i64, &device)).into_shape([3, 4]);
        let v = a.i((.., slice!(None, None, 2)));
        let c = v.astype::<f64>();
        assert!(c.is_owned());
        assert!(c.c_contig());
        let expected = rt::tensor_from_nested!([[0f64, 2.], [4., 6.], [8., 10.]], &device);
        assert_equal(&c, &expected, None);

        // broadcast (zero-stride) input materializes all elements
        let col = a.i((.., slice!(0, 1)));
        let bcast = rt::broadcast_to(&col, vec![3usize, 4]);
        let c = bcast.astype::<f64>();
        assert_eq!(c.shape(), &[3, 4]);
        let expected = rt::tensor_from_nested!([[0f64, 0., 0., 0.], [4., 4., 4., 4.], [8., 8., 8., 8.]], &device);
        assert_equal(&c, &expected, None);
    }

    #[test]
    fn test_cast_semantics() {
        crate::specify_test!("test_cast_semantics");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // float -> int truncates toward zero
        let f = rt::tensor_from_nested!([[-1.9f64, 2.5, 0.0]], &device);
        let i = f.astype::<i32>();
        let expected = rt::tensor_from_nested!([[-1i32, 2, 0]], &device);
        assert_equal(&i, &expected, None);

        // numeric -> bool is truthiness (nonzero); bool has no allclose support,
        // so compare the buffers directly
        let b = i.astype::<bool>();
        let expected = rt::tensor_from_nested!([[true, true, false]], &device);
        assert_eq!(b.raw().to_vec(), expected.raw().to_vec());

        // bool -> numeric
        let n = b.astype::<u8>();
        let expected = rt::tensor_from_nested!([[1u8, 1, 0]], &device);
        assert_equal(&n, &expected, None);

        // real -> complex
        let z = i.astype::<Complex<f64>>();
        let expected =
            rt::asarray((vec![Complex::new(-1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(0.0, 0.0)], &device))
                .into_shape([1, 3]);
        assert_equal(&z, &expected, None);

        // complex -> complex (downcast)
        let z32 = z.astype::<Complex<f32>>();
        assert_equal(&z32, &expected, None);
    }

    #[test]
    fn test_cast_preserves_col_major_order() {
        crate::specify_test!("test_cast_preserves_col_major_order");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let a = rt::arange((12i64, &device)).into_shape([3, 4]);
        assert!(a.f_contig());

        let c = a.astype::<f64>();
        assert!(c.is_owned());
        assert!(c.f_contig());

        let expected: Tensor<f64, _> =
            rt::asarray((vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &device)).into_shape([3, 4]);
        assert_equal(&c, &expected, None);
    }

    #[test]
    fn test_zero_size() {
        crate::specify_test!("test_zero_size");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<i64, _> = rt::arange((0i64, &device));
        let b = a.astype::<f64>();
        assert_eq!(b.shape(), &[0]);
        assert_eq!(b.size(), 0);
    }
}
