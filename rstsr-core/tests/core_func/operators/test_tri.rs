//! Custom (non-docstring) tests for the triangular pack/unpack functions:
//! the `uplo`/`symm` behavior that the docstring examples do not cover, the
//! free-function/method equivalence, and the `_with_output` forms.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::DeviceType;
use crate::TESTCFG;

#[cfg(test)]
mod custom_tri {
    use super::*;
    static FUNC: &str = "custom_tri";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);

        // both convenience forms: pack values and their symmetric restore
        let packed_l = a.pack_tril();
        assert_eq!(packed_l.to_vec(), vec![1.0, 4.0, 5.0, 7.0, 8.0, 9.0]);
        let packed_u = a.pack_triu();
        assert_eq!(packed_u.to_vec(), vec![1.0, 2.0, 3.0, 5.0, 6.0, 9.0]);
        assert_eq!(format!("{}", packed_l.unpack_tril(FlagSymm::Sy)), "[[ 1 4 7]\n [ 4 5 8]\n [ 7 8 9]]");
        assert_eq!(format!("{}", packed_u.unpack_triu(FlagSymm::Sy)), "[[ 1 2 3]\n [ 2 5 6]\n [ 3 6 9]]");

        // the general forms agree with the lower/upper convenience forms
        assert_eq!(a.pack_tri(FlagUpLo::L).to_vec(), packed_l.to_vec());
        assert_eq!(a.pack_tri(FlagUpLo::U).to_vec(), packed_u.to_vec());
        assert_eq!(format!("{}", packed_l.unpack_tri(FlagUpLo::L, FlagSymm::Sy)), "[[ 1 4 7]\n [ 4 5 8]\n [ 7 8 9]]");

        // the free functions agree with the associated methods
        assert_eq!(rt::pack_tri(&a, FlagUpLo::L).to_vec(), packed_l.to_vec());
        assert_eq!(rt::pack_tri(&a, FlagUpLo::U).to_vec(), packed_u.to_vec());
        assert_eq!(rt::pack_tril(&a).to_vec(), packed_l.to_vec());
        assert_eq!(rt::pack_triu(&a).to_vec(), packed_u.to_vec());
        assert_eq!(
            format!("{}", rt::unpack_tril(&packed_l, FlagSymm::Sy)),
            format!("{}", packed_l.unpack_tril(FlagSymm::Sy))
        );
        assert_eq!(
            format!("{}", rt::unpack_triu(&packed_u, FlagSymm::Sy)),
            format!("{}", packed_u.unpack_triu(FlagSymm::Sy))
        );

        let p = rt::tensor_from_nested!([1.0, 4.0, 5.0, 7.0, 8.0, 9.0], &device);

        // `symm` derives the opposite triangle: `Ay` negates and zeroes the diagonal
        assert_eq!(format!("{}", p.unpack_tri(FlagUpLo::L, FlagSymm::Ay)), "[[ 0 -4 -7]\n [ 4 0 -8]\n [ 7 8 0]]");

        // `N` assumes no symmetry and zero-fills the opposite triangle, for both `uplo`
        assert_eq!(format!("{}", p.unpack_tri(FlagUpLo::L, FlagSymm::N)), "[[ 1 0 0]\n [ 4 5 0]\n [ 7 8 9]]");
        assert_eq!(format!("{}", p.unpack_tri(FlagUpLo::U, FlagSymm::N)), "[[ 1 4 5]\n [ 0 7 8]\n [ 0 0 9]]");

        // a strided packed vector takes the non-contiguous inner path; `N` still zero-fills
        let base = rt::linspace((0.0, 11.0, 12, &device));
        let strided = base.i(slice!(0, 12, 2));
        assert_eq!(format!("{}", strided.unpack_tri(FlagUpLo::L, FlagSymm::N)), "[[ 0 0 0]\n [ 2 4 0]\n [ 6 8 10]]");
    }

    #[test]
    fn test_with_output() {
        crate::specify_test!("test_with_output");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);

        // pack: the free function and the method form both write the packed vector
        let mut packed = rt::zeros(([6], &device));
        rt::pack_tri_with_output(&a, FlagUpLo::L, &mut packed);
        assert_eq!(packed.to_vec(), vec![1.0, 4.0, 5.0, 7.0, 8.0, 9.0]);

        let mut packed_u = rt::zeros(([6], &device));
        a.pack_tri_with_output(FlagUpLo::U, &mut packed_u);
        assert_eq!(packed_u.to_vec(), vec![1.0, 2.0, 3.0, 5.0, 6.0, 9.0]);

        // unpack: same values, including an F-layout (non-contiguous under RowMajor) output
        let mut square = rt::zeros(([3, 3], &device));
        rt::unpack_tri_with_output(&packed, FlagUpLo::L, FlagSymm::Sy, &mut square);
        assert_eq!(format!("{square}"), "[[ 1 4 7]\n [ 4 5 8]\n [ 7 8 9]]");

        let mut square_f = rt::zeros(([3, 3], &device)).into_layout(Layout::new(vec![3, 3], vec![1, 3], 0).unwrap());
        packed_u.unpack_tri_with_output(FlagUpLo::U, FlagSymm::N, &mut square_f);
        assert_eq!(format!("{square_f}"), "[[ 1 2 3]\n [ 0 5 6]\n [ 0 0 9]]");

        // shape mismatch is rejected instead of writing out of bounds
        let mut bad_pack = rt::zeros(([5], &device));
        assert!(rt::pack_tri_with_output_f(&a, FlagUpLo::L, &mut bad_pack).is_err());
        let mut bad_unpack = rt::zeros(([2, 2], &device));
        assert!(rt::unpack_tri_with_output_f(&packed, FlagUpLo::L, FlagSymm::Sy, &mut bad_unpack).is_err());

        // a broadcast (stride-0) output would alias elements
        let (storage, _) = rt::zeros(([1], &device)).into_raw_parts();
        let mut broadcast = Tensor::<f64, DeviceType, IxD>::new(storage, Layout::new(vec![6], vec![0], 0).unwrap());
        assert!(rt::pack_tri_with_output_f(&a, FlagUpLo::L, &mut broadcast).is_err());
    }
}
