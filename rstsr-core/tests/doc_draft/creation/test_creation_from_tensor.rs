//! Doc-draft twins of the creation_from_tensor docstrings.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_diag {
    use super::*;
    static FUNC: &str = "doc_diag";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // 2-D input: extract the diagonal (offset 0)
        let a = rt::arange((9, &device)).into_shape([3, 3]);
        println!("{}", rt::diag(&a));
        assert_eq!(format!("{}", rt::diag(&a)), "[ 0 4 8]");

        // 2-D input with offset
        println!("{}", rt::diag((&a, 1)));
        assert_eq!(format!("{}", rt::diag((&a, 1))), "[ 1 5]");

        // 1-D input: construct a diagonal matrix
        let v = rt::arange((3, &device));
        println!("{}", rt::diag((&v, -1)));
        assert_eq!(format!("{}", rt::diag((&v, -1))), "[[ 0 0 0 0]\n [ 0 0 0 0]\n [ 0 1 0 0]\n [ 0 0 2 0]]");

        // column-major default order: a constructed diagonal matrix follows the
        // device default order (F-contiguous) with identical logical content.
        let mut device_c = TESTCFG.device.clone();
        device_c.set_default_order(ColMajor);

        let v_c = rt::arange((3, &device_c));
        let m_c = rt::diag((&v_c, -1));
        println!("{m_c}");
        println!("{:?}", m_c.layout());
        assert_eq!(format!("{m_c}"), "[[ 0 0 0 0]\n [ 0 0 0 0]\n [ 0 1 0 0]\n [ 0 0 2 0]]");
        assert!(m_c.f_contig());

        // the same logical tensor stored F-contiguously: extraction is unchanged
        let a_f = a.to_contig(ColMajor);
        assert!(a_f.f_contig());
        assert_eq!(format!("{}", rt::diag(&a_f)), "[ 0 4 8]");
        assert_eq!(format!("{}", rt::diag((&a_f, 1))), "[ 1 5]");
    }
}

#[cfg(test)]
mod doc_meshgrid {
    use super::*;
    static FUNC: &str = "doc_meshgrid";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x = rt::arange((3, &device));
        let y = rt::arange((2, &device));
        let grids = rt::meshgrid(([&x, &y], "ij"));
        println!("{}", grids[0]);
        println!("{}", grids[1]);
        assert_eq!(format!("{}", grids[0]), "[[ 0 0]\n [ 1 1]\n [ 2 2]]");
        assert_eq!(format!("{}", grids[1]), "[[ 0 1]\n [ 0 1]\n [ 0 1]]");

        // copy = true: grids are owned contiguous copies
        let grids_c = rt::meshgrid(([&x, &y], "ij", true));
        assert!(grids_c.iter().all(|grid| grid.is_owned()));
        assert!(grids_c[0].c_contig());

        // copy = false: broadcast views sharing the inputs' memory (stride-0
        // axes, not owned), as in NumPy
        let grids_v = rt::meshgrid(([&x, &y], "ij", false));
        assert!(grids_v.iter().all(|grid| !grid.is_owned()));
        println!("{}", grids_v[0]);
        println!("{:?}", grids_v[0].layout());
        assert_eq!(format!("{}", grids_v[0]), "[[ 0 0]\n [ 1 1]\n [ 2 2]]");
        assert_eq!(
            format!("{:?}", grids_v[0].layout()),
            "2-Dim (dyn), contiguous: Custom\nshape: [3, 2], stride: [1, 0], offset: 0"
        );
        assert_eq!(format!("{}", grids_v[1]), "[[ 0 1]\n [ 0 1]\n [ 0 1]]");
        assert_eq!(
            format!("{:?}", grids_v[1].layout()),
            "2-Dim (dyn), contiguous: Custom\nshape: [3, 2], stride: [0, 1], offset: 0"
        );
    }
}

#[cfg(test)]
mod doc_concat {
    use super::*;
    static FUNC: &str = "doc_concat";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let b = rt::full(([2, 2], 9, &device));
        println!("{}", rt::concat(([a, b], 1)));
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let b = rt::full(([2, 2], 9, &device));
        assert_eq!(format!("{}", rt::concat(([a, b], 1))), "[[ 0 1 2 9 9]\n [ 3 4 5 9 9]]");

        // default axis is 0
        let a = rt::arange((3, &device));
        let b = rt::arange((3, 6, &device));
        println!("{}", rt::concat([&a, &b]));
        assert_eq!(format!("{}", rt::concat([&a, &b])), "[ 0 1 2 3 4 5]");
    }
}

#[cfg(test)]
mod doc_stack_family {
    use super::*;
    static FUNC: &str = "doc_stack_family";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // hstack of 1-D inputs concatenates along axis 0
        let a = rt::arange((3, &device));
        let b = rt::arange((3, 6, &device));
        println!("{}", rt::hstack([&a, &b]));
        assert_eq!(format!("{}", rt::hstack([&a, &b])), "[ 0 1 2 3 4 5]");

        // hstack of 2-D inputs concatenates along axis 1
        let a = rt::arange((4, &device)).into_shape([2, 2]);
        let b = rt::full(([2, 1], 9, &device));
        println!("{}", rt::hstack([&a, &b]));
        assert_eq!(format!("{}", rt::hstack([&a, &b])), "[[ 0 1 9]\n [ 2 3 9]]");

        // vstack promotes 1-D inputs to rows
        let a = rt::arange((3, &device));
        let b = rt::arange((3, 6, &device));
        println!("{}", rt::vstack([&a, &b]));
        assert_eq!(format!("{}", rt::vstack([&a, &b])), "[[ 0 1 2]\n [ 3 4 5]]");

        // stack joins along a new axis
        let a = rt::arange((4, &device));
        let b = rt::full(([4], 9, &device));
        println!("{}", rt::stack([&a, &b]));
        assert_eq!(format!("{}", rt::stack([&a, &b])), "[[ 0 1 2 3]\n [ 9 9 9 9]]");

        // unstack splits along an axis into views
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let v = rt::unstack(&a);
        println!("{}", v.len());
        println!("{}", v[0]);
        println!("{}", v[1]);
        assert_eq!(v.len(), 2);
        assert_eq!(format!("{}", v[0]), "[ 0 1 2]");
        assert_eq!(format!("{}", v[1]), "[ 3 4 5]");
    }
}

#[cfg(test)]
mod doc_atleast {
    use super::*;
    static FUNC: &str = "doc_atleast";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((3, &device));
        println!("{}", rt::atleast_2d(&a));
        assert_eq!(format!("{}", rt::atleast_2d(&a)), "[[ 0 1 2]]");

        // atleast_1d keeps 1-D unchanged; the result is a view
        let v = rt::atleast_1d(&a);
        println!("{}", v);
        assert_eq!(format!("{}", v), "[ 0 1 2]");

        // atleast_3d promotes 1-D to (1, N, 1)
        let w = rt::atleast_3d(&a);
        println!("{}", w);
        println!("{:?}", w.layout());
        assert_eq!(w.shape(), &[1, 3, 1]);
        assert_eq!(format!("{w}"), "[[[ 0]\n  [ 1]\n  [ 2]]]");
    }
}
