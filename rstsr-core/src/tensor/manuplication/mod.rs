//! This module handles tensor data manipulation.

pub mod broadcast;
pub mod expand_dims;
pub mod flip;
pub mod into_dim;
pub mod reshape;
pub mod reshape_assume_contig;
pub mod squeeze;
pub mod to_contig;
pub mod to_layout;
pub mod transpose;

pub mod exports {
    use super::*;

    pub use broadcast::*;
    pub use expand_dims::*;
    pub use flip::*;
    pub use into_dim::*;
    pub use reshape::*;
    pub use reshape_assume_contig::*;
    pub use squeeze::*;
    pub use to_contig::*;
    pub use to_layout::*;
    pub use transpose::*;
}

#[cfg(test)]
mod test_reshape {
    use crate::prelude_dev::*;

    #[test]
    fn test_playground() {
        #[cfg(not(feature = "col_major"))]
        {
            let a1 = linspace((1.0, 24.0, 24));
            let a2 = a1.to_shape([2, 3, 4]);
            let default_order = a1.device().default_order();
            println!("{a2:?}");
            println!("{:?}", core::ptr::eq(a1.as_ptr(), a2.as_ptr()));

            let v = layout_reshapeable(a1.layout(), &vec![2, 3, 4], default_order).unwrap();
            println!("{v:?}");

            let b1 = linspace((1.0, 24.0, 24)).into_layout(vec![2, 3, 4].f());
            let b2 = b1.to_shape([24]);
            println!("{b2:?}");
            println!("{:?}", core::ptr::eq(b1.as_ptr(), b2.as_ptr()));

            let v = layout_reshapeable(b1.layout(), &vec![24], default_order).unwrap();
            println!("{v:?}");
        }
        #[cfg(feature = "col_major")]
        {
            let a1 = linspace((1.0, 24.0, 24));
            let a2 = a1.to_shape([2, 3, 4]);
            let default_order = a1.device().default_order();
            println!("{a2:?}");
            println!("{:?}", core::ptr::eq(a1.as_ptr(), a2.as_ptr()));
            println!("a2[:, :, 0] =\n{:}", a2.i((.., .., 0)));
            println!("a2[:, :, 1] =\n{:}", a2.i((.., .., 1)));
            println!("a2[:, :, 2] =\n{:}", a2.i((.., .., 2)));
            println!("a2[:, :, 3] =\n{:}", a2.i((.., .., 3)));

            let v = layout_reshapeable(a1.layout(), &vec![2, 3, 4], default_order).unwrap();
            println!("{v:?}");

            let b1 = linspace((1.0, 24.0, 24)).into_layout(vec![2, 3, 4].f());
            let b2 = b1.to_shape([24]);
            println!("{b2:?}");
            println!("{:?}", core::ptr::eq(b1.as_ptr(), b2.as_ptr()));

            let v = layout_reshapeable(b1.layout(), &vec![24], default_order).unwrap();
            println!("{v:?}");
        }
    }

    #[test]
    fn test_contig() {
        #[cfg(not(feature = "col_major"))]
        {
            let layout_in = vec![2, 3, 4].c();
            let default_order = RowMajor;
            let layout_out = layout_reshapeable(&layout_in, &vec![2, 3, 4], default_order).unwrap();
            assert_eq!(layout_out.unwrap(), vec![2, 3, 4].c());

            let layout_out = layout_reshapeable(&layout_in, &vec![3, 2, 4], default_order).unwrap();
            assert_eq!(layout_out.unwrap(), vec![3, 2, 4].c());

            let layout_out = layout_reshapeable(&layout_in, &vec![1, 4, 1, 6], default_order).unwrap();
            assert_eq!(layout_out.unwrap(), vec![1, 4, 1, 6].c());
        }
        #[cfg(feature = "col_major")]
        {
            let layout_in = vec![2, 3, 4].f();
            let default_order = ColMajor;
            let layout_out = layout_reshapeable(&layout_in, &vec![2, 3, 4], default_order).unwrap();
            assert_eq!(layout_out.unwrap(), vec![2, 3, 4].f());

            let layout_out = layout_reshapeable(&layout_in, &vec![3, 2, 4], default_order).unwrap();
            assert_eq!(layout_out.unwrap(), vec![3, 2, 4].f());

            let layout_out = layout_reshapeable(&layout_in, &vec![1, 4, 1, 6], default_order).unwrap();
            assert_eq!(layout_out.unwrap(), vec![1, 4, 1, 6].f());
        }
    }

    #[test]
    fn test_partial_contig() {
        #[cfg(not(feature = "col_major"))]
        {
            // np.zeros(12, 15, 18); a[3:, :, ::3]
            // this case is actually contiguous, but with stride 3
            let layout_in = Layout::new(vec![9, 15, 6], vec![270, 18, 3], 810).unwrap();
            let default_order = RowMajor;

            let layout_out = layout_reshapeable(&layout_in, &vec![15, 9, 2, 3], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![15, 9, 2, 3]);
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![162, 18, 9, 3]);
            assert_eq!(layout_out.as_ref().unwrap().offset(), layout_in.offset());

            let layout_out = layout_reshapeable(&layout_in, &vec![10, 27, 3], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![10, 27, 3]);
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![243, 9, 3]);
            assert_eq!(layout_out.as_ref().unwrap().offset(), layout_in.offset());

            // insert some new axes
            let layout_out = layout_reshapeable(&layout_in, &vec![1, 10, 1, 27, 3], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![1, 10, 1, 27, 3]);
            // strides follows c-contiguous, but zero is also valid for broadcast
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![2430, 243, 243, 9, 3]);

            // np.zeros(12, 15, 18); a[3:, :, 3:15:2]
            // this case is not contiguous in last two dimensions
            let layout_in = Layout::new(vec![9, 15, 6], vec![270, 18, 2], 813).unwrap();

            let layout_out = layout_reshapeable(&layout_in, &vec![15, 9, 2, 3], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![15, 9, 2, 3]);
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![162, 18, 6, 2]);
            assert_eq!(layout_out.as_ref().unwrap().offset(), layout_in.offset());

            let layout_out = layout_reshapeable(&layout_in, &vec![10, 27, 3], default_order).unwrap();
            assert!(layout_out.is_none());
        }
        #[cfg(feature = "col_major")]
        {
            let layout_in = Layout::new(vec![6, 15, 9], vec![3, 18, 270], 810).unwrap();
            let default_order = ColMajor;

            let layout_out = layout_reshapeable(&layout_in, &vec![3, 2, 9, 15], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![3, 2, 9, 15]);
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![3, 9, 18, 162]);
            assert_eq!(layout_out.as_ref().unwrap().offset(), layout_in.offset());

            let layout_out = layout_reshapeable(&layout_in, &vec![3, 27, 10], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![3, 27, 10]);
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![3, 9, 243]);
            assert_eq!(layout_out.as_ref().unwrap().offset(), layout_in.offset());

            // insert some new axes
            let layout_out = layout_reshapeable(&layout_in, &vec![3, 27, 1, 10, 1], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![3, 27, 1, 10, 1]);
            // strides follows f-contiguous, but zero is also valid for broadcast
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![3, 9, 243, 243, 2430]);

            // np.zeros(12, 15, 18); a[3:, :, 3:15:2]
            // this case is not contiguous in last two dimensions
            let layout_in = Layout::new(vec![6, 15, 9], vec![2, 18, 270], 813).unwrap();

            let layout_out = layout_reshapeable(&layout_in, &vec![3, 2, 9, 15], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![3, 2, 9, 15]);
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![2, 6, 18, 162]);
            assert_eq!(layout_out.as_ref().unwrap().offset(), layout_in.offset());

            let layout_out = layout_reshapeable(&layout_in, &vec![10, 27, 3], default_order).unwrap();
            assert!(layout_out.is_none());
        }
    }

    #[test]
    fn test_minus_stride() {
        #[cfg(not(feature = "col_major"))]
        {
            // np.zeros(12, 15, 18); a[3:, ::-1, ::-3]
            // this case should be seen contiguous in last two dimensions
            let layout_in = Layout::new(vec![9, 15, 6], vec![270, -18, -3], 1079).unwrap();
            let default_order = RowMajor;

            let layout_out = layout_reshapeable(&layout_in, &vec![15, 9, 2, 3], default_order).unwrap();
            assert!(layout_out.is_none());

            let layout_out = layout_reshapeable(&layout_in, &vec![3, 3, 10, 9], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![3, 3, 10, 9]);
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![810, 270, -27, -3]);
        }
    }

    #[test]
    fn test_broadcast_reshape() {
        #[cfg(not(feature = "col_major"))]
        {
            // a = np.zeros(12, 15, 18);
            // b = np.broadcast_to(a[:, None], (12, 16, 15, 18))
            let layout_in = unsafe { Layout::new_unchecked(vec![12, 16, 15, 18], vec![270, 0, 18, 1], 0) };
            let default_order = RowMajor;

            let layout_out = layout_reshapeable(&layout_in, &vec![4, 3, 4, 4, 9, 1, 30], default_order).unwrap();
            assert_eq!(layout_out.as_ref().unwrap().shape(), &vec![4, 3, 4, 4, 9, 1, 30]);
            assert_eq!(layout_out.as_ref().unwrap().stride(), &vec![810, 270, 0, 0, 30, 30, 1]);

            let layout_out = layout_reshapeable(&layout_in, &vec![16, 12, 15, 18], default_order).unwrap();
            assert!(layout_out.is_none());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude_dev::*;

    #[test]
    fn test_to_shape_assume_contig() {
        let a = linspace((2.5, 3.2, 16));
        let b = a.to_shape_assume_contig_f([4, 4]).unwrap();
        println!("{b:.3?}");
    }

    #[test]
    fn test_expand_dims() {
        let a: Tensor<f64, _> = zeros([4, 9, 8]);
        let b = a.expand_dims(2);
        assert_eq!(b.shape(), &[4, 9, 1, 8]);
        let b = a.expand_dims([1, 3]);
        assert_eq!(b.shape(), &[4, 1, 9, 1, 8]);
        let b = a.expand_dims([1, -1]);
        assert_eq!(b.shape(), &[4, 1, 9, 8, 1]);
        let b = a.expand_dims([-1, -4, 1, 0]);
        assert_eq!(b.shape(), &[1, 1, 4, 1, 9, 8, 1]);
    }

    #[test]
    fn test_squeeze() {
        let a: Tensor<f64, _> = zeros([4, 1, 9, 1, 8, 1]);
        let b = a.squeeze(3);
        assert_eq!(b.shape(), &[4, 1, 9, 8, 1]);
        let b = a.squeeze([1, 3]);
        assert_eq!(b.shape(), &[4, 9, 8, 1]);
        let b = a.squeeze([1, -1]);
        assert_eq!(b.shape(), &[4, 9, 1, 8]);
        let b = a.squeeze_f(-7);
        assert!(b.is_err());
    }

    #[test]
    fn test_flip() {
        let a = arange(24.0).into_shape([2, 3, 4]).into_owned();
        println!("{a:?}");

        let b = a.flip(1);
        println!("{b:?}");
        assert_eq!(b.shape(), &[2, 3, 4]);
        let c = a.flip([0, -1]);
        println!("{c:?}");
        assert_eq!(c.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_swapaxes() {
        let a = arange(24.0).into_shape([2, 3, 4]).into_owned();
        println!("{a:?}");

        let b = a.swapaxes(0, 1);
        println!("{b:?}");
        assert_eq!(b.shape(), &[3, 2, 4]);
    }

    #[test]
    fn test_to_shape() {
        let a = linspace((0.0, 15.0, 16));
        let mut a = a.to_shape([4, 4]);
        a.layout = Layout::new(vec![2, 2], vec![2, 4], 0).unwrap();
        println!("{a:?}");
        let b = a.to_shape([2, 2]);
        println!("{b:?}");

        let c = a.to_shape([2, -1]);
        println!("{c:?}");
        assert_eq!(c.shape(), &[2, 2]);

        let d = a.to_shape_f([3, -1]);
        assert!(d.is_err());
    }

    #[test]
    fn test_broadcast_to() {
        #[cfg(not(feature = "col_major"))]
        {
            let a = linspace((0.0, 15.0, 16));
            let a = a.into_shape_assume_contig_f([4, 1, 4]).unwrap();
            let a = a.to_broadcast_f([6, 4, 3, 4]).unwrap();
            println!("{a:?}");
            assert_eq!(a.layout(), unsafe { &Layout::new_unchecked([6, 4, 3, 4], [0, 4, 0, 1], 0) });
        }
        #[cfg(feature = "col_major")]
        {
            let a = linspace((0.0, 15.0, 16));
            let a = a.into_shape_assume_contig_f([4, 1, 4]).unwrap();
            let a = a.to_broadcast_f([4, 3, 4, 6]).unwrap();
            println!("{a:?}");
            assert_eq!(a.layout(), unsafe { &Layout::new_unchecked([4, 3, 4, 6], [1, 0, 4, 0], 0) });
        }
    }

    #[test]
    fn test_to_layout() {
        let a = linspace((0.0, 15.0, 16));
        let a = a.change_shape([4, 4]);
        let a = a.into_layout(Layout::new([2, 8], [12, 120], 8).unwrap());
        println!("{a:?}");
    }
}
