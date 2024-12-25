//! Conversion to/from Faer

use crate::prelude_dev::*;
use core::mem::ManuallyDrop;
use faer::complex_native::{c32, c64};
use faer::{MatMut, MatRef, SimpleEntity};
use faer_ext::{IntoFaer, IntoFaerComplex};
use num::Complex;

impl<'a, T, B> IntoFaer for TensorView<'a, T, Ix2, B>
where
    T: SimpleEntity,
    B: DeviceStorageAPI<T, RawVec = Vec<T>>,
{
    type Faer = MatRef<'a, T>;

    fn into_faer(self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let rawvec = self.data().storage().rawvec();
        let ptr = rawvec.as_ptr();
        unsafe { faer::mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

impl<'a, T, B> IntoFaer for TensorViewMut<'a, T, Ix2, B>
where
    T: SimpleEntity,
    B: DeviceStorageAPI<T, RawVec = Vec<T>>,
{
    type Faer = MatMut<'a, T>;

    fn into_faer(mut self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let rawvec = self.data_mut().storage_mut().rawvec_mut();
        let ptr = rawvec.as_mut_ptr();
        unsafe { faer::mat::from_raw_parts_mut(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

impl<'a, B> IntoFaerComplex for TensorView<'a, Complex<f64>, Ix2, B>
where
    B: DeviceStorageAPI<Complex<f64>, RawVec = Vec<Complex<f64>>>,
{
    type Faer = MatRef<'a, c64>;

    fn into_faer_complex(self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let rawvec = self.data().storage().rawvec();
        let ptr = rawvec.as_ptr() as *const c64;
        unsafe { faer::mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

impl<'a, B> IntoFaerComplex for TensorView<'a, Complex<f32>, Ix2, B>
where
    B: DeviceStorageAPI<Complex<f32>, RawVec = Vec<Complex<f32>>>,
{
    type Faer = MatRef<'a, c32>;

    fn into_faer_complex(self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let rawvec = self.data().storage().rawvec();
        let ptr = rawvec.as_ptr() as *const c32;
        unsafe { faer::mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

macro_rules! impl_into_rstsr {
    ($ty: ty, $ty_faer: ty) => {
        impl<'a> IntoRSTSR for MatRef<'a, $ty_faer> {
            type RSTSR = TensorView<'a, $ty, Ix2, DeviceFaer>;

            fn into_rstsr(self) -> Self::RSTSR {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let row_stride = self.row_stride();
                let col_stride = self.col_stride();
                let ptr = self.as_ptr();

                let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0);
                let (_, upper_bound) = layout.bounds_index().unwrap();
                let rawvec =
                    unsafe { Vec::from_raw_parts(ptr as *mut $ty, upper_bound, upper_bound) };
                let storage = ManuallyDrop::new(Storage::new(rawvec, DeviceFaer::default()));
                let data = DataRef::from_manually_drop(storage);
                let tensor = unsafe { TensorView::new_unchecked(data, layout) };
                return tensor;
            }
        }
        impl<'a> IntoRSTSR for MatMut<'a, $ty_faer> {
            type RSTSR = TensorViewMut<'a, $ty, Ix2, DeviceFaer>;

            fn into_rstsr(self) -> Self::RSTSR {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let row_stride = self.row_stride();
                let col_stride = self.col_stride();
                let ptr = self.as_ptr();

                let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0);
                let (_, upper_bound) = layout.bounds_index().unwrap();
                let rawvec =
                    unsafe { Vec::from_raw_parts(ptr as *mut $ty, upper_bound, upper_bound) };
                let storage = ManuallyDrop::new(Storage::new(rawvec, DeviceFaer::default()));
                let data = DataMut::from_manually_drop(storage);
                let tensor = unsafe { TensorViewMut::new_unchecked(data, layout) };
                return tensor;
            }
        }
    };
}

impl_into_rstsr!(f32, f32);
impl_into_rstsr!(f64, f64);
impl_into_rstsr!(Complex<f32>, c32);
impl_into_rstsr!(Complex<f64>, c64);
