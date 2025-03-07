//! Conversion to/from Faer

use crate::prelude_dev::*;
use core::mem::ManuallyDrop;
use faer::complex_native::{c32, c64};
use faer::{MatMut, MatRef, SimpleEntity};
use faer_ext::{IntoFaer, IntoFaerComplex};
use num::Complex;

/* #region conversion to Faer objects */

impl<'a, T, B> IntoFaer for TensorView<'a, T, B, Ix2>
where
    T: SimpleEntity,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Faer = MatRef<'a, T>;

    fn into_faer(self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let ptr = self.raw().as_ptr();
        unsafe { faer::mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

impl<'a, T, B> IntoFaer for TensorViewMut<'a, T, B, Ix2>
where
    T: SimpleEntity,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Faer = MatMut<'a, T>;

    fn into_faer(mut self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let ptr = self.raw_mut().as_mut_ptr();
        unsafe { faer::mat::from_raw_parts_mut(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

impl<'a, B> IntoFaerComplex for TensorView<'a, Complex<f64>, B, Ix2>
where
    B: DeviceAPI<Complex<f64>, Raw = Vec<Complex<f64>>>,
{
    type Faer = MatRef<'a, c64>;

    fn into_faer_complex(self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let ptr = self.raw().as_ptr() as *const c64;
        unsafe { faer::mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

impl<'a, B> IntoFaerComplex for TensorView<'a, Complex<f32>, B, Ix2>
where
    B: DeviceAPI<Complex<f32>, Raw = Vec<Complex<f32>>>,
{
    type Faer = MatRef<'a, c32>;

    fn into_faer_complex(self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let ptr = self.raw().as_ptr() as *const c32;
        unsafe { faer::mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

macro_rules! impl_into_rstsr {
    ($ty: ty, $ty_faer: ty) => {
        impl<'a> IntoRSTSR for MatRef<'a, $ty_faer> {
            type RSTSR = TensorView<'a, $ty, DeviceFaer, Ix2>;

            fn into_rstsr(self) -> Self::RSTSR {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let row_stride = self.row_stride();
                let col_stride = self.col_stride();
                let ptr = self.as_ptr();

                let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0).unwrap();
                let (_, upper_bound) = layout.bounds_index().unwrap();
                let raw = unsafe { Vec::from_raw_parts(ptr as *mut $ty, upper_bound, upper_bound) };
                let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
                let storage = Storage::new(data, DeviceFaer::default());
                let tensor = unsafe { TensorView::new_unchecked(storage, layout) };
                return tensor;
            }
        }

        impl<'a> IntoRSTSR for MatMut<'a, $ty_faer> {
            type RSTSR = TensorViewMut<'a, $ty, DeviceFaer, Ix2>;

            fn into_rstsr(self) -> Self::RSTSR {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let row_stride = self.row_stride();
                let col_stride = self.col_stride();
                let ptr = self.as_ptr();

                let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0).unwrap();
                let (_, upper_bound) = layout.bounds_index().unwrap();
                let raw = unsafe { Vec::from_raw_parts(ptr as *mut $ty, upper_bound, upper_bound) };
                let data = DataMut::from_manually_drop(ManuallyDrop::new(raw));
                let storage = Storage::new(data, DeviceFaer::default());
                let tensor = unsafe { TensorMut::new_unchecked(storage, layout) };
                return tensor;
            }
        }
    };
}

impl_into_rstsr!(f32, f32);
impl_into_rstsr!(f64, f64);
impl_into_rstsr!(Complex<f32>, c32);
impl_into_rstsr!(Complex<f64>, c64);

/* #endregion */

/* #region device conversion */

impl<'a, R, T, D> DeviceChangeAPI<'a, DeviceCpuSerial, R, T, D> for DeviceFaer
where
    T: Clone + Send + Sync + 'a,
    D: DimAPI,
    R: DataCloneAPI<Data = Vec<T>>,
{
    type Repr = R;
    type ReprTo = DataRef<'a, Vec<T>>;

    fn change_device(
        tensor: TensorAny<R, T, DeviceFaer, D>,
        device: &DeviceCpuSerial,
    ) -> Result<TensorAny<Self::Repr, T, DeviceCpuSerial, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, _) = storage.into_raw_parts();
        let storage = Storage::new(data, device.clone());
        let tensor = TensorAny::new(storage, layout);
        Ok(tensor)
    }

    fn into_device(
        tensor: TensorAny<R, T, DeviceFaer, D>,
        device: &DeviceCpuSerial,
    ) -> Result<TensorAny<DataOwned<Vec<T>>, T, DeviceCpuSerial, D>> {
        let tensor = tensor.into_owned();
        DeviceChangeAPI::change_device(tensor, device)
    }

    fn to_device(
        tensor: &'a TensorAny<R, T, DeviceFaer, D>,
        device: &DeviceCpuSerial,
    ) -> Result<TensorView<'a, T, DeviceCpuSerial, D>> {
        let view = tensor.view();
        DeviceChangeAPI::change_device(view, device)
    }
}

impl<'a, R, T, D> DeviceChangeAPI<'a, DeviceFaer, R, T, D> for DeviceCpuSerial
where
    T: Clone + Send + Sync + 'a,
    D: DimAPI,
    R: DataCloneAPI<Data = Vec<T>>,
{
    type Repr = R;
    type ReprTo = DataRef<'a, Vec<T>>;

    fn change_device(
        tensor: TensorAny<R, T, DeviceCpuSerial, D>,
        device: &DeviceFaer,
    ) -> Result<TensorAny<Self::Repr, T, DeviceFaer, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, _) = storage.into_raw_parts();
        let storage = Storage::new(data, device.clone());
        let tensor = TensorAny::new(storage, layout);
        Ok(tensor)
    }

    fn into_device(
        tensor: TensorAny<R, T, DeviceCpuSerial, D>,
        device: &DeviceFaer,
    ) -> Result<TensorAny<DataOwned<Vec<T>>, T, DeviceFaer, D>> {
        let tensor = tensor.into_owned();
        DeviceChangeAPI::change_device(tensor, device)
    }

    fn to_device(
        tensor: &'a TensorAny<R, T, DeviceCpuSerial, D>,
        device: &DeviceFaer,
    ) -> Result<TensorView<'a, T, DeviceFaer, D>> {
        let view = tensor.view();
        DeviceChangeAPI::change_device(view, device)
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_conversion() {
        let device_serial = DeviceCpuSerial {};
        let device_faer = DeviceFaer::new(0);
        let a = linspace((1.0, 5.0, 5, &device_faer));
        let b = a.to_device(&device_serial);
        println!("{:?}", b);
        let a = linspace((1.0, 5.0, 5, &device_serial));
        let a_view = a.view();
        let b = a_view.to_device(&device_faer);
        println!("{:?}", b);
    }
}
