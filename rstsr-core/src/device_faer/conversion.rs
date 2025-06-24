//! Conversion to/from Faer

use crate::prelude_dev::*;
use core::mem::ManuallyDrop;
use faer::prelude::*;
use faer_ext::IntoFaer;

/* #region conversion to Faer objects */

impl<'a, T, B> IntoFaer for TensorView<'a, T, B, Ix2>
where
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Faer = MatRef<'a, T>;

    fn into_faer(self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let offset = self.offset();
        let ptr = unsafe { self.raw().as_ptr().add(offset) };
        unsafe { MatRef::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

impl<'a, T, B> IntoFaer for TensorViewMut<'a, T, B, Ix2>
where
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Faer = MatMut<'a, T>;

    fn into_faer(mut self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let offset = self.offset();
        let ptr = unsafe { self.raw_mut().as_mut_ptr().add(offset) };
        unsafe { MatMut::from_raw_parts_mut(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

impl<'a, T> IntoRSTSR for MatRef<'a, T> {
    type RSTSR = TensorView<'a, T, DeviceFaer, Ix2>;

    fn into_rstsr(self) -> Self::RSTSR {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        let ptr = self.as_ptr();

        let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0).unwrap();
        let (_, upper_bound) = layout.bounds_index().unwrap();
        let raw = unsafe { Vec::from_raw_parts(ptr as *mut T, upper_bound, upper_bound) };
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, DeviceFaer::default());
        let tensor = unsafe { TensorView::new_unchecked(storage, layout) };
        return tensor;
    }
}

impl<'a, T> IntoRSTSR for ColRef<'a, T> {
    type RSTSR = TensorView<'a, T, DeviceFaer, Ix1>;

    fn into_rstsr(self) -> Self::RSTSR {
        let nrows = self.nrows();
        let stride = self.row_stride();
        let ptr = self.as_ptr();

        let layout = Layout::new([nrows], [stride], 0).unwrap();
        let (_, upper_bound) = layout.bounds_index().unwrap();
        let raw = unsafe { Vec::from_raw_parts(ptr as *mut T, upper_bound, upper_bound) };
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, DeviceFaer::default());
        let tensor = unsafe { TensorView::new_unchecked(storage, layout) };
        return tensor;
    }
}

impl<'a, T> IntoRSTSR for MatMut<'a, T> {
    type RSTSR = TensorViewMut<'a, T, DeviceFaer, Ix2>;

    fn into_rstsr(self) -> Self::RSTSR {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        let ptr = self.as_ptr();

        let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0).unwrap();
        let (_, upper_bound) = layout.bounds_index().unwrap();
        let raw = unsafe { Vec::from_raw_parts(ptr as *mut T, upper_bound, upper_bound) };
        let data = DataMut::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, DeviceFaer::default());
        let tensor = unsafe { TensorMut::new_unchecked(storage, layout) };
        return tensor;
    }
}

/* #endregion */

/* #region device conversion */

#[duplicate_item(
    DevA DevB;
   [DeviceFaer     ] [DeviceCpuSerial];
   [DeviceCpuSerial] [DeviceFaer     ];
   [DeviceFaer     ] [DeviceFaer     ];
)]
impl<'a, R, T, D> DeviceChangeAPI<'a, DevB, R, T, D> for DevA
where
    T: Clone + Send + Sync + 'a,
    D: DimAPI,
    R: DataCloneAPI<Data = Vec<T>>,
{
    type Repr = R;
    type ReprTo = DataRef<'a, Vec<T>>;

    fn change_device(tensor: TensorAny<R, T, DevA, D>, device: &DevB) -> Result<TensorAny<Self::Repr, T, DevB, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, _) = storage.into_raw_parts();
        let storage = Storage::new(data, device.clone());
        let tensor = TensorAny::new(storage, layout);
        Ok(tensor)
    }

    fn into_device(
        tensor: TensorAny<R, T, DevA, D>,
        device: &DevB,
    ) -> Result<TensorAny<DataOwned<Vec<T>>, T, DevB, D>> {
        let tensor = tensor.into_owned();
        DeviceChangeAPI::change_device(tensor, device)
    }

    fn to_device(tensor: &'a TensorAny<R, T, DevA, D>, device: &DevB) -> Result<TensorView<'a, T, DevB, D>> {
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
        let device_serial = DeviceCpuSerial::default();
        let device_faer = DeviceFaer::new(0);
        let a = linspace((1.0, 5.0, 5, &device_faer));
        let b = a.to_device(&device_serial);
        println!("{b:?}");
        let a = linspace((1.0, 5.0, 5, &device_serial));
        let a_view = a.view();
        let b = a_view.to_device(&device_faer);
        println!("{b:?}");
    }

    #[test]
    fn test_self_conversion() {
        let device_a = DeviceFaer::new(1);
        let device_b = DeviceFaer::new(0);
        let a = linspace((1.0, 5.0, 5, &device_b));
        let b = a.to_device(&device_a);
        println!("{b:?}");
        let a = linspace((1.0, 5.0, 5, &device_a));
        let a_view = a.view();
        let b = a_view.to_device(&device_b);
        println!("{b:?}");
    }
}
