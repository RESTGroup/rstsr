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
        // SAFETY: `offset` is within the storage by layout validity (checked when the
        // tensor was constructed).
        let ptr = unsafe { self.raw().as_ptr().add(offset) };
        // SAFETY: strides come from the validated 2-D layout, so faer's element access
        // stays within the slice's allocation; faer only reads through `MatRef`.
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
        // SAFETY: `offset` within the storage by layout validity; the exclusive `&mut`
        // borrow provides write provenance.
        let ptr = unsafe { self.raw_mut().as_mut_ptr().add(offset) };
        // SAFETY: strides from the validated 2-D layout keep all element access within
        // the slice allocation; exclusive `&mut` borrow backs the write access.
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
        // SAFETY: the Vec is a NON-OWNING handle over faer's buffer: wrapped in
        // `ManuallyDrop` + `DataRef` below, it is never deallocated. `upper_bound` is
        // the validated layout bound and faer guarantees the referenced elements are
        // initialized and live for `'a`.
        let raw = unsafe { Vec::from_raw_parts(ptr as *mut T, upper_bound, upper_bound) };
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, DeviceFaer::default());
        let tensor = unsafe { TensorView::new_unchecked(storage, layout) };
        return tensor;
    }
}

impl<T> IntoRSTSR for Mat<T> {
    type RSTSR = Tensor<T, DeviceFaer, Ix2>;

    fn into_rstsr(self) -> Self::RSTSR {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        let ptr = self.as_ptr();
        core::mem::forget(self); // prevent double free

        let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0).unwrap();
        let (_, upper_bound) = layout.bounds_index().unwrap();
        // SAFETY: `self` (faer `Mat`, global-allocator backed) is forgotten, moving its
        // allocation into the `Vec` handle; element count equals the validated layout
        // bound. Caveat: if faer over-allocated or over-aligned, the `Vec` dealloc
        // layout would differ from the alloc layout (benign on mainstream allocators;
        // see audit README).
        let raw = unsafe { Vec::from_raw_parts(ptr as *mut T, upper_bound, upper_bound) };
        let data = DataOwned::from(raw);
        let storage = Storage::new(data, DeviceFaer::default());
        let tensor = unsafe { Tensor::new_unchecked(storage, layout) };
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
        // SAFETY: non-owning handle over faer's column buffer, as in `MatRef` above;
        // never deallocated (`ManuallyDrop` + `DataRef`).
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
        // SAFETY: non-owning handle over faer's buffer with mutable access transferred
        // from the exclusive `MatMut`; never deallocated (`ManuallyDrop` + `DataMut`).
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
