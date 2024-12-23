use core::mem::ManuallyDrop;

use crate::prelude_dev::*;

pub trait TensorToDeviceAPI<B> {
    type DataRepr;
    type Dim: DimAPI;
    fn into_device(self, device: &B) -> Result<TensorBase<Self::DataRepr, Self::Dim>>;
}

impl<T, D, B1, B2> TensorToDeviceAPI<B2> for Tensor<T, D, B1>
where
    D: DimAPI,
    B1: DeviceAPI<T>,
    B2: DeviceAPI<T>,
    Storage<T, B1>: DeviceStorageConversionAPI<B2, T = T>,
{
    type DataRepr = DataOwned<Storage<T, B2>>;
    type Dim = D;
    fn into_device(self, device: &B2) -> Result<Tensor<T, D, B2>> {
        let layout = self.layout().clone();
        let storage = self.into_data().into_storage().into_device(device)?;
        Tensor::new_f(DataOwned::from(storage), layout)
    }
}

impl<'a, T, D, B1, B2> TensorToDeviceAPI<B2> for TensorView<'a, T, D, B1>
where
    D: DimAPI,
    B1: DeviceAPI<T, RawVec = Vec<T>>,
    B2: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    type DataRepr = DataRef<'a, Storage<T, B2>>;
    type Dim = D;
    fn into_device(self, device: &B2) -> Result<TensorView<'a, T, D, B2>> {
        let layout = self.layout().clone();
        let oldvec = self.storage().rawvec();

        let ptr = oldvec.as_ptr();
        let len = oldvec.len();
        let rawvec: Vec<T> = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let storage = ManuallyDrop::new(Storage::new(rawvec, device.clone()));
        let data = DataRef::from_manually_drop(storage);
        let tensor_view = unsafe { TensorView::new_unchecked(data, layout) };
        Ok(tensor_view)
    }
}
