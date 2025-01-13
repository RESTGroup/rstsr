use crate::prelude_dev::*;

pub trait TensorToDeviceAPI<B> {
    type Storage;
    type Dim: DimAPI;
    fn into_device(self, device: &B) -> Result<TensorBase<Self::Storage, Self::Dim>>;
}

impl<R, T, D, B1, B2> TensorToDeviceAPI<B2> for TensorAny<R, T, B1, D>
where
    D: DimAPI,
    B1: DeviceAPI<T>,
    B2: DeviceAPI<T, Raw = B1::Raw>,
    R: DataAPI<Data = B1::Raw>,
    B1: DeviceStorageConversionAPI<R, T, B2>,
{
    type Storage = Storage<R, T, B2>;
    type Dim = D;
    fn into_device(self, device: &B2) -> Result<TensorAny<R, T, B2, D>> {
        let (storage, layout) = self.into_raw_parts();
        let (data, _) = storage.into_raw_parts();
        let storage = Storage::new(data, device.clone());
        TensorBase::new_f(storage, layout)
    }
}
