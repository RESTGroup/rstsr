use crate::prelude_dev::*;

impl<'a, R, T, D> DeviceChangeAPI<'a, DeviceCpuSerial, R, T, D> for DeviceCpuSerial
where
    T: Clone + Send + Sync + 'a,
    D: DimAPI,
    R: DataCloneAPI<Data = Vec<T>>,
{
    type Repr = R;
    type ReprTo = DataRef<'a, Vec<T>>;

    fn change_device(
        tensor: TensorAny<R, T, DeviceCpuSerial, D>,
        device: &DeviceCpuSerial,
    ) -> Result<TensorAny<Self::Repr, T, DeviceCpuSerial, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, _) = storage.into_raw_parts();
        let storage = Storage::new(data, device.clone());
        let tensor = TensorAny::new(storage, layout);
        Ok(tensor)
    }

    fn into_device(
        tensor: TensorAny<R, T, DeviceCpuSerial, D>,
        device: &DeviceCpuSerial,
    ) -> Result<TensorAny<DataOwned<Vec<T>>, T, DeviceCpuSerial, D>> {
        let tensor = tensor.into_owned();
        DeviceChangeAPI::change_device(tensor, device)
    }

    fn to_device(
        tensor: &'a TensorAny<R, T, DeviceCpuSerial, D>,
        device: &DeviceCpuSerial,
    ) -> Result<TensorView<'a, T, DeviceCpuSerial, D>> {
        let view = tensor.view();
        DeviceChangeAPI::change_device(view, device)
    }
}
