use crate::prelude_dev::*;

macro_rules! impl_change_device {
    ($Device: ty) => {
        impl<'a, R, T, D> DeviceChangeAPI<'a, $Device, R, T, D> for DeviceBLAS
        where
            T: Clone + Send + Sync + 'a,
            D: DimAPI,
            R: DataCloneAPI<Data = Vec<T>>,
        {
            type Repr = R;
            type ReprTo = DataRef<'a, Vec<T>>;

            fn change_device(
                tensor: TensorAny<R, T, DeviceBLAS, D>,
                device: &$Device,
            ) -> Result<TensorAny<Self::Repr, T, $Device, D>> {
                let (storage, layout) = tensor.into_raw_parts();
                let (data, _) = storage.into_raw_parts();
                let storage = Storage::new(data, device.clone());
                let tensor = TensorAny::new(storage, layout);
                Ok(tensor)
            }

            fn into_device(
                tensor: TensorAny<R, T, DeviceBLAS, D>,
                device: &$Device,
            ) -> Result<TensorAny<DataOwned<Vec<T>>, T, $Device, D>> {
                let tensor = tensor.into_owned();
                DeviceChangeAPI::change_device(tensor, device)
            }

            fn to_device(
                tensor: &'a TensorAny<R, T, DeviceBLAS, D>,
                device: &$Device,
            ) -> Result<TensorView<'a, T, $Device, D>> {
                let view = tensor.view();
                DeviceChangeAPI::change_device(view, device)
            }
        }

        impl<'a, R, T, D> DeviceChangeAPI<'a, DeviceBLAS, R, T, D> for $Device
        where
            T: Clone + Send + Sync + 'a,
            D: DimAPI,
            R: DataCloneAPI<Data = Vec<T>>,
        {
            type Repr = R;
            type ReprTo = DataRef<'a, Vec<T>>;

            fn change_device(
                tensor: TensorAny<R, T, $Device, D>,
                device: &DeviceBLAS,
            ) -> Result<TensorAny<Self::Repr, T, DeviceBLAS, D>> {
                let (storage, layout) = tensor.into_raw_parts();
                let (data, _) = storage.into_raw_parts();
                let storage = Storage::new(data, device.clone());
                let tensor = TensorAny::new(storage, layout);
                Ok(tensor)
            }

            fn into_device(
                tensor: TensorAny<R, T, $Device, D>,
                device: &DeviceBLAS,
            ) -> Result<TensorAny<DataOwned<Vec<T>>, T, DeviceBLAS, D>> {
                let tensor = tensor.into_owned();
                DeviceChangeAPI::change_device(tensor, device)
            }

            fn to_device(
                tensor: &'a TensorAny<R, T, $Device, D>,
                device: &DeviceBLAS,
            ) -> Result<TensorView<'a, T, DeviceBLAS, D>> {
                let view = tensor.view();
                DeviceChangeAPI::change_device(view, device)
            }
        }
    };
}

impl_change_device!(DeviceCpuSerial);
#[cfg(feature = "faer")]
impl_change_device!(DeviceFaer);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_conversion_cpu_serial() {
        let device_serial = DeviceCpuSerial {};
        let device_openblas = DeviceBLAS::new(0);
        let a = linspace((1.0, 5.0, 5, &device_openblas));
        let b = a.to_device(&device_serial);
        println!("{:?}", b);
        let a = linspace((1.0, 5.0, 5, &device_serial));
        let a_view = a.view();
        let b = a_view.to_device(&device_openblas);
        println!("{:?}", b);
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_device_conversion_faer() {
        let device_faer = DeviceFaer::new(0);
        let device_openblas = DeviceBLAS::new(0);
        let a = linspace((1.0, 5.0, 5, &device_openblas));
        let b = a.to_device(&device_faer);
        println!("{:?}", b);
        let a = linspace((1.0, 5.0, 5, &device_faer));
        let a_view = a.view();
        let b = a_view.to_device(&device_openblas);
        println!("{:?}", b);
    }
}
