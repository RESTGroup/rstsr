use crate::prelude_dev::*;

macro_rules! impl_change_device {
    ($DevA: ty, $DevB: ty) => {
        impl<'a, R, T, D> DeviceChangeAPI<'a, $DevB, R, T, D> for $DevA
        where
            T: Clone + Send + Sync + 'a,
            D: DimAPI,
            R: DataCloneAPI<Data = Vec<T>>,
        {
            type Repr = R;
            type ReprTo = DataRef<'a, Vec<T>>;

            fn change_device(
                tensor: TensorAny<R, T, $DevA, D>,
                device: &$DevB,
            ) -> Result<TensorAny<Self::Repr, T, $DevB, D>> {
                let (storage, layout) = tensor.into_raw_parts();
                let (data, _) = storage.into_raw_parts();
                let storage = Storage::new(data, device.clone());
                let tensor = TensorAny::new(storage, layout);
                Ok(tensor)
            }

            fn into_device(
                tensor: TensorAny<R, T, $DevA, D>,
                device: &$DevB,
            ) -> Result<TensorAny<DataOwned<Vec<T>>, T, $DevB, D>> {
                let tensor = tensor.into_owned();
                DeviceChangeAPI::change_device(tensor, device)
            }

            fn to_device(tensor: &'a TensorAny<R, T, $DevA, D>, device: &$DevB) -> Result<TensorView<'a, T, $DevB, D>> {
                let view = tensor.view();
                DeviceChangeAPI::change_device(view, device)
            }
        }
    };
}

impl_change_device!(DeviceCpuSerial, DeviceBLAS);
impl_change_device!(DeviceBLAS, DeviceCpuSerial);
impl_change_device!(DeviceBLAS, DeviceBLAS);
#[cfg(feature = "faer")]
impl_change_device!(DeviceFaer, DeviceBLAS);
#[cfg(feature = "faer")]
impl_change_device!(DeviceBLAS, DeviceFaer);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_conversion_cpu_serial() {
        let device_serial = DeviceCpuSerial::default();
        let device_openblas = DeviceBLAS::new(0);
        let a = linspace((1.0, 5.0, 5, &device_openblas));
        let b = a.to_device(&device_serial);
        println!("{b:?}");
        let a = linspace((1.0, 5.0, 5, &device_serial));
        let a_view = a.view();
        let b = a_view.to_device(&device_openblas);
        println!("{b:?}");
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_device_conversion_faer() {
        let device_faer = DeviceFaer::new(0);
        let device_openblas = DeviceBLAS::new(0);
        let a = linspace((1.0, 5.0, 5, &device_openblas));
        let b = a.to_device(&device_faer);
        println!("{b:?}");
        let a = linspace((1.0, 5.0, 5, &device_faer));
        let a_view = a.view();
        let b = a_view.to_device(&device_openblas);
        println!("{b:?}");
    }
}
