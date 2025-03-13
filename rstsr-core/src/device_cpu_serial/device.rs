use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};

#[derive(Clone, Debug)]
pub struct DeviceCpuSerial;

impl Default for DeviceCpuSerial {
    fn default() -> Self {
        DeviceCpuSerial
    }
}

impl DeviceBaseAPI for DeviceCpuSerial {
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> DeviceRawAPI<T> for DeviceCpuSerial {
    type Raw = Vec<T>;
}

impl<T> DeviceStorageAPI<T> for DeviceCpuSerial {
    fn len<R>(storage: &Storage<R, T, Self>) -> usize
    where
        R: DataAPI<Data = Self::Raw>,
    {
        storage.raw().len()
    }

    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>>
    where
        Self::Raw: Clone,
        R: DataCloneAPI<Data = Self::Raw>,
    {
        Ok(storage.raw().clone())
    }

    fn into_cpu_vec<R>(storage: Storage<R, T, Self>) -> Result<Vec<T>>
    where
        Self::Raw: Clone,
        R: DataCloneAPI<Data = Self::Raw>,
    {
        let (raw, _) = storage.into_raw_parts();
        Ok(raw.into_owned().into_raw())
    }
}

impl<T> DeviceAPI<T> for DeviceCpuSerial {}
impl<T, D> DeviceComplexFloatAPI<T, D> for DeviceCpuSerial
where
    T: ComplexFloat,
    D: DimAPI,
{
}

impl<T, D> DeviceNumAPI<T, D> for DeviceCpuSerial
where
    T: Clone + Num,
    D: DimAPI,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_same_device() {
        let device1 = DeviceCpuSerial;
        let device2 = DeviceCpuSerial;
        assert!(device1.same_device(&device2));
    }

    #[test]
    fn test_cpu_storage_to_vec() {
        let storage = Storage::new(DataOwned::from(vec![1, 2, 3]), DeviceCpuSerial);
        let vec = storage.to_cpu_vec().unwrap();
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_cpu_storage_into_vec() {
        let storage = Storage::new(DataOwned::from(vec![1, 2, 3]), DeviceCpuSerial);
        let vec = storage.into_cpu_vec().unwrap();
        assert_eq!(vec, vec![1, 2, 3]);
    }
}
