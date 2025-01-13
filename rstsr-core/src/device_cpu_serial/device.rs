use crate::prelude_dev::*;

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

impl<T> DeviceRawAPI<T> for DeviceCpuSerial
where
    T: Clone,
{
    type Raw = Vec<T>;
}

impl<T> DeviceStorageAPI<T> for DeviceCpuSerial
where
    T: Clone,
{
    fn len<R>(storage: &Storage<R, T, Self>) -> usize
    where
        R: DataAPI<Data = Self::Raw>,
    {
        storage.raw().len()
    }

    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataAPI<Data = Self::Raw>,
    {
        Ok(storage.raw().clone())
    }

    fn into_cpu_vec<R>(storage: Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataAPI<Data = Self::Raw>,
    {
        let (raw, _) = storage.into_raw_parts();
        Ok(raw.into_owned().into_raw())
    }

    #[inline]
    fn get_index<R>(storage: &Storage<R, T, Self>, index: usize) -> T
    where
        R: DataAPI<Data = Self::Raw>,
    {
        storage.raw()[index].clone()
    }

    #[inline]
    fn get_index_ptr<R>(storage: &Storage<R, T, Self>, index: usize) -> *const T
    where
        R: DataAPI<Data = Self::Raw>,
    {
        storage.raw().get(index).unwrap() as *const T
    }

    #[inline]
    fn get_index_mut_ptr<R>(storage: &mut Storage<R, T, Self>, index: usize) -> *mut T
    where
        R: DataMutAPI<Data = Self::Raw>,
    {
        storage.raw_mut().get_mut(index).unwrap() as *mut T
    }

    #[inline]
    fn set_index<R>(storage: &mut Storage<R, T, Self>, index: usize, value: T)
    where
        R: DataMutAPI<Data = Self::Raw>,
    {
        storage.raw_mut()[index] = value;
    }
}

impl<T> DeviceAPI<T> for DeviceCpuSerial where T: Clone {}

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
