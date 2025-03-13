use crate::prelude_dev::*;

#[derive(Clone, Debug, Default)]
pub struct DeviceHDF5;

impl DeviceBaseAPI for DeviceHDF5 {
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> DeviceRawAPI<T> for DeviceHDF5 {
    type Raw = Dataset;
}

impl<T> DeviceStorageAPI<T> for DeviceHDF5
where
    T: H5Type,
{
    fn len<R>(storage: &Storage<R, T, Self>) -> usize
    where
        R: DataAPI<Data = Self::Raw>,
    {
        storage.raw().size()
    }

    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>>
    where
        Self::Raw: Clone,
        R: DataCloneAPI<Data = Self::Raw>,
    {
        storage.raw().read_raw::<T>().map_err(|e| Error::DeviceError(e.to_string()))
    }

    fn into_cpu_vec<R>(storage: Storage<R, T, Self>) -> Result<Vec<T>>
    where
        Self::Raw: Clone,
        R: DataCloneAPI<Data = Self::Raw>,
    {
        Self::to_cpu_vec(&storage)
    }
}

impl<T> DeviceAPI<T> for DeviceHDF5 where T: H5Type {}
