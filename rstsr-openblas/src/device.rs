use crate::prelude_dev::*;
use num::complex::ComplexFloat;

impl DeviceBLAS {
    pub fn new(num_threads: usize) -> Self {
        DeviceBLAS { base: DeviceCpuRayon::new(num_threads) }
    }

    pub fn var_num_threads(&self) -> usize {
        self.base.var_num_threads()
    }
}

impl DeviceRayonAPI for DeviceBLAS {
    fn set_num_threads(&mut self, num_threads: usize) {
        self.base.set_num_threads(num_threads);
    }

    fn get_num_threads(&self) -> usize {
        self.base.get_num_threads()
    }

    fn get_pool(&self) -> &rayon::ThreadPool {
        self.base.get_pool()
    }

    fn get_serial_pool(&self) -> &rayon::ThreadPool {
        self.base.get_serial_pool()
    }
}

impl Default for DeviceBLAS {
    fn default() -> Self {
        DeviceBLAS::new(0)
    }
}

impl DeviceBaseAPI for DeviceBLAS {
    fn same_device(&self, other: &Self) -> bool {
        self.var_num_threads() == other.var_num_threads()
    }
}

impl<T> DeviceRawAPI<T> for DeviceBLAS
where
    T: Clone,
{
    type Raw = Vec<T>;
}

impl<T> DeviceStorageAPI<T> for DeviceBLAS
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
        &storage.raw()[index] as *const T
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

impl<T> DeviceAPI<T> for DeviceBLAS where T: Clone {}
impl<T, D> DeviceComplexFloatAPI<T, D> for DeviceBLAS
where
    T: ComplexFloat + Send + Sync,
    D: DimAPI,
{
}
