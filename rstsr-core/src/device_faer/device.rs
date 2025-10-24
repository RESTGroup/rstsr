use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};
use rstsr_dtype_traits::DTypeIntoFloatAPI;

#[derive(Clone, Debug)]
pub struct DeviceFaer {
    base: DeviceCpuRayon,
}

pub(crate) use DeviceFaer as DeviceRayonAutoImpl;

impl DeviceFaer {
    pub fn new(num_threads: usize) -> Self {
        DeviceFaer { base: DeviceCpuRayon::new(num_threads) }
    }
}

impl DeviceRayonAPI for DeviceFaer {
    #[inline]
    fn set_num_threads(&mut self, num_threads: usize) {
        self.base.set_num_threads(num_threads);
    }

    #[inline]
    fn get_num_threads(&self) -> usize {
        self.base.get_num_threads()
    }

    #[inline]
    fn get_pool(&self) -> &ThreadPool {
        self.base.get_pool()
    }

    #[inline]
    fn get_current_pool(&self) -> Option<&ThreadPool> {
        self.base.get_current_pool()
    }
}

impl Default for DeviceFaer {
    fn default() -> Self {
        DeviceFaer::new(0)
    }
}

impl DeviceBaseAPI for DeviceFaer {
    fn same_device(&self, other: &Self) -> bool {
        let same_num_threads = self.get_num_threads() == other.get_num_threads();
        let same_default_order = self.default_order() == other.default_order();
        same_num_threads && same_default_order
    }

    fn default_order(&self) -> FlagOrder {
        self.base.default_order()
    }

    fn set_default_order(&mut self, order: FlagOrder) {
        self.base.set_default_order(order);
    }
}

impl<T> DeviceRawAPI<T> for DeviceFaer {
    type Raw = Vec<T>;
}

impl<T> DeviceStorageAPI<T> for DeviceFaer {
    fn len<R>(storage: &Storage<R, T, Self>) -> usize
    where
        R: DataAPI<Data = Self::Raw>,
    {
        storage.raw().len()
    }

    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>>
    where
        Self::Raw: Clone,
        R: DataAPI<Data = Self::Raw>,
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

    #[inline]
    fn get_index<R>(storage: &Storage<R, T, Self>, index: usize) -> T
    where
        T: Clone,
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

impl<T> DeviceAPI<T> for DeviceFaer {}

impl<T, D> DeviceComplexFloatAPI<T, D> for DeviceFaer
where
    T: ComplexFloat + DTypeIntoFloatAPI<FloatType = T> + Send + Sync,
    T::Real: DTypeIntoFloatAPI<FloatType = T::Real> + Send + Sync,
    D: DimAPI,
{
}

impl<T, D> DeviceNumAPI<T, D> for DeviceFaer
where
    T: Clone + Num + Send + Sync,
    D: DimAPI,
{
}
