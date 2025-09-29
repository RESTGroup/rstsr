use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};
use rstsr_dtype_traits::PromotionSpecialAPI;

#[derive(Clone, Debug, Default)]
pub struct DeviceCpuSerial {
    default_order: FlagOrder,
}

impl DeviceBaseAPI for DeviceCpuSerial {
    fn same_device(&self, other: &Self) -> bool {
        self.default_order == other.default_order
    }

    fn default_order(&self) -> FlagOrder {
        self.default_order
    }

    fn set_default_order(&mut self, order: FlagOrder) {
        self.default_order = order;
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

impl<T> DeviceAPI<T> for DeviceCpuSerial {}

impl<T, D> DeviceComplexFloatAPI<T, D> for DeviceCpuSerial
where
    T: ComplexFloat + PromotionSpecialAPI<FloatType = T>,
    T::Real: PromotionSpecialAPI<FloatType = T::Real>,
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
        let device1 = DeviceCpuSerial::default();
        let device2 = DeviceCpuSerial::default();
        assert!(device1.same_device(&device2));
    }

    #[test]
    fn test_cpu_storage_to_vec() {
        let storage = Storage::new(DataOwned::from(vec![1, 2, 3]), DeviceCpuSerial::default());
        let vec = storage.to_cpu_vec().unwrap();
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_cpu_storage_into_vec() {
        let storage = Storage::new(DataOwned::from(vec![1, 2, 3]), DeviceCpuSerial::default());
        let vec = storage.into_cpu_vec().unwrap();
        assert_eq!(vec, vec![1, 2, 3]);
    }
}
