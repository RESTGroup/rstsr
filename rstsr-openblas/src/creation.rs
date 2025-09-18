use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};

// for creation, we use most of the functions from DeviceCpuSerial
impl<T> DeviceCreationAnyAPI<T> for DeviceBLAS
where
    Self: DeviceRawAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = Vec<MaybeUninit<T>>>,
{
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial::default().empty_impl(len)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }

    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<DataOwned<Vec<T>>, T, Self>>
    where
        T: Clone,
    {
        let storage = DeviceCpuSerial::default().full_impl(len, fill)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }

    fn outof_cpu_vec(&self, vec: Vec<T>) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        Ok(Storage::new(DataOwned::from(vec), self.clone()))
    }

    fn from_cpu_vec(&self, vec: &[T]) -> Result<Storage<DataOwned<Vec<T>>, T, Self>>
    where
        T: Clone,
    {
        let raw = vec.to_vec();
        Ok(Storage::new(DataOwned::from(raw), self.clone()))
    }

    fn uninit_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<MaybeUninit<T>>>, MaybeUninit<T>, Self>> {
        let raw = unsafe { uninitialized_vec(len) }?;
        Ok(Storage::new(raw.into(), self.clone()))
    }

    unsafe fn assume_init_impl(
        storage: Storage<DataOwned<Vec<MaybeUninit<T>>>, MaybeUninit<T>, Self>,
    ) -> Result<Storage<DataOwned<Vec<T>>, T, Self>>
    where
        Self: DeviceRawAPI<MaybeUninit<T>>,
    {
        let (data, device) = storage.into_raw_parts();
        let vec = data.into_raw();
        // transmute `Vec<MaybeUninit<T>>` to `Vec<T>`
        let vec = core::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(vec);
        let data = vec.into();
        Ok(Storage::new(data, device))
    }
}

impl<T> DeviceCreationNumAPI<T> for DeviceBLAS
where
    T: Num + Clone,
    Self: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial::default().zeros_impl(len)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }

    fn ones_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial::default().ones_impl(len)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }

    fn arange_int_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial::default().arange_int_impl(len)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }
}

impl<T> DeviceCreationPartialOrdNumAPI<T> for DeviceBLAS
where
    T: Num + PartialOrd + Clone,
    Self: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn arange_impl(&self, start: T, end: T, step: T) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial::default().arange_impl(start, end, step)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }
}

impl<T> DeviceCreationComplexFloatAPI<T> for DeviceBLAS
where
    T: ComplexFloat + Clone + Send + Sync,
    Self: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn linspace_impl(&self, start: T, end: T, n: usize, endpoint: bool) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial::default().linspace_impl(start, end, n, endpoint)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }
}

impl<T> DeviceCreationTriAPI<T> for DeviceBLAS
where
    T: Num + Clone,
    Self: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn tril_impl<D>(&self, raw: &mut Self::Raw, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI,
    {
        DeviceCpuSerial::default().tril_impl(raw, layout, k)
    }

    fn triu_impl<D>(&self, raw: &mut Self::Raw, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI,
    {
        DeviceCpuSerial::default().triu_impl(raw, layout, k)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_linspace() {
        let device = DeviceBLAS::default();
        let a = linspace((1.0, 5.0, 5, &device));
        assert_eq!(a.raw(), &vec![1., 2., 3., 4., 5.]);
    }
}
