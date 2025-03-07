use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};

// for creation, we use most of the functions from DeviceCpuSerial
impl<T> DeviceCreationAnyAPI<T> for DeviceFaer
where
    Self: DeviceRawAPI<T, Raw = Vec<T>>,
{
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial.empty_impl(len)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }

    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<DataOwned<Vec<T>>, T, Self>>
    where
        T: Clone,
    {
        let storage = DeviceCpuSerial.full_impl(len, fill)?;
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
}

impl<T> DeviceCreationNumAPI<T> for DeviceFaer
where
    T: Num + Clone,
    Self: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial.zeros_impl(len)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }

    fn ones_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial.ones_impl(len)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }

    fn arange_int_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial.arange_int_impl(len)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }
}

impl<T> DeviceCreationPartialOrdNumAPI<T> for DeviceFaer
where
    T: Num + PartialOrd + Clone,
    Self: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn arange_impl(
        &self,
        start: T,
        end: T,
        step: T,
    ) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial.arange_impl(start, end, step)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }
}

impl<T> DeviceCreationComplexFloatAPI<T> for DeviceFaer
where
    T: ComplexFloat + Clone + Send + Sync,
    Self: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn linspace_impl(
        &self,
        start: T,
        end: T,
        n: usize,
        endpoint: bool,
    ) -> Result<Storage<DataOwned<Vec<T>>, T, Self>> {
        let storage = DeviceCpuSerial.linspace_impl(start, end, n, endpoint)?;
        let (data, _) = storage.into_raw_parts();
        Ok(Storage::new(data, self.clone()))
    }
}

impl<T> DeviceCreationTriAPI<T> for DeviceFaer
where
    T: Num + Clone,
    Self: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn tril_impl<D>(&self, raw: &mut Self::Raw, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI,
    {
        DeviceCpuSerial.tril_impl(raw, layout, k)
    }

    fn triu_impl<D>(&self, raw: &mut Self::Raw, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI,
    {
        DeviceCpuSerial.triu_impl(raw, layout, k)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_linspace() {
        let device = DeviceFaer::default();
        let a = linspace((1.0, 5.0, 5, &device));
        assert_eq!(a.raw(), &vec![1., 2., 3., 4., 5.]);
    }
}
