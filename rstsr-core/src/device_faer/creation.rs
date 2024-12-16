use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Float, Num};

// for creation, we use most of the functions from DeviceCpuSerial
impl<T> DeviceCreationAnyAPI<T> for DeviceFaer
where
    T: Clone,
    Self: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<T, Self>> {
        let storage = DeviceCpuSerial.empty_impl(len)?;
        Ok(Storage::new(storage.into_rawvec(), self.clone()))
    }

    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<T, Self>> {
        let storage = DeviceCpuSerial.full_impl(len, fill)?;
        Ok(Storage::new(storage.into_rawvec(), self.clone()))
    }

    fn outof_cpu_vec(&self, vec: Vec<T>) -> Result<Storage<T, Self>> {
        Ok(Storage::new(vec, self.clone()))
    }

    fn from_cpu_vec(&self, vec: &[T]) -> Result<Storage<T, Self>> {
        let rawvec = vec.to_vec();
        Ok(Storage::new(rawvec, self.clone()))
    }
}

impl<T> DeviceCreationNumAPI<T> for DeviceFaer
where
    T: Num + Clone,
    Self: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<T, Self>> {
        let storage = DeviceCpuSerial.zeros_impl(len)?;
        Ok(Storage::new(storage.into_rawvec(), self.clone()))
    }

    fn ones_impl(&self, len: usize) -> Result<Storage<T, Self>> {
        let storage = DeviceCpuSerial.ones_impl(len)?;
        Ok(Storage::new(storage.into_rawvec(), self.clone()))
    }

    fn arange_int_impl(&self, len: usize) -> Result<Storage<T, Self>> {
        let storage = DeviceCpuSerial.arange_int_impl(len)?;
        Ok(Storage::new(storage.into_rawvec(), self.clone()))
    }
}

impl<T> DeviceCreationComplexFloatAPI<T> for DeviceFaer
where
    T: ComplexFloat + Clone + Send + Sync,
    Self: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    fn linspace_impl(
        &self,
        start: T,
        end: T,
        n: usize,
        endpoint: bool,
    ) -> Result<Storage<T, Self>> {
        let storage = DeviceCpuSerial.linspace_impl(start, end, n, endpoint)?;
        Ok(Storage::new(storage.into_rawvec(), self.clone()))
    }
}

impl<T> DeviceCreationFloatAPI<T> for DeviceFaer
where
    T: Float + Clone + Debug + Send + Sync,
    Self: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    fn arange_impl(&self, start: T, end: T, step: T) -> Result<Storage<T, Self>> {
        let storage = DeviceCpuSerial.arange_impl(start, end, step)?;
        Ok(Storage::new(storage.into_rawvec(), self.clone()))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_linspace() {
        let device = DeviceFaer::default();
        let a = Tensor::linspace(1.0, 5.0, 5, &device);
        assert_eq!(a.data().storage().rawvec(), &vec![1., 2., 3., 4., 5.]);
    }
}
