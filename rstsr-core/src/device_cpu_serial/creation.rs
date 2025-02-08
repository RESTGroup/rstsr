use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};

impl<T> DeviceCreationAnyAPI<T> for DeviceCpuSerial
where
    T: Clone,
    DeviceCpuSerial: DeviceRawAPI<T, Raw = Vec<T>>,
{
    unsafe fn empty_impl(
        &self,
        len: usize,
    ) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        let raw = uninitialized_vec(len);
        Ok(Storage::new(raw.into(), self.clone()))
    }

    fn full_impl(
        &self,
        len: usize,
        fill: T,
    ) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        let raw = vec![fill; len];
        Ok(Storage::new(raw.into(), self.clone()))
    }

    fn outof_cpu_vec(&self, vec: Vec<T>) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        Ok(Storage::new(vec.into(), self.clone()))
    }

    fn from_cpu_vec(&self, vec: &[T]) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        Ok(Storage::new(vec.to_vec().into(), self.clone()))
    }
}

impl<T> DeviceCreationNumAPI<T> for DeviceCpuSerial
where
    T: Num + Clone,
    DeviceCpuSerial: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        let raw = vec![T::zero(); len];
        Ok(Storage::new(raw.into(), self.clone()))
    }

    fn ones_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        let raw = vec![T::one(); len];
        Ok(Storage::new(raw.into(), self.clone()))
    }

    fn arange_int_impl(
        &self,
        len: usize,
    ) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        let mut raw = Vec::with_capacity(len);
        let mut v = T::zero();
        for _ in 0..len {
            raw.push(v.clone());
            v = v + T::one();
        }
        Ok(Storage::new(raw.into(), self.clone()))
    }
}

impl<T> DeviceCreationPartialOrdNumAPI<T> for DeviceCpuSerial
where
    T: Num + PartialOrd + Clone,
    DeviceCpuSerial: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn arange_impl(
        &self,
        start: T,
        end: T,
        step: T,
    ) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        rstsr_assert!(step != T::zero(), InvalidValue)?;
        let mut raw = Vec::new();
        let mut current = start.clone();
        while current < end {
            raw.push(current.clone());
            current = current + step.clone();
        }
        Ok(Storage::new(raw.into(), self.clone()))
    }
}

impl<T> DeviceCreationComplexFloatAPI<T> for DeviceCpuSerial
where
    T: ComplexFloat + Clone,
    DeviceCpuSerial: DeviceRawAPI<T, Raw = Vec<T>>,
{
    fn linspace_impl(
        &self,
        start: T,
        end: T,
        n: usize,
        endpoint: bool,
    ) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        // handle special cases
        if n == 0 {
            return Ok(Storage::new(vec![].into(), self.clone()));
        } else if n == 1 {
            return Ok(Storage::new(vec![start].into(), self.clone()));
        }

        let mut raw = Vec::with_capacity(n);
        let step = match endpoint {
            true => (end - start) / T::from(n - 1).unwrap(),
            false => (end - start) / T::from(n).unwrap(),
        };
        let mut v = start;
        for _ in 0..n {
            raw.push(v);
            v = v + step;
        }
        Ok(Storage::new(raw.into(), self.clone()))
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_creation() {
        use super::*;
        use num::Complex;

        let device = DeviceCpuSerial {};
        let storage: Storage<_, f64, _> = device.zeros_impl(10).unwrap();
        println!("{:?}", storage);
        let storage: Storage<_, f64, _> = device.ones_impl(10).unwrap();
        println!("{:?}", storage);
        let storage: Storage<_, f64, _> = device.arange_int_impl(10).unwrap();
        println!("{:?}", storage);
        let storage: Storage<_, f64, _> = unsafe { device.empty_impl(10).unwrap() };
        println!("{:?}", storage);
        let storage = device.from_cpu_vec(&[1.0; 10]).unwrap();
        println!("{:?}", storage);
        let storage = device.outof_cpu_vec(vec![1.0; 10]).unwrap();
        println!("{:?}", storage);
        let storage = device.linspace_impl(0.0, 1.0, 10, true).unwrap();
        println!("{:?}", storage);
        let storage =
            device.linspace_impl(Complex::new(1.0, 2.0), Complex::new(3.5, 4.7), 10, true).unwrap();
        println!("{:?}", storage);
        let storage = device.arange_impl(0.0, 1.0, 0.1).unwrap();
        println!("{:?}", storage);
    }
}
