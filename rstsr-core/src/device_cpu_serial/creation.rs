use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};

impl<T> DeviceCreationAnyAPI<T> for DeviceCpuSerial
where
    Self: DeviceRawAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = Vec<MaybeUninit<T>>>,
{
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        let raw = uninitialized_vec(len)?;
        Ok(Storage::new(raw.into(), self.clone()))
    }

    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>>
    where
        T: Clone,
    {
        let raw = vec![fill; len];
        Ok(Storage::new(raw.into(), self.clone()))
    }

    fn outof_cpu_vec(&self, vec: Vec<T>) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
        Ok(Storage::new(vec.into(), self.clone()))
    }

    fn from_cpu_vec(&self, vec: &[T]) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>>
    where
        T: Clone,
    {
        Ok(Storage::new(vec.to_vec().into(), self.clone()))
    }

    fn uninit_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<MaybeUninit<T>>>, MaybeUninit<T>, Self>>
    where
        Self: DeviceRawAPI<MaybeUninit<T>>,
    {
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

    fn arange_int_impl(&self, len: usize) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
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
    fn arange_impl(&self, start: T, end: T, step: T) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>> {
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

impl<T> DeviceCreationTriAPI<T> for DeviceCpuSerial
where
    T: Num + Clone,
{
    fn tril_impl<D>(&self, raw: &mut Vec<T>, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI,
    {
        tril_cpu_serial(raw, layout, k)
    }

    fn triu_impl<D>(&self, raw: &mut Vec<T>, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI,
    {
        triu_cpu_serial(raw, layout, k)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_creation() {
        use super::*;
        use num::Complex;

        let device = DeviceCpuSerial::default();
        let storage: Storage<_, f64, _> = device.zeros_impl(10).unwrap();
        println!("{storage:?}");
        let storage: Storage<_, f64, _> = device.ones_impl(10).unwrap();
        println!("{storage:?}");
        let storage: Storage<_, f64, _> = device.arange_int_impl(10).unwrap();
        println!("{storage:?}");
        let storage: Storage<_, f64, _> = unsafe { device.empty_impl(10).unwrap() };
        println!("{storage:?}");
        let storage = device.from_cpu_vec(&[1.0; 10]).unwrap();
        println!("{storage:?}");
        let storage = device.outof_cpu_vec(vec![1.0; 10]).unwrap();
        println!("{storage:?}");
        let storage = device.linspace_impl(0.0, 1.0, 10, true).unwrap();
        println!("{storage:?}");
        let storage = device.linspace_impl(Complex::new(1.0, 2.0), Complex::new(3.5, 4.7), 10, true).unwrap();
        println!("{storage:?}");
        let storage = device.arange_impl(0.0, 1.0, 0.1).unwrap();
        println!("{storage:?}");

        // tril/triu
        let mut vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let layout = [3, 3].c();
        device.tril_impl(&mut vec, &layout, -1).unwrap();
        println!("{vec:?}");
        assert_eq!(vec, vec![0, 0, 0, 4, 0, 0, 7, 8, 0]);
        let mut vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        device.triu_impl(&mut vec, &layout, -1).unwrap();
        println!("{vec:?}");
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6, 0, 8, 9]);
    }
}
