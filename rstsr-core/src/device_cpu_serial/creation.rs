use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};

impl<T> DeviceCreationAnyAPI<T> for DeviceCpuSerial
where
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
    ) -> Result<Storage<DataOwned<Vec<T>>, T, DeviceCpuSerial>>
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

pub fn tril_cpu_serial<T, D>(raw: &mut [T], layout: &Layout<D>, k: isize) -> Result<()>
where
    T: Num + Clone,
    D: DimAPI,
{
    let (la_rest, la_ix2) = layout.dim_split_at(-2)?;
    let mut la_ix2 = la_ix2.into_dim::<Ix2>()?;
    for offset in IterLayoutColMajor::new(&la_rest)? {
        unsafe { la_ix2.set_offset(offset) };
        tril_ix2_cpu_serial(raw, &la_ix2, k)?;
    }
    Ok(())
}

pub fn tril_ix2_cpu_serial<T>(raw: &mut [T], layout: &Layout<Ix2>, k: isize) -> Result<()>
where
    T: Num + Clone,
{
    let [nrow, ncol] = *layout.shape();
    for i in 0..nrow {
        let j_start = (i as isize + k + 1).max(0) as usize;
        for j in j_start..ncol {
            unsafe {
                raw[layout.index_uncheck(&[i, j]) as usize] = T::zero();
            }
        }
    }
    Ok(())
}

pub fn triu_cpu_serial<T, D>(raw: &mut [T], layout: &Layout<D>, k: isize) -> Result<()>
where
    T: Num + Clone,
    D: DimAPI,
{
    let (la_rest, la_ix2) = layout.dim_split_at(-2)?;
    let mut la_ix2 = la_ix2.into_dim::<Ix2>()?;
    for offset in IterLayoutColMajor::new(&la_rest)? {
        unsafe { la_ix2.set_offset(offset) };
        triu_ix2_cpu_serial(raw, &la_ix2, k)?;
    }
    Ok(())
}

pub fn triu_ix2_cpu_serial<T>(raw: &mut [T], layout: &Layout<Ix2>, k: isize) -> Result<()>
where
    T: Num + Clone,
{
    let [nrow, _] = *layout.shape();
    for i in 0..nrow {
        let j_end = (i as isize + k).max(0) as usize;
        for j in 0..j_end {
            unsafe {
                raw[layout.index_uncheck(&[i, j]) as usize] = T::zero();
            }
        }
    }
    Ok(())
}

impl<T> DeviceCreationTriAPI<T> for DeviceCpuSerial
where
    T: Num + Clone,
    DeviceCpuSerial: DeviceRawAPI<T, Raw = Vec<T>>,
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

        // tril/triu
        let mut vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let layout = [3, 3].c();
        device.tril_impl(&mut vec, &layout, -1).unwrap();
        println!("{:?}", vec);
        assert_eq!(vec, vec![0, 0, 0, 4, 0, 0, 7, 8, 0]);
        let mut vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        device.triu_impl(&mut vec, &layout, -1).unwrap();
        println!("{:?}", vec);
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6, 0, 8, 9]);
    }
}
