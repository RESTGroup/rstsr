use crate::prelude_dev::*;

#[derive(Clone, Debug)]
pub struct DeviceOpenBLAS {
    base: DeviceCpuRayon,
}

impl DeviceOpenBLAS {
    pub fn new(num_threads: usize) -> Self {
        DeviceOpenBLAS { base: DeviceCpuRayon::new(num_threads) }
    }

    pub fn var_num_threads(&self) -> usize {
        self.base.var_num_threads()
    }
}

impl DeviceRayonAPI for DeviceOpenBLAS {
    fn set_num_threads(&mut self, num_threads: usize) {
        self.base.set_num_threads(num_threads);
    }

    fn get_num_threads(&self) -> usize {
        self.base.get_num_threads()
    }

    fn get_pool(&self, n: usize) -> Result<rayon::ThreadPool> {
        self.base.get_pool(n)
    }
}

impl Default for DeviceOpenBLAS {
    fn default() -> Self {
        DeviceOpenBLAS::new(0)
    }
}

impl DeviceBaseAPI for DeviceOpenBLAS {
    fn same_device(&self, other: &Self) -> bool {
        self.var_num_threads() == other.var_num_threads()
    }
}

impl<T> DeviceRawVecAPI<T> for DeviceOpenBLAS
where
    T: Clone,
{
    type RawVec = Vec<T>;
}

impl<T> DeviceStorageAPI<T> for DeviceOpenBLAS
where
    T: Clone,
{
    fn new(vector: Self::RawVec, device: Self) -> Storage<T, Self> {
        Storage::<T, Self>::new(vector, device)
    }

    fn len(storage: &Storage<T, Self>) -> usize {
        storage.rawvec().len()
    }

    fn to_cpu_vec(storage: &Storage<T, Self>) -> Result<Vec<T>> {
        Ok(storage.rawvec().clone())
    }

    fn into_cpu_vec(storage: Storage<T, Self>) -> Result<Vec<T>> {
        Ok(storage.into_rawvec())
    }

    #[inline]
    fn get_index(storage: &Storage<T, Self>, index: usize) -> T {
        storage.rawvec()[index].clone()
    }

    #[inline]
    fn get_index_ptr(storage: &Storage<T, Self>, index: usize) -> *const T {
        &storage.rawvec()[index] as *const T
    }

    #[inline]
    fn get_index_mut_ptr(storage: &mut Storage<T, Self>, index: usize) -> *mut T {
        &mut storage.rawvec_mut()[index] as *mut T
    }

    #[inline]
    fn set_index(storage: &mut Storage<T, Self>, index: usize, value: T) {
        storage.rawvec_mut()[index] = value;
    }
}

impl<T> DeviceAPI<T> for DeviceOpenBLAS where T: Clone {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_conversion() {
        let device_serial = DeviceCpuSerial {};
        let device_faer = DeviceOpenBLAS::new(0);
        let a = linspace((1.0, 5.0, 5, &device_serial));
        let b = a.into_device(&device_faer).unwrap();
        println!("{:?}", b);
        let a = linspace((1.0, 5.0, 5, &device_serial));
        let b = a.view().into_device(&device_faer).unwrap();
        println!("{:?}", b);
    }
}
