use crate::prelude_dev::*;

#[derive(Clone, Debug)]
pub struct DeviceFaer {
    base: DeviceCpuRayon,
}

impl DeviceFaer {
    pub fn new(num_threads: usize) -> Self {
        DeviceFaer { base: DeviceCpuRayon::new(num_threads) }
    }

    pub fn var_num_threads(&self) -> usize {
        self.base.var_num_threads()
    }

    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.base.set_num_threads(num_threads);
    }

    pub fn get_num_threads(&self) -> usize {
        self.base.get_num_threads()
    }

    pub fn get_pool(&self, n: usize) -> Result<rayon::ThreadPool> {
        self.base.get_pool(n)
    }
}

impl Default for DeviceFaer {
    fn default() -> Self {
        DeviceFaer::new(0)
    }
}

impl DeviceBaseAPI for DeviceFaer {
    fn same_device(&self, other: &Self) -> bool {
        self.var_num_threads() == other.var_num_threads()
    }
}

impl<T> DeviceRawVecAPI<T> for DeviceFaer
where
    T: Clone,
{
    type RawVec = Vec<T>;
}

impl<T> DeviceStorageAPI<T> for DeviceFaer
where
    T: Clone,
{
    fn new(vector: Self::RawVec, device: Self) -> Storage<T, Self> {
        Storage::<T, Self> { rawvec: vector, device }
    }

    fn len(storage: &Storage<T, Self>) -> usize {
        storage.rawvec.len()
    }

    fn to_cpu_vec(storage: &Storage<T, Self>) -> Result<Vec<T>> {
        Ok(storage.rawvec.clone())
    }

    fn into_cpu_vec(storage: Storage<T, Self>) -> Result<Vec<T>> {
        Ok(storage.rawvec)
    }

    #[inline]
    fn get_index(storage: &Storage<T, Self>, index: usize) -> T {
        storage.rawvec[index].clone()
    }

    #[inline]
    fn get_index_ptr(storage: &Storage<T, Self>, index: usize) -> *const T {
        &storage.rawvec[index] as *const T
    }

    #[inline]
    fn get_index_mut_ptr(storage: &mut Storage<T, Self>, index: usize) -> *mut T {
        &mut storage.rawvec[index] as *mut T
    }

    #[inline]
    fn set_index(storage: &mut Storage<T, Self>, index: usize, value: T) {
        storage.rawvec[index] = value;
    }
}

impl<T> DeviceAPI<T> for DeviceFaer where T: Clone {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_conversion() {
        let device_serial = DeviceCpuSerial {};
        let device_faer = DeviceFaer::new(0);
        let a = linspace((1.0, 5.0, 5, &device_serial));
        let b = a.into_device(&device_faer).unwrap();
        println!("{:?}", b);
        let a = linspace((1.0, 5.0, 5, &device_serial));
        let b = a.view().into_device(&device_faer).unwrap();
        println!("{:?}", b);
    }
}
