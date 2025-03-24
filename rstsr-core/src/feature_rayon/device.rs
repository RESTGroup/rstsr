use crate::prelude_dev::*;

extern crate alloc;
use alloc::sync::Arc;

pub trait DeviceRayonAPI {
    fn set_num_threads(&mut self, num_threads: usize);
    fn get_num_threads(&self) -> usize;
    fn get_pool(&self) -> &ThreadPool;
    fn get_serial_pool(&self) -> &ThreadPool;
}

/// This is base device for Parallel CPU device.
///
/// This device is not intended to be used directly, but to be used as a base.
/// Possible inherited devices could be Faer or Blas.
///
/// This device is intended not to implement `DeviceAPI<T>`.
#[derive(Clone, Debug)]
pub struct DeviceCpuRayon {
    num_threads: usize,
    pool: Arc<ThreadPool>,
    serial_pool: Arc<ThreadPool>,
}

impl DeviceCpuRayon {
    pub fn new(num_threads: usize) -> Self {
        let pool = Arc::new(Self::generate_pool(num_threads).unwrap());
        let serial_serial = Arc::new(Self::generate_pool(1).unwrap());
        DeviceCpuRayon { num_threads, pool, serial_pool: serial_serial }
    }

    pub fn var_num_threads(&self) -> usize {
        self.num_threads
    }

    fn generate_pool(n: usize) -> Result<ThreadPool> {
        rayon::ThreadPoolBuilder::new().num_threads(n).build().map_err(Error::from)
    }
}

impl Default for DeviceCpuRayon {
    fn default() -> Self {
        DeviceCpuRayon::new(0)
    }
}

impl DeviceBaseAPI for DeviceCpuRayon {
    fn same_device(&self, other: &Self) -> bool {
        self.num_threads == other.num_threads
    }
}

impl DeviceRayonAPI for DeviceCpuRayon {
    fn set_num_threads(&mut self, num_threads: usize) {
        let num_threads_old = self.num_threads;
        if num_threads_old != num_threads {
            let pool = Self::generate_pool(num_threads).unwrap();
            self.num_threads = num_threads;
            self.pool = Arc::new(pool);
        }
    }

    fn get_num_threads(&self) -> usize {
        // if in rayon parallel worker, only one thread is used; otherwise use all
        // threads
        if rayon::current_thread_index().is_some() {
            1
        } else {
            match self.num_threads {
                0 => rayon::current_num_threads(),
                _ => rayon::current_num_threads().min(self.num_threads),
            }
        }
    }

    fn get_pool(&self) -> &ThreadPool {
        if self.get_num_threads() == 1 || rayon::current_thread_index().is_some() {
            self.serial_pool.as_ref()
        } else {
            self.pool.as_ref()
        }
    }

    fn get_serial_pool(&self) -> &ThreadPool {
        self.serial_pool.as_ref()
    }
}
