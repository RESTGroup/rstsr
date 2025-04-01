use crate::prelude_dev::*;

extern crate alloc;
use alloc::sync::Arc;

pub trait DeviceRayonAPI {
    /// Set the number of threads for the device.
    fn set_num_threads(&mut self, num_threads: usize);

    /// Get the number of threads for the device.
    ///
    /// This function should give the number of threads for the pool. It is not
    /// related to whether the current work is done in parallel or serial.
    fn get_num_threads(&self) -> usize;

    /// Get the thread pool for the device.
    ///
    /// **Note**:
    ///
    /// For developers, this function should not be used directly. Instead, use
    /// `get_current_pool` to detect whether using thread pool of its own (Some)
    /// or using parent thread pool (None).
    fn get_pool(&self) -> &ThreadPool;

    /// Get the current thread pool for the device.
    ///
    /// - If in parallel worker, this returns None. This means the program
    ///   should use the thread pool from the parent. It is important that this
    ///   does not necessarily means this work should be done in serial.
    /// - If not in rayon parallel worker, this returns the thread pool.
    fn get_current_pool(&self) -> Option<&ThreadPool>;
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
    default_order: FlagOrder,
}

impl DeviceCpuRayon {
    pub fn new(num_threads: usize) -> Self {
        let pool = Arc::new(Self::generate_pool(num_threads).unwrap());
        DeviceCpuRayon { num_threads, pool, default_order: FlagOrder::default() }
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
        self.default_order == other.default_order
    }

    fn default_order(&self) -> FlagOrder {
        self.default_order
    }

    fn set_default_order(&mut self, order: FlagOrder) {
        self.default_order = order;
    }
}

impl DeviceRayonAPI for DeviceCpuRayon {
    #[inline]
    fn set_num_threads(&mut self, num_threads: usize) {
        let num_threads_old = self.num_threads;
        if num_threads_old != num_threads {
            let pool = Self::generate_pool(num_threads).unwrap();
            self.num_threads = num_threads;
            self.pool = Arc::new(pool);
        }
    }

    #[inline]
    fn get_num_threads(&self) -> usize {
        match self.num_threads {
            0 => self.pool.current_num_threads(),
            _ => self.num_threads,
        }
    }

    #[inline]
    fn get_pool(&self) -> &ThreadPool {
        self.pool.as_ref()
    }

    #[inline]
    fn get_current_pool(&self) -> Option<&ThreadPool> {
        match rayon::current_thread_index() {
            Some(_) => None,
            None => Some(self.pool.as_ref()),
        }
    }
}
