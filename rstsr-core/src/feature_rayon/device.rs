use crate::prelude_dev::*;

pub trait DeviceRayonAPI {
    fn set_num_threads(&mut self, num_threads: usize);
    fn get_num_threads(&self) -> usize;
    fn get_pool(&self, n: usize) -> Result<rayon::ThreadPool>;
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
}

impl DeviceCpuRayon {
    pub fn new(num_threads: usize) -> Self {
        DeviceCpuRayon { num_threads }
    }

    pub fn var_num_threads(&self) -> usize {
        self.num_threads
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
        self.num_threads = num_threads;
    }

    fn get_num_threads(&self) -> usize {
        match self.num_threads {
            0 => rayon::current_num_threads(),
            _ => rayon::current_num_threads().min(self.num_threads),
        }
    }

    fn get_pool(&self, n: usize) -> Result<rayon::ThreadPool> {
        rstsr_pattern!(n, 0..=self.get_num_threads(), RayonError, "Specified too much threads.")?;
        let nthreads = if n == 0 { self.get_num_threads() } else { n };
        rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().map_err(Error::from)
    }
}
