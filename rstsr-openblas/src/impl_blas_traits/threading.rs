use crate::prelude_dev::*;
use rstsr_blas_traits::prelude_dev::*;

impl BlasThreadAPI for DeviceBLAS {
    fn get_blas_num_threads(&self) -> usize {
        rstsr_openblas_ffi::get_num_threads()
    }

    fn set_blas_num_threads(&self, nthreads: usize) {
        rstsr_openblas_ffi::set_num_threads(nthreads);
    }

    fn with_blas_num_threads<T>(&self, nthreads: usize, f: impl FnOnce() -> T) -> T {
        rstsr_openblas_ffi::with_num_threads(nthreads, f)
    }
}
