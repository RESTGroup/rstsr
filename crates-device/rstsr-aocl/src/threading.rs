//! AOCL threading

use crate::prelude_dev::*;
use rstsr_blas_traits::prelude_dev::*;

/* #region threading number control */

struct AOCLConfig;

impl AOCLConfig {
    fn set_num_threads(&mut self, n: usize) {
        unsafe { rstsr_aocl_ffi::blis::bli_thread_set_num_threads(n as _) };
    }

    fn get_num_threads(&mut self) -> usize {
        unsafe { rstsr_aocl_ffi::blis::bli_thread_get_num_threads() as usize }
    }
}

/// Set number of threads for AOCL.
///
/// This function should be safe to call from multiple threads.
pub fn set_num_threads(n: usize) {
    AOCLConfig.set_num_threads(n);
}

/// Get the number of threads currently set for AOCL.
///
/// This function should be safe to call from multiple threads.
pub fn get_num_threads() -> usize {
    AOCLConfig.get_num_threads()
}

pub fn with_num_threads<F, R>(nthreads: usize, f: F) -> R
where
    F: FnOnce() -> R,
{
    let n = get_num_threads();
    set_num_threads(nthreads);
    let r = f();
    set_num_threads(n);
    return r;
}

/* #endregion */

/* #region trait impl */

impl BlasThreadAPI for DeviceBLAS {
    fn get_blas_num_threads(&self) -> usize {
        crate::threading::get_num_threads()
    }

    fn set_blas_num_threads(&self, nthreads: usize) {
        crate::threading::set_num_threads(nthreads);
    }

    fn with_blas_num_threads<T>(&self, nthreads: usize, f: impl FnOnce() -> T) -> T {
        crate::threading::with_num_threads(nthreads, f)
    }
}

/* #endregion */
