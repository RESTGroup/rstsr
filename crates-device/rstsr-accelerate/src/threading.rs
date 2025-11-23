//! Apple Accelerate threading, now only single-threading supported.

use crate::prelude_dev::*;
use rstsr_blas_traits::prelude_dev::*;

/* #region threading number control */

struct AccelerateConfig;

impl AccelerateConfig {
    #[allow(dead_code)]
    fn set_num_threads(&mut self, _n: usize) {
        // Direct multi-thread control only available after macOS 15
    }

    fn get_num_threads(&mut self) -> usize {
        // Direct multi-thread control only available after macOS 15
        return 1;
    }
}

/// Set number of threads for Apple Accelerate.
///
/// This function should be safe to call from multiple threads.
pub fn set_num_threads(_n: usize) {
    // Direct multi-thread control only available after macOS 15
}

/// Get the number of threads currently set for Apple Accelerate.
///
/// This function should be safe to call from multiple threads.
pub fn get_num_threads() -> usize {
    AccelerateConfig.get_num_threads()
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
