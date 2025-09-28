//! KML threading

use crate::prelude_dev::*;
use rstsr_blas_traits::prelude_dev::*;

/* #region threading number control */

struct KMLConfig;

impl KMLConfig {
    fn set_num_threads(&mut self, n: usize) {
        if rayon::current_thread_index().is_some() {
            // Inside rayon parallel region
            // For RSTSR usage, here `n` should always be 1 before actual computation.
            // if n != 1 { eprintln!("Warning: Setting KML threads to {n} inside a rayon parallel region instead
            // of 1. This may lead to thread lock in KML."); }
            unsafe { rstsr_kml_ffi::kblas::BlasSetNumThreadsLocal(n as _) };
            unsafe { rstsr_kml_ffi::service::KmlSetNumThreads(n as _) };
        } else {
            // Outside rayon parallel region
            unsafe { rstsr_kml_ffi::kblas::BlasSetNumThreads(n as _) };
            unsafe { rstsr_kml_ffi::service::KmlSetNumThreads(n as _) };
        }
    }

    fn get_num_threads(&mut self) -> usize {
        if rayon::current_thread_index().is_some() {
            // Inside rayon parallel region
            unsafe { rstsr_kml_ffi::kblas::BlasGetNumThreadsLocal() as usize }
        } else {
            // Outside rayon parallel region
            unsafe { rstsr_kml_ffi::kblas::BlasGetNumThreads() as usize }
        }
    }
}

/// Set number of threads for KML.
///
/// This function should be safe to call from multiple threads.
pub fn set_num_threads(n: usize) {
    KMLConfig.set_num_threads(n);
}

/// Get the number of threads currently set for KML.
///
/// This function should be safe to call from multiple threads.
pub fn get_num_threads() -> usize {
    KMLConfig.get_num_threads()
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
