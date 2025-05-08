//! openblas threading

use crate::prelude_dev::*;
#[cfg(feature = "openmp")]
use core::ffi::c_int;
use rstsr_blas_traits::prelude_dev::*;
use std::sync::Mutex;

use rstsr_openblas_ffi::cblas::{OPENBLAS_OPENMP, OPENBLAS_SEQUENTIAL, OPENBLAS_THREAD};

/* #region required openmp ffi */

#[cfg(feature = "openmp")]
extern "C" {
    pub fn omp_set_num_threads(arg1: c_int);
    pub fn omp_get_num_threads() -> c_int;
}

/* #endregion */

/* #region parallel scheme */

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenBLASParallel {
    Sequential,
    Thread,
    OpenMP,
}

pub fn get_parallel() -> OpenBLASParallel {
    unsafe {
        match rstsr_openblas_ffi::cblas::openblas_get_parallel().try_into().unwrap() {
            OPENBLAS_SEQUENTIAL => OpenBLASParallel::Sequential,
            OPENBLAS_THREAD => OpenBLASParallel::Thread,
            OPENBLAS_OPENMP => {
                #[cfg(not(feature = "openmp"))]
                panic!(concat!(
                    "OpenMP is not enabled in `rstsr-openblas-ffi`, but detected using shared library `libopenblas` compiled with OpenMP.\n",
                    "Please either\n",
                    "- enable feature `openmp` when building `rstsr-openblas-ffi` and rebuild this crate;\n",
                    "- run with libopenblas compiled with pthread (rebuild `rstsr-openblas-ffi` is not required in this case).",
                ));
                #[cfg(feature = "openmp")]
                OpenBLASParallel::OpenMP
            },
            _ => panic!("Unknown parallelism type"),
        }
    }
}

/* #endregion */

/* #region threading number control */

struct OpenBLASConfig {
    parallel: Option<u32>,
}

impl OpenBLASConfig {
    fn set_num_threads(&mut self, n: usize) {
        unsafe {
            match self.get_parallel() {
                OPENBLAS_THREAD => rstsr_openblas_ffi::cblas::openblas_set_num_threads(n as i32),
                #[cfg(feature = "openmp")]
                OPENBLAS_OPENMP => omp_set_num_threads(n as c_int),
                _ => (),
            }
        }
    }

    fn get_num_threads(&mut self) -> usize {
        unsafe {
            match self.get_parallel() {
                OPENBLAS_THREAD => rstsr_openblas_ffi::cblas::openblas_get_num_threads() as usize,
                #[cfg(feature = "openmp")]
                OPENBLAS_OPENMP => omp_get_num_threads() as usize,
                _ => 1,
            }
        }
    }

    fn get_parallel(&mut self) -> u32 {
        match self.parallel {
            Some(p) => p,
            None => {
                let p = unsafe { rstsr_openblas_ffi::cblas::openblas_get_parallel() } as u32;
                #[cfg(not(feature = "openmp"))]
                if p == OPENBLAS_OPENMP {
                    panic!(concat!(
                        "OpenMP is not enabled in `rstsr-openblas-ffi`, but detected using shared library `libopenblas` compiled with OpenMP.\n",
                        "Please either\n",
                        "- enable feature `openmp` when building `rstsr-openblas-ffi` and rebuild this crate;\n",
                        "- run with libopenblas compiled with pthread (rebuild `rstsr-openblas-ffi` is not required in this case).",
                    ));
                }
                self.parallel = Some(p);
                p
            },
        }
    }
}

static INTERNAL: Mutex<OpenBLASConfig> = Mutex::new(OpenBLASConfig { parallel: None });

/// Set number of threads for OpenBLAS.
///
/// This function should be safe to call from multiple threads.
pub fn set_num_threads(n: usize) {
    INTERNAL.lock().unwrap().set_num_threads(n);
}

pub fn get_num_threads() -> usize {
    INTERNAL.lock().unwrap().get_num_threads()
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
