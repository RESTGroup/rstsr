//! openblas threading
//!
//! # Todo
//!
//! Current implementation only handles pthreads parallelism.
//! For OpenMP, some other ways may be required.

use std::sync::Mutex;

struct OpenBLASConfig;

impl OpenBLASConfig {
    fn set_num_threads(&self, n: usize) {
        unsafe {
            crate::ffi::cblas::openblas_set_num_threads(n as i32);
        }
    }

    fn get_num_threads(&self) -> usize {
        unsafe { crate::ffi::cblas::openblas_get_num_threads() as usize }
    }
}

static INTERNAL: Mutex<OpenBLASConfig> = Mutex::new(OpenBLASConfig);

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
