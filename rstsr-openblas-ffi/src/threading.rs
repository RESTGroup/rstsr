//! openblas threading
//!
//! # Todo
//!
//! Current implementation only handles pthreads parallelism.
//! For OpenMP, some other ways may be required.

use std::sync::Mutex;

static SET_THREAD: Mutex<()> = Mutex::new(());

/// Set number of threads for OpenBLAS.
///
/// This function should be safe to call from multiple threads.
pub fn set_num_threads(n: usize) {
    let _lock = SET_THREAD.lock().unwrap();
    unsafe {
        crate::ffi::cblas::openblas_set_num_threads(n as i32);
    }
}

pub fn get_num_threads() -> usize {
    unsafe { crate::ffi::cblas::openblas_get_num_threads() as usize }
}
