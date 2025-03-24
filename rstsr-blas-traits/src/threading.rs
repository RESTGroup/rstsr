//! BLAS threading control.
//!
//! Most BLAS distributions have a way to control the number of threads used by
//! the library. This module provides trait to set the number of threads used by
//! the BLAS library.

pub trait BlasThreadAPI {
    /// Set the number of threads used by the BLAS library.
    fn set_blas_num_threads(&self, nthreads: usize);
    /// Get the number of threads used by the BLAS library.
    fn get_blas_num_threads(&self) -> usize;
    /// Set the number of threads in closure.
    fn with_blas_num_threads<T>(&self, nthreads: usize, f: impl FnOnce() -> T) -> T {
        let n = self.get_blas_num_threads();
        self.set_blas_num_threads(nthreads);
        let result = f();
        self.set_blas_num_threads(n);
        result
    }
}
