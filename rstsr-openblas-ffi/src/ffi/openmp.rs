use std::ffi::c_int;

extern "C" {
    pub fn omp_set_num_threads(arg1: c_int);
    pub fn omp_get_num_threads() -> c_int;
}
