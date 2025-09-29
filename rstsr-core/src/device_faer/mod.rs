//! Backend for CPU, some using rayon for parallel, but matmul and linalg implemented by faer.

pub mod conversion;
pub mod creation;
pub mod device;
pub mod matmul;
pub mod matmul_impl;
pub mod rayon_auto_impl;
