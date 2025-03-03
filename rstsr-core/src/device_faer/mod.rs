//! Backend for CPU, some using rayon for parallel, but matmul should be
//! implemented elsewhere.

pub mod conversion;
pub mod creation;
pub mod device;
pub mod matmul;
pub mod matmul_impl;
pub mod rayon_auto_impl;
