#![allow(clippy::needless_return)]
#![allow(non_camel_case_types)]
#![doc = include_str!("../readme.md")]

pub mod conversion;
pub mod creation;
pub mod device;
pub mod impl_blas_traits;
pub mod macro_impl;
pub mod matmul;
pub mod matmul_impl;
pub mod prelude_dev;

pub use device::DeviceOpenBLAS;
