#![allow(clippy::needless_return)]
#![allow(non_camel_case_types)]
#![doc = include_str!("../readme.md")]

pub mod conversion;
pub mod creation;
pub mod device;
pub mod macro_impl;
pub mod matmul;
pub mod matmul_impl;
pub mod prelude_dev;

pub mod impl_blas_traits;
#[cfg(feature = "linalg")]
pub mod impl_linalg_traits;

use rstsr_core::prelude_dev::DeviceCpuRayon;

#[derive(Clone, Debug)]
pub struct DeviceOpenBLAS {
    base: DeviceCpuRayon,
}

pub(crate) use DeviceOpenBLAS as DeviceBLAS;
