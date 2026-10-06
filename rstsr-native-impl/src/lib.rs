#![cfg_attr(not(any(test, feature = "std")), no_std)]
#[cfg(feature = "rayon")]
pub mod cpu_rayon;
pub mod cpu_serial;
pub mod scalar_math;

pub mod prelude_dev;
