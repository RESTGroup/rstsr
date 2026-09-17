#![allow(unused_imports)]

// `rstsr_sci_traits` self-alias lives in lib.rs (`extern crate self as ...`).
pub(crate) use rstsr_core::prelude_dev::*;

#[cfg(feature = "faer")]
pub(crate) type DeviceRayonAutoImpl = rstsr_core::prelude_dev::DeviceFaer;
