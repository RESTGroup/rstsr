#![cfg_attr(not(test), no_std)]
#![doc = include_str!("docs/lib.md")]
// This option is for myself as python-like developer.
#![allow(clippy::needless_return)]
// Resolution to something like `tensor.slice_mut() += scalar`.
// This code is not allowed in rust, but `*&mut tensor.slice_mut() += scalar` is allowed.
#![allow(clippy::deref_addrof)]

pub mod prelude;
pub mod prelude_dev;

pub mod error;
pub mod flags;

pub mod layout;
pub use layout::{DimAPI, Layout};

pub mod storage;

pub mod tensor;
pub mod tensorbase;
pub use tensorbase::{
    Tensor, TensorArc, TensorBase, TensorCow, TensorMut, TensorRef, TensorView, TensorViewMut,
};

pub mod format;

#[cfg(feature = "rayon")]
pub mod feature_rayon;

#[cfg(feature = "faer")]
pub mod device_faer;

pub mod device_cpu_serial;

mod dev_utilities;

#[cfg(feature = "faer_as_default")]
pub type DeviceCpu = device_faer::device::DeviceFaer;

#[cfg(not(feature = "faer_as_default"))]
pub type DeviceCpu = device_cpu_serial::DeviceCpuSerial;

pub mod doc_api_specification {
    #![doc = include_str!("docs/api_specification.md")]

    #[allow(unused_imports)]
    use crate::prelude_dev::*;
}
