#![allow(refining_impl_trait)]
#![allow(clippy::needless_return)]
#![cfg_attr(not(test), no_std)]
#![doc = include_str!("readme.md")]

pub mod prelude_dev;

pub mod flags;
pub mod error;

pub mod layout;
pub use layout::{DimAPI, Layout};

pub mod storage;

pub mod tensor;
pub mod tensorbase;
pub use tensorbase::{Tensor, TensorBase};

pub mod format;

pub mod cpu_backend;

mod dev_utilities;

#[cfg(feature = "cuda")]
pub mod cuda_backend;
