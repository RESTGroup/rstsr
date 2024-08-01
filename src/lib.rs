#![allow(refining_impl_trait)]

/* #region Configuration */

pub static C_PREFER: bool = cfg!(feature = "c_prefer");

/* #endregion */

pub mod error;
pub use error::{Error, Result};

pub mod layout;

pub mod storage;

pub mod tensor;
pub mod tensorbase;
pub use tensorbase::{Tensor, TensorBase};

pub mod cpu_backend;

#[cfg(feature = "cuda")]
pub mod cuda_backend;
