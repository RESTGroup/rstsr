//! This module contains some features for rayon (parallel).
//!
//! - Layout parallel iterator
//! - Tensor parallel iterator

pub mod assignment;
pub mod device;
pub mod matmul_naive;
pub mod operators;
pub mod par_iter;
pub mod reduction;

pub use assignment::*;
pub use device::*;
pub use operators::*;
pub use par_iter::*;
pub use reduction::*;
