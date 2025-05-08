//! This module contains some features for rayon (parallel).
//!
//! - Layout parallel iterator
//! - Tensor parallel iterator

pub mod device;
pub mod par_iter;

pub use device::*;
