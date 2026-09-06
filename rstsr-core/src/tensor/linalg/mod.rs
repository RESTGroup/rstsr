//! Basic linear algebra operations: matrix multiplication ([`matmul`]),
//! vector dot product ([`vecdot`]), and matrix transpose
//! ([`matrix_transpose`]).
//!
//! This module covers the array-API-level linalg functions of rstsr-core;
//! BLAS-level interfaces live in the separate `rstsr-linalg-traits` crate.

pub mod matmul;
pub mod matrix_transpose;
pub mod vecdot;

pub mod exports {
    use super::*;

    pub use matmul::*;
    pub use matrix_transpose::*;
    pub use vecdot::*;
}
