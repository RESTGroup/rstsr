//! Backend for CPU, serial only.

pub mod adv_indexing;
pub mod assignment;
pub mod creation;
pub mod device;
pub mod matmul;
pub mod operators;
pub mod reduction;

pub use device::*;
pub use operators::*;
