//! Backend for CPU, serial only.

pub mod assignment;
pub mod creation;
pub mod device;
pub mod matmul;
pub mod operators;
pub mod reduction;

pub use assignment::*;
pub use device::*;
pub use operators::*;
pub use reduction::*;
