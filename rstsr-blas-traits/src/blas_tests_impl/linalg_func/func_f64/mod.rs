// per-device alias defined by the test entry
pub use crate::DeviceBLAS;
pub use rstsr::prelude::*;
pub use rstsr_core::prelude_dev::fingerprint;
pub use rstsr_test_manifest::get_vec;

mod cholesky;
mod det;
mod eigh;
mod eigvalsh;
mod inv;
mod pinv;
mod qr;
mod slogdet;
mod solve_general;
mod solve_symmetric;
mod solve_triangular;
mod svd;
mod svdvals;
