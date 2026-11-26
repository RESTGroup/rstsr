// per-device alias defined by the test entry
pub use crate::DeviceBLAS;
pub use rstsr::prelude::*;
pub use rstsr_core::prelude_dev::fingerprint;
pub use rstsr_test_manifest::get_vec;

pub mod geev;
mod gelsd;
mod geqrf;
mod gesdd;
mod gesv;
mod gesvd;
mod getrf;
mod getrf_getri;
mod ggev;
pub mod orgqr;
mod potrf;
mod syev;
mod syevd;
mod sygv;
mod sygvd;
mod sysv;
mod sytrf;

// layout tests
mod svd_layout;
