// per-device alias defined by the test entry
pub use crate::DeviceBLAS;
pub use rstsr::prelude::*;
pub use rstsr_core::prelude_dev::fingerprint;
pub use rstsr_test_manifest::get_vec;

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

macro_rules! c64 {
    ($real:expr, $imag:expr) => {
        c64::new($real, $imag)
    };
    ($real:expr) => {
        c64::new($real, 0.0)
    };
}

mod cholesky;
mod det;
mod eig;
mod eigh;
mod eigvals;
mod eigvalsh;
mod generalized_eigh;
mod inv;
mod lstsq;
mod pinv;
mod qr;
mod slogdet;
mod solve_general;
mod solve_symmetric;
mod solve_triangular;
mod svd;
mod svdvals;
