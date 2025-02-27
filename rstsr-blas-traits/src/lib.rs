#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]

#[cfg(not(feature = "ilp64"))]
pub type blasint = i32;
#[cfg(feature = "ilp64")]
pub type blasint = i64;

pub mod blas_scalar;
pub mod cblas_flags;
pub mod prelude_dev;
pub mod util;

pub mod blas3;
