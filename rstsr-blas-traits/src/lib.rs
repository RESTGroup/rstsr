#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::type_complexity)]

#[cfg(not(feature = "ilp64"))]
pub type blas_int = i32;
#[cfg(feature = "ilp64")]
pub type blas_int = i64;

pub mod blas_scalar;
pub mod cblas_flags;
pub mod prelude;
pub mod prelude_dev;
pub mod threading;
pub mod util;

pub mod blas3;

pub mod lapack_eigh;
pub mod lapack_solve;
