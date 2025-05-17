#![allow(clippy::type_complexity)]
#![allow(non_camel_case_types)]

pub mod prelude;
pub mod prelude_dev;
pub mod ref_impl_blas;
pub mod traits_def;

#[cfg(feature = "faer")]
pub mod faer_impl;
