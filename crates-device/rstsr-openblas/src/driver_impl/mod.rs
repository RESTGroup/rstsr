pub mod cblas;

#[cfg(not(feature = "lapacke"))]
pub mod lapack;
#[cfg(feature = "lapacke")]
pub mod lapacke;

use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;

impl<T> BlasDriverBaseAPI<T> for DeviceBLAS where T: BlasFloat<Real: BlasFloat> {}

#[duplicate_item(T; [f32]; [f64]; [Complex<f32>]; [Complex<f64>])]
impl BlasDriverAPI<T> for DeviceBLAS {}

#[duplicate_item(T; [f32]; [f64]; [Complex<f32>]; [Complex<f64>])]
impl LapackDriverAPI<T> for DeviceBLAS {}
