pub mod cblas;
pub mod lapack;

use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::prelude::*;

impl<T> BlasDriverBaseAPI<T> for DeviceBLAS
where
    T: BlasFloat,
    T::Real: BlasFloat + rstsr_dtype_traits::MinMaxAPI + num::Bounded,
{
}

#[duplicate_item(T; [f32]; [f64]; [Complex<f32>]; [Complex<f64>])]
impl BlasDriverAPI<T> for DeviceBLAS {}

#[duplicate_item(T; [f32]; [f64]; [Complex<f32>]; [Complex<f64>])]
impl LapackDriverAPI<T> for DeviceBLAS {}
