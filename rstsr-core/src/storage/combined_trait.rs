use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};

// This trait can be used for simple implementation of f32, f64, Complex<f32>,
// Complex<f64> types operations.
pub trait DeviceComplexFloatAPI<T, D = IxD>:
    DeviceAPI<T>
    // trait bound for complex part
    + DeviceAPI<T::Real>
    + DeviceCreationNumAPI<T>
    + DeviceCreationAnyAPI<T>
    + DeviceCreationComplexFloatAPI<T>
    + OpAssignArbitaryAPI<T, D, D>
    + OpAssignArbitaryAPI<T, D, IxD>
    + OpAssignArbitaryAPI<T, IxD, D>
    + OpAssignArbitaryAPI<T, IxD, IxD>
    + OpAssignAPI<T, D>
    + OpAssignAPI<T, IxD>
    + DeviceConjAPI<T, D, TOut = T>
    + DeviceConjAPI<T, IxD, TOut = T>
    // trait bound for real part
    + DeviceCreationNumAPI<T::Real>
    + DeviceCreationAnyAPI<T::Real>
    + DeviceCreationComplexFloatAPI<T::Real>
    + OpAssignArbitaryAPI<T::Real, D, D>
    + OpAssignArbitaryAPI<T::Real, D, IxD>
    + OpAssignArbitaryAPI<T::Real, IxD, D>
    + OpAssignArbitaryAPI<T::Real, IxD, IxD>
    + OpAssignAPI<T::Real, D>
    + OpAssignAPI<T::Real, IxD>
    + DeviceConjAPI<T::Real, D, TOut = T::Real>
    + DeviceConjAPI<T::Real, IxD, TOut = T::Real>
where
    T: ComplexFloat,
    D: DimAPI,
{
}

pub trait DeviceNumAPI<T, D = IxD>:
    DeviceAPI<T>
    + DeviceCreationNumAPI<T>
    + DeviceCreationAnyAPI<T>
    + OpAssignArbitaryAPI<T, D, D>
    + OpAssignArbitaryAPI<T, D, IxD>
    + OpAssignArbitaryAPI<T, IxD, D>
    + OpAssignArbitaryAPI<T, IxD, IxD>
    + OpAssignAPI<T, D>
    + OpAssignAPI<T, IxD>
where
    T: Num,
    D: DimAPI,
{
}
