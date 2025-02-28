use crate::prelude_dev::*;
use num::complex::ComplexFloat;

// This trait can be used for simple implementation of f32, f64, Complex<f32>,
// Complex<f64> types operations.
pub trait DeviceComplexFloatAPI<T, D = IxD>:
    DeviceAPI<T>
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
where
    T: ComplexFloat,
    D: DimAPI,
{
}
