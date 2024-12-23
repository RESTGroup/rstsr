use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};

pub trait DeviceCreationAnyAPI<T>
where
    Self: DeviceRawVecAPI<T>,
{
    /// # Safety
    ///
    /// This function is unsafe because it does not initialize the memory.
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<T, Self>>;
    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<T, Self>>;
    fn outof_cpu_vec(&self, vec: Vec<T>) -> Result<Storage<T, Self>>;
    #[allow(clippy::wrong_self_convention)]
    fn from_cpu_vec(&self, vec: &[T]) -> Result<Storage<T, Self>>;
}

pub trait DeviceCreationNumAPI<T>
where
    T: Num,
    Self: DeviceRawVecAPI<T>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<T, Self>>;
    fn ones_impl(&self, len: usize) -> Result<Storage<T, Self>>;
    fn arange_int_impl(&self, len: usize) -> Result<Storage<T, Self>>;
}

pub trait DeviceCreationComplexFloatAPI<T>
where
    T: ComplexFloat,
    Self: DeviceRawVecAPI<T>,
{
    fn linspace_impl(&self, start: T, end: T, n: usize, endpoint: bool)
        -> Result<Storage<T, Self>>;
}

pub trait DeviceCreationPartialOrdNumAPI<T>
where
    T: Num + PartialOrd,
    Self: DeviceRawVecAPI<T>,
{
    fn arange_impl(&self, start: T, end: T, step: T) -> Result<Storage<T, Self>>;
}
