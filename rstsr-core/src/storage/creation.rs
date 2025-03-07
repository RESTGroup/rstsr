use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Num};

pub trait DeviceCreationAnyAPI<T>
where
    Self: DeviceRawAPI<T>,
{
    /// # Safety
    ///
    /// This function is unsafe because it does not initialize the memory.
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>
    where
        T: Clone;
    fn outof_cpu_vec(&self, vec: Vec<T>) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
    #[allow(clippy::wrong_self_convention)]
    fn from_cpu_vec(&self, vec: &[T]) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>
    where
        T: Clone;
}

pub trait DeviceCreationNumAPI<T>
where
    T: Num,
    Self: DeviceRawAPI<T>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
    fn ones_impl(&self, len: usize) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
    fn arange_int_impl(&self, len: usize) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
}

pub trait DeviceCreationComplexFloatAPI<T>
where
    T: ComplexFloat,
    Self: DeviceRawAPI<T>,
{
    fn linspace_impl(
        &self,
        start: T,
        end: T,
        n: usize,
        endpoint: bool,
    ) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
}

pub trait DeviceCreationPartialOrdNumAPI<T>
where
    T: Num + PartialOrd,
    Self: DeviceRawAPI<T>,
{
    fn arange_impl(
        &self,
        start: T,
        end: T,
        step: T,
    ) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
}

pub trait DeviceCreationTriAPI<T>
where
    T: Num,
    Self: DeviceRawAPI<T>,
{
    fn tril_impl<D>(&self, raw: &mut Self::Raw, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI;
    fn triu_impl<D>(&self, raw: &mut Self::Raw, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI;
}
