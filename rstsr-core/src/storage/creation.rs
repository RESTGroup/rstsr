use crate::prelude_dev::*;

pub trait DeviceCreationAnyAPI<T>: DeviceRawAPI<T> {
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

    #[allow(clippy::type_complexity)]
    fn uninit_impl(
        &self,
        len: usize,
    ) -> Result<Storage<DataOwned<<Self as DeviceRawAPI<MaybeUninit<T>>>::Raw>, MaybeUninit<T>, Self>>
    where
        Self: DeviceRawAPI<MaybeUninit<T>>;

    /// # Safety
    ///
    /// This function is unsafe because it assumes that the input storage is fully initialized.
    #[allow(clippy::type_complexity)]
    unsafe fn assume_init_impl(
        storage: Storage<DataOwned<<Self as DeviceRawAPI<MaybeUninit<T>>>::Raw>, MaybeUninit<T>, Self>,
    ) -> Result<Storage<DataOwned<<Self as DeviceRawAPI<T>>::Raw>, T, Self>>
    where
        Self: DeviceRawAPI<MaybeUninit<T>>;
}

pub trait DeviceCreationNumAPI<T>: DeviceRawAPI<T> {
    fn zeros_impl(&self, len: usize) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
    fn ones_impl(&self, len: usize) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
    fn arange_int_impl(&self, len: usize) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
}

pub trait DeviceCreationComplexFloatAPI<T>: DeviceRawAPI<T> {
    fn linspace_impl(
        &self,
        start: T,
        end: T,
        n: usize,
        endpoint: bool,
    ) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
}

pub trait DeviceCreationPartialOrdNumAPI<T>: DeviceRawAPI<T> {
    fn arange_impl(&self, start: T, end: T, step: T) -> Result<Storage<DataOwned<Self::Raw>, T, Self>>;
}

pub trait DeviceCreationTriAPI<T>: DeviceRawAPI<T> {
    fn tril_impl<D>(&self, raw: &mut Self::Raw, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI;
    fn triu_impl<D>(&self, raw: &mut Self::Raw, layout: &Layout<D>, k: isize) -> Result<()>
    where
        D: DimAPI;
}
