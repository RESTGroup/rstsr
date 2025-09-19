//! Data assignments on device

use crate::prelude_dev::*;

pub trait OpAssignArbitaryAPI<T, DC, DA>
where
    DC: DimAPI,
    DA: DimAPI,
    Self: DeviceRawAPI<T> + DeviceRawAPI<MaybeUninit<T>>,
{
    /// Element-wise assignment in col-major order, without no promise that
    /// input layouts are broadcastable.
    fn assign_arbitary(
        &self,
        c: &mut <Self as DeviceRawAPI<T>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<DA>,
    ) -> Result<()>;

    fn assign_arbitary_uninit(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<DA>,
    ) -> Result<()>;
}

pub trait OpAssignAPI<T, D>
where
    D: DimAPI,
    Self: DeviceRawAPI<T> + DeviceRawAPI<MaybeUninit<T>>,
{
    /// Element-wise assignment for same layout arrays.
    fn assign(
        &self,
        c: &mut <Self as DeviceRawAPI<T>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
    ) -> Result<()>;

    fn assign_uninit(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
    ) -> Result<()>;

    fn fill(&self, c: &mut <Self as DeviceRawAPI<T>>::Raw, lc: &Layout<D>, fill: T) -> Result<()>;
}
