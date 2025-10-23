//! Data assignments on device

use crate::prelude_dev::*;

pub trait OpAssignArbitaryAPI<TC, DC, DA, TA = TC>
where
    DC: DimAPI,
    DA: DimAPI,
    Self: DeviceRawAPI<TA> + DeviceRawAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
{
    /// Element-wise assignment in col-major order, without no promise that
    /// input layouts are broadcastable.
    fn assign_arbitary(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
    ) -> Result<()>;

    fn assign_arbitary_uninit(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
    ) -> Result<()>;
}

pub trait OpAssignAPI<TC, D, TA = TC>
where
    D: DimAPI,
    Self: DeviceRawAPI<TA> + DeviceRawAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
{
    /// Element-wise assignment for same layout arrays.
    fn assign(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
    ) -> Result<()>;

    fn assign_uninit(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
    ) -> Result<()>;

    fn fill(&self, c: &mut <Self as DeviceRawAPI<TC>>::Raw, lc: &Layout<D>, fill: TA) -> Result<()>;
}
