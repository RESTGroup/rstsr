//! Data assignments on device

use crate::prelude_dev::*;

pub trait OpAssignArbitaryAPI<T, DC, DA>
where
    DC: DimAPI,
    DA: DimAPI,
    Self: DeviceAPI<T>,
{
    /// Element-wise assignment in col-major order, without no promise that
    /// input layouts are broadcastable.
    fn assign_arbitary(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<DC>,
        a: &Storage<T, Self>,
        la: &Layout<DA>,
    ) -> Result<()>;
}

pub trait OpAssignAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    /// Element-wise assignment for same layout arrays.
    fn assign(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<D>,
        a: &Storage<T, Self>,
        la: &Layout<D>,
    ) -> Result<()>;

    fn fill(&self, c: &mut Storage<T, Self>, lc: &Layout<D>, fill: T) -> Result<()>;
}
