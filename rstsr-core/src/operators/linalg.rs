use crate::prelude_dev::*;

pub trait DeviceVecdotAPI<TA, TB, TC, DA, DB, DC>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    fn vecdot(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<DB>,
        axes_a: &[isize],
        axes_b: &[isize],
    ) -> Result<()>;
}

pub trait DeviceTensordotAPI<TA, TB, TC, DA, DB, DC>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    /// Generalized tensor contraction: contract `axes_a` of `a` with `axes_b`
    /// of `b`, writing the non-contracted axes (a then b, original order) into
    /// the `MaybeUninit` output `c`. No conjugation. `axes_a`/`axes_b` are
    /// already normalized, non-negative and pairwise aligned.
    fn tensordot(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<DB>,
        axes_a: &[isize],
        axes_b: &[isize],
    ) -> Result<()>;
}
