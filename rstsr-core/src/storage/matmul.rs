//! Matrix, vector multiplication and related operations.

#![allow(clippy::too_many_arguments)]

use core::ops::{Add, Mul};

use crate::prelude_dev::*;

pub trait DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn matmul(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<DB>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceGEMMAPI<TA, TB, TC>
where
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn gemm(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<Ix2>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<Ix2>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<Ix2>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceSYMMAPI<TA, TB, TC>
where
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn symm(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<Ix2>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<Ix2>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<Ix2>,
        side: FlagSide,
        uplo: FlagUpLo,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceSYRKAPI<TA, TC>
where
    TA: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TC>,
{
    fn syrk(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<Ix2>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<Ix2>,
        uplo: FlagUpLo,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceHERKAPI<TA, TC>
where
    TA: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TC>,
{
    fn herk(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<Ix2>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<Ix2>,
        uplo: FlagUpLo,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceGEMVAPI<TA, TB, TC>
where
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn gemv(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<Ix1>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<Ix2>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<Ix1>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;

    fn gevm(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<Ix1>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<Ix1>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<Ix2>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceInnerDotAPI<TA, TB, TC>
where
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn inner_dot(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<Ix0>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<Ix1>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<Ix1>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}
