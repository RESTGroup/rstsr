//! Matrix, vector multiplication and related operations.

#![allow(clippy::too_many_arguments)]

use crate::prelude_dev::*;

pub trait DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
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

    /// Matrix multiplication into an **uninitialized** output: writes
    /// `c = alpha * (a @ b)` and never reads `c`.
    ///
    /// `c` is `MaybeUninit`-typed storage. Implementations must initialize
    /// every element of `lc` (via `MaybeUninit::write` or equivalent) and
    /// must not read or drop the previous contents — this is the fresh-output
    /// path of the allocating matmul; the `beta`-scaling path, which reads
    /// `c` (initialized by the caller), is [`DeviceMatMulAPI::matmul`].
    fn matmul_uninit(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<DB>,
        alpha: TC,
    ) -> Result<()>
    where
        Self: DeviceRawAPI<MaybeUninit<TC>>;
}

pub trait DeviceGEMMAPI<TA, TB, TC>
where
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
