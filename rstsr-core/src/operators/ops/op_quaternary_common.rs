//! Quaternary elementwise operations: one output plus three tensor inputs.
//!
//! Currently only the select operation [`OpWhereAPI`] (NumPy `where` /
//! array API `where(condition, x1, x2)`); operand letters map as
//! (a, b, c) = (cond, x, y), output letter `d`.

use crate::prelude_dev::*;

// select operation: where(cond, x, y); operands map to (a, b, c) = (cond, x, y)

pub trait OpWhereAPI<TX, TY, D>
where
    D: DimAPI,
    Self: DeviceAPI<bool> + DeviceAPI<TX> + DeviceAPI<TY> + DeviceAPI<MaybeUninit<Self::TOut>>,
{
    type TOut;

    fn op_mutd_refa_refb_refc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<bool>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TX>>::Raw,
        lb: &Layout<D>,
        c: &<Self as DeviceRawAPI<TY>>::Raw,
        lc: &Layout<D>,
    ) -> Result<()>;

    fn op_mutd_refa_refb_numc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<bool>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TX>>::Raw,
        lb: &Layout<D>,
        c: TY,
    ) -> Result<()>;

    fn op_mutd_refa_numb_refc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<bool>>::Raw,
        la: &Layout<D>,
        b: TX,
        c: &<Self as DeviceRawAPI<TY>>::Raw,
        lc: &Layout<D>,
    ) -> Result<()>;
}
