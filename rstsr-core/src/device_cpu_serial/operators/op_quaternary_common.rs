//! Quaternary elementwise operations (see
//! `crate::operators::ops::op_quaternary_common`).

use crate::prelude_dev::*;

// Special case for where (select): promote only the selected branch, so the
// discarded operand is never cloned or promoted.

#[inline]
fn select_promote<TX, TY>(cond: bool, x: TX, y: TY) -> <TX as DTypePromoteAPI<TY>>::Res
where
    TX: DTypePromoteAPI<TY>,
{
    if cond {
        TX::promote_self(x)
    } else {
        <TX as DTypePromoteAPI<TY>>::promote_other(y)
    }
}

impl<TX, TY, D> OpWhereAPI<TX, TY, D> for DeviceCpuSerial
where
    TX: Clone + DTypePromoteAPI<TY, Res: Clone>,
    TY: Clone,
    D: DimAPI,
{
    type TOut = TX::Res;

    fn op_mutd_refa_refb_refc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<bool>,
        la: &Layout<D>,
        b: &Vec<TX>,
        lb: &Layout<D>,
        c: &Vec<TY>,
        lc: &Layout<D>,
    ) -> Result<()> {
        let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &bool, b: &TX, c: &TY| {
            d.write(select_promote(*a, b.clone(), c.clone()));
        };
        self.op_mutd_refa_refb_refc_func(d, ld, a, la, b, lb, c, lc, &mut func)
    }

    fn op_mutd_refa_refb_numc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<bool>,
        la: &Layout<D>,
        b: &Vec<TX>,
        lb: &Layout<D>,
        c: TY,
    ) -> Result<()> {
        let c = <TX as DTypePromoteAPI<TY>>::promote_other(c);
        let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &bool, b: &TX| {
            d.write(if *a { TX::promote_self(b.clone()) } else { c.clone() });
        };
        self.op_mutc_refa_refb_func(d, ld, a, la, b, lb, &mut func)
    }

    fn op_mutd_refa_numb_refc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<bool>,
        la: &Layout<D>,
        b: TX,
        c: &Vec<TY>,
        lc: &Layout<D>,
    ) -> Result<()> {
        let b = TX::promote_self(b);
        let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &bool, c: &TY| {
            d.write(if *a { b.clone() } else { <TX as DTypePromoteAPI<TY>>::promote_other(c.clone()) });
        };
        self.op_mutc_refa_refb_func(d, ld, a, la, c, lc, &mut func)
    }
}
