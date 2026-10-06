//! Quaternary elementwise operations (see
//! `crate::operators::ops::op_quaternary_common`).

use crate::prelude_dev::*;

// Special case for where (select)

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
            let (b, c) = TX::promote_pair(b.clone(), c.clone());
            d.write(if *a { b } else { c });
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
        let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &bool, b: &TX| {
            let (b, c) = TX::promote_pair(b.clone(), c.clone());
            d.write(if *a { b } else { c });
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
        let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &bool, c: &TY| {
            let (b, c) = TX::promote_pair(b.clone(), c.clone());
            d.write(if *a { b } else { c });
        };
        self.op_mutc_refa_refb_func(d, ld, a, la, c, lc, &mut func)
    }
}
