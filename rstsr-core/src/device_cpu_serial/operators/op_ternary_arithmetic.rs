use crate::prelude_dev::*;

#[duplicate_item(
     OpAPI         Op       func                                  ;
    [OpAddAPI   ] [Add   ] [|c, a, b| { c.write(a.clone() +  b.clone()); }];
    [OpSubAPI   ] [Sub   ] [|c, a, b| { c.write(a.clone() -  b.clone()); }];
    [OpMulAPI   ] [Mul   ] [|c, a, b| { c.write(a.clone() *  b.clone()); }];
    [OpDivAPI   ] [Div   ] [|c, a, b| { c.write(a.clone() /  b.clone()); }];
    [OpRemAPI   ] [Rem   ] [|c, a, b| { c.write(a.clone() %  b.clone()); }];
    [OpBitOrAPI ] [BitOr ] [|c, a, b| { c.write(a.clone() |  b.clone()); }];
    [OpBitAndAPI] [BitAnd] [|c, a, b| { c.write(a.clone() &  b.clone()); }];
    [OpBitXorAPI] [BitXor] [|c, a, b| { c.write(a.clone() ^  b.clone()); }];
    [OpShlAPI   ] [Shl   ] [|c, a, b| { c.write(a.clone() << b.clone()); }];
    [OpShrAPI   ] [Shr   ] [|c, a, b| { c.write(a.clone() >> b.clone()); }];
)]
impl<TA, TB, TC, D> OpAPI<TA, TB, TC, D> for DeviceCpuSerial
where
    TA: Clone + Op<TB, Output = TC>,
    TB: Clone,
    TC: Clone,
    D: DimAPI,
{
    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut func)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut func)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut func)
    }
}
