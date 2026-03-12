use crate::prelude_dev::*;
use core::mem::transmute;

#[duplicate_item(
     OpAPI               Op             func                    ;
    [OpAddAssignAPI   ] [AddAssign   ] [|a, b| unsafe { *a.assume_init_mut() +=  b.clone() }];
    [OpSubAssignAPI   ] [SubAssign   ] [|a, b| unsafe { *a.assume_init_mut() -=  b.clone() }];
    [OpMulAssignAPI   ] [MulAssign   ] [|a, b| unsafe { *a.assume_init_mut() *=  b.clone() }];
    [OpDivAssignAPI   ] [DivAssign   ] [|a, b| unsafe { *a.assume_init_mut() /=  b.clone() }];
    [OpRemAssignAPI   ] [RemAssign   ] [|a, b| unsafe { *a.assume_init_mut() %=  b.clone() }];
    [OpBitOrAssignAPI ] [BitOrAssign ] [|a, b| unsafe { *a.assume_init_mut() |=  b.clone() }];
    [OpBitAndAssignAPI] [BitAndAssign] [|a, b| unsafe { *a.assume_init_mut() &=  b.clone() }];
    [OpBitXorAssignAPI] [BitXorAssign] [|a, b| unsafe { *a.assume_init_mut() ^=  b.clone() }];
    [OpShlAssignAPI   ] [ShlAssign   ] [|a, b| unsafe { *a.assume_init_mut() <<= b.clone() }];
    [OpShrAssignAPI   ] [ShrAssign   ] [|a, b| unsafe { *a.assume_init_mut() >>= b.clone() }];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone + Op<TB>,
    TB: Clone,
    D: DimAPI,
{
    fn op_muta_refb(&self, a: &mut Vec<TA>, la: &Layout<D>, b: &Vec<TB>, lb: &Layout<D>) -> Result<()> {
        let a = unsafe { transmute::<&mut Vec<TA>, &mut Vec<MaybeUninit<TA>>>(a) };
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta_numb(&self, a: &mut Vec<TA>, la: &Layout<D>, b: TB) -> Result<()> {
        let a = unsafe { transmute::<&mut Vec<TA>, &mut Vec<MaybeUninit<TA>>>(a) };
        self.op_muta_numb_func(a, la, b, &mut func)
    }
}

#[duplicate_item(
     OpAPI                 Op       func                               ;
    [OpLConsumeAddAPI   ] [Add   ] [|a, b| unsafe { a.write(a.assume_init_read() +  b.clone()); }];
    [OpLConsumeSubAPI   ] [Sub   ] [|a, b| unsafe { a.write(a.assume_init_read() -  b.clone()); }];
    [OpLConsumeMulAPI   ] [Mul   ] [|a, b| unsafe { a.write(a.assume_init_read() *  b.clone()); }];
    [OpLConsumeDivAPI   ] [Div   ] [|a, b| unsafe { a.write(a.assume_init_read() /  b.clone()); }];
    [OpLConsumeRemAPI   ] [Rem   ] [|a, b| unsafe { a.write(a.assume_init_read() %  b.clone()); }];
    [OpLConsumeBitOrAPI ] [BitOr ] [|a, b| unsafe { a.write(a.assume_init_read() |  b.clone()); }];
    [OpLConsumeBitAndAPI] [BitAnd] [|a, b| unsafe { a.write(a.assume_init_read() &  b.clone()); }];
    [OpLConsumeBitXorAPI] [BitXor] [|a, b| unsafe { a.write(a.assume_init_read() ^  b.clone()); }];
    [OpLConsumeShlAPI   ] [Shl   ] [|a, b| unsafe { a.write(a.assume_init_read() << b.clone()); }];
    [OpLConsumeShrAPI   ] [Shr   ] [|a, b| unsafe { a.write(a.assume_init_read() >> b.clone()); }];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone + Op<TB, Output = TA>,
    TB: Clone,
    D: DimAPI,
{
    fn op_muta_refb(&self, a: &mut Vec<TA>, la: &Layout<D>, b: &Vec<TB>, lb: &Layout<D>) -> Result<()> {
        let a = unsafe { transmute::<&mut Vec<TA>, &mut Vec<MaybeUninit<TA>>>(a) };
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta_numb(&self, a: &mut Vec<TA>, la: &Layout<D>, b: TB) -> Result<()> {
        let a = unsafe { transmute::<&mut Vec<TA>, &mut Vec<MaybeUninit<TA>>>(a) };
        self.op_muta_numb_func(a, la, b, &mut func)
    }
}

#[duplicate_item(
     OpAPI                 Op       func                               ;
    [OpRConsumeAddAPI   ] [Add   ] [|a, b| unsafe { a.write(b.clone() +  a.assume_init_read()); }];
    [OpRConsumeSubAPI   ] [Sub   ] [|a, b| unsafe { a.write(b.clone() -  a.assume_init_read()); }];
    [OpRConsumeMulAPI   ] [Mul   ] [|a, b| unsafe { a.write(b.clone() *  a.assume_init_read()); }];
    [OpRConsumeDivAPI   ] [Div   ] [|a, b| unsafe { a.write(b.clone() /  a.assume_init_read()); }];
    [OpRConsumeRemAPI   ] [Rem   ] [|a, b| unsafe { a.write(b.clone() %  a.assume_init_read()); }];
    [OpRConsumeBitOrAPI ] [BitOr ] [|a, b| unsafe { a.write(b.clone() |  a.assume_init_read()); }];
    [OpRConsumeBitAndAPI] [BitAnd] [|a, b| unsafe { a.write(b.clone() &  a.assume_init_read()); }];
    [OpRConsumeBitXorAPI] [BitXor] [|a, b| unsafe { a.write(b.clone() ^  a.assume_init_read()); }];
    [OpRConsumeShlAPI   ] [Shl   ] [|a, b| unsafe { a.write(b.clone() << a.assume_init_read()); }];
    [OpRConsumeShrAPI   ] [Shr   ] [|a, b| unsafe { a.write(b.clone() >> a.assume_init_read()); }];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone + Op<TB, Output = TB>,
    TB: Clone,
    D: DimAPI,
{
    fn op_muta_refb(&self, b: &mut Vec<TB>, lb: &Layout<D>, a: &Vec<TA>, la: &Layout<D>) -> Result<()> {
        let b = unsafe { transmute::<&mut Vec<TB>, &mut Vec<MaybeUninit<TB>>>(b) };
        self.op_muta_refb_func(b, lb, a, la, &mut func)
    }

    fn op_muta_numb(&self, b: &mut Vec<TB>, lb: &Layout<D>, a: TA) -> Result<()> {
        let b = unsafe { transmute::<&mut Vec<TB>, &mut Vec<MaybeUninit<TB>>>(b) };
        self.op_muta_numb_func(b, lb, a, &mut func)
    }
}

#[duplicate_item(
     OpAPI      Op    func                              func_inplace        ;
    [OpNegAPI] [Neg] [|a, b| { a.write(-b.clone()); }] [|a| unsafe { a.write(-a.assume_init_read()); }];
    [OpNotAPI] [Not] [|a, b| { a.write(!b.clone()); }] [|a| unsafe { a.write(!a.assume_init_read()); }];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone + Op<Output = TA>,
    TB: Clone + Op<Output = TA>,
    D: DimAPI,
{
    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<TA>>, la: &Layout<D>, b: &Vec<TB>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<TA>, la: &Layout<D>) -> Result<()> {
        let a = unsafe { transmute::<&mut Vec<TA>, &mut Vec<MaybeUninit<TA>>>(a) };
        self.op_muta_func(a, la, &mut func_inplace)
    }
}
