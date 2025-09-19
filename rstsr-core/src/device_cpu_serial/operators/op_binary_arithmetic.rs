use crate::prelude_dev::*;
use core::mem::transmute;

#[duplicate_item(
     DeviceOpAPI             Op             func                    ;
    [DeviceAddAssignAPI   ] [AddAssign   ] [|a, b| unsafe { *a.assume_init_mut() +=  b.clone() }];
    [DeviceSubAssignAPI   ] [SubAssign   ] [|a, b| unsafe { *a.assume_init_mut() -=  b.clone() }];
    [DeviceMulAssignAPI   ] [MulAssign   ] [|a, b| unsafe { *a.assume_init_mut() *=  b.clone() }];
    [DeviceDivAssignAPI   ] [DivAssign   ] [|a, b| unsafe { *a.assume_init_mut() /=  b.clone() }];
    [DeviceRemAssignAPI   ] [RemAssign   ] [|a, b| unsafe { *a.assume_init_mut() %=  b.clone() }];
    [DeviceBitOrAssignAPI ] [BitOrAssign ] [|a, b| unsafe { *a.assume_init_mut() |=  b.clone() }];
    [DeviceBitAndAssignAPI] [BitAndAssign] [|a, b| unsafe { *a.assume_init_mut() &=  b.clone() }];
    [DeviceBitXorAssignAPI] [BitXorAssign] [|a, b| unsafe { *a.assume_init_mut() ^=  b.clone() }];
    [DeviceShlAssignAPI   ] [ShlAssign   ] [|a, b| unsafe { *a.assume_init_mut() <<= b.clone() }];
    [DeviceShrAssignAPI   ] [ShrAssign   ] [|a, b| unsafe { *a.assume_init_mut() >>= b.clone() }];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceCpuSerial
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
     DeviceOpAPI               Op       func                               ;
    [DeviceLConsumeAddAPI   ] [Add   ] [|a, b| unsafe { a.write(a.assume_init_read() +  b.clone()); }];
    [DeviceLConsumeSubAPI   ] [Sub   ] [|a, b| unsafe { a.write(a.assume_init_read() -  b.clone()); }];
    [DeviceLConsumeMulAPI   ] [Mul   ] [|a, b| unsafe { a.write(a.assume_init_read() *  b.clone()); }];
    [DeviceLConsumeDivAPI   ] [Div   ] [|a, b| unsafe { a.write(a.assume_init_read() /  b.clone()); }];
    [DeviceLConsumeRemAPI   ] [Rem   ] [|a, b| unsafe { a.write(a.assume_init_read() %  b.clone()); }];
    [DeviceLConsumeBitOrAPI ] [BitOr ] [|a, b| unsafe { a.write(a.assume_init_read() |  b.clone()); }];
    [DeviceLConsumeBitAndAPI] [BitAnd] [|a, b| unsafe { a.write(a.assume_init_read() &  b.clone()); }];
    [DeviceLConsumeBitXorAPI] [BitXor] [|a, b| unsafe { a.write(a.assume_init_read() ^  b.clone()); }];
    [DeviceLConsumeShlAPI   ] [Shl   ] [|a, b| unsafe { a.write(a.assume_init_read() << b.clone()); }];
    [DeviceLConsumeShrAPI   ] [Shr   ] [|a, b| unsafe { a.write(a.assume_init_read() >> b.clone()); }];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceCpuSerial
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
     DeviceOpAPI               Op       func                               ;
    [DeviceRConsumeAddAPI   ] [Add   ] [|a, b| unsafe { a.write(b.clone() +  a.assume_init_read()); }];
    [DeviceRConsumeSubAPI   ] [Sub   ] [|a, b| unsafe { a.write(b.clone() -  a.assume_init_read()); }];
    [DeviceRConsumeMulAPI   ] [Mul   ] [|a, b| unsafe { a.write(b.clone() *  a.assume_init_read()); }];
    [DeviceRConsumeDivAPI   ] [Div   ] [|a, b| unsafe { a.write(b.clone() /  a.assume_init_read()); }];
    [DeviceRConsumeRemAPI   ] [Rem   ] [|a, b| unsafe { a.write(b.clone() %  a.assume_init_read()); }];
    [DeviceRConsumeBitOrAPI ] [BitOr ] [|a, b| unsafe { a.write(b.clone() |  a.assume_init_read()); }];
    [DeviceRConsumeBitAndAPI] [BitAnd] [|a, b| unsafe { a.write(b.clone() &  a.assume_init_read()); }];
    [DeviceRConsumeBitXorAPI] [BitXor] [|a, b| unsafe { a.write(b.clone() ^  a.assume_init_read()); }];
    [DeviceRConsumeShlAPI   ] [Shl   ] [|a, b| unsafe { a.write(b.clone() << a.assume_init_read()); }];
    [DeviceRConsumeShrAPI   ] [Shr   ] [|a, b| unsafe { a.write(b.clone() >> a.assume_init_read()); }];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceCpuSerial
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
     DeviceOpAPI    Op    func                              func_inplace        ;
    [DeviceNegAPI] [Neg] [|a, b| { a.write(-b.clone()); }] [|a| unsafe { a.write(-a.assume_init_read()); }];
    [DeviceNotAPI] [Not] [|a, b| { a.write(!b.clone()); }] [|a| unsafe { a.write(!a.assume_init_read()); }];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    D: DimAPI,
{
    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<TA>>, la: &Layout<D>, b: &Vec<TB>, lb: &Layout<D>) -> Result<()>
    where
        TB: Op<Output = TA>,
    {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<TA>, la: &Layout<D>) -> Result<()>
    where
        TA: Op<Output = TA>,
    {
        let a = unsafe { transmute::<&mut Vec<TA>, &mut Vec<MaybeUninit<TA>>>(a) };
        self.op_muta_func(a, la, &mut func_inplace)
    }
}
