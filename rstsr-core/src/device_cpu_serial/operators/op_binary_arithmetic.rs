use crate::prelude_dev::*;
use core::ops::*;

#[duplicate_item(
     DeviceOpAPI             Op             func                    ;
    [DeviceAddAssignAPI   ] [AddAssign   ] [|a, b| *a +=  b.clone()];
    [DeviceSubAssignAPI   ] [SubAssign   ] [|a, b| *a -=  b.clone()];
    [DeviceMulAssignAPI   ] [MulAssign   ] [|a, b| *a *=  b.clone()];
    [DeviceDivAssignAPI   ] [DivAssign   ] [|a, b| *a /=  b.clone()];
    [DeviceRemAssignAPI   ] [RemAssign   ] [|a, b| *a %=  b.clone()];
    [DeviceBitOrAssignAPI ] [BitOrAssign ] [|a, b| *a |=  b.clone()];
    [DeviceBitAndAssignAPI] [BitAndAssign] [|a, b| *a &=  b.clone()];
    [DeviceBitXorAssignAPI] [BitXorAssign] [|a, b| *a ^=  b.clone()];
    [DeviceShlAssignAPI   ] [ShlAssign   ] [|a, b| *a <<= b.clone()];
    [DeviceShrAssignAPI   ] [ShrAssign   ] [|a, b| *a >>= b.clone()];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone + Op<TB>,
    TB: Clone,
    D: DimAPI,
{
    fn op_muta_refb(
        &self,
        a: &mut Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta_numb(&self, a: &mut Vec<TA>, la: &Layout<D>, b: TB) -> Result<()> {
        self.op_muta_numb_func(a, la, b, &mut func)
    }
}

#[duplicate_item(
     DeviceOpAPI               Op       func                               ;
    [DeviceLConsumeAddAPI   ] [Add   ] [|a, b| *a = a.clone() +  b.clone()];
    [DeviceLConsumeSubAPI   ] [Sub   ] [|a, b| *a = a.clone() -  b.clone()];
    [DeviceLConsumeMulAPI   ] [Mul   ] [|a, b| *a = a.clone() *  b.clone()];
    [DeviceLConsumeDivAPI   ] [Div   ] [|a, b| *a = a.clone() /  b.clone()];
    [DeviceLConsumeRemAPI   ] [Rem   ] [|a, b| *a = a.clone() %  b.clone()];
    [DeviceLConsumeBitOrAPI ] [BitOr ] [|a, b| *a = a.clone() |  b.clone()];
    [DeviceLConsumeBitAndAPI] [BitAnd] [|a, b| *a = a.clone() &  b.clone()];
    [DeviceLConsumeBitXorAPI] [BitXor] [|a, b| *a = a.clone() ^  b.clone()];
    [DeviceLConsumeShlAPI   ] [Shl   ] [|a, b| *a = a.clone() << b.clone()];
    [DeviceLConsumeShrAPI   ] [Shr   ] [|a, b| *a = a.clone() >> b.clone()];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone + Op<TB, Output = TA>,
    TB: Clone,
    D: DimAPI,
{
    fn op_muta_refb(
        &self,
        a: &mut Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta_numb(&self, a: &mut Vec<TA>, la: &Layout<D>, b: TB) -> Result<()> {
        self.op_muta_numb_func(a, la, b, &mut func)
    }
}

#[duplicate_item(
     DeviceOpAPI               Op       func                               ;
    [DeviceRConsumeAddAPI   ] [Add   ] [|a, b| *a = b.clone() +  a.clone()];
    [DeviceRConsumeSubAPI   ] [Sub   ] [|a, b| *a = b.clone() -  a.clone()];
    [DeviceRConsumeMulAPI   ] [Mul   ] [|a, b| *a = b.clone() *  a.clone()];
    [DeviceRConsumeDivAPI   ] [Div   ] [|a, b| *a = b.clone() /  a.clone()];
    [DeviceRConsumeRemAPI   ] [Rem   ] [|a, b| *a = b.clone() %  a.clone()];
    [DeviceRConsumeBitOrAPI ] [BitOr ] [|a, b| *a = b.clone() |  a.clone()];
    [DeviceRConsumeBitAndAPI] [BitAnd] [|a, b| *a = b.clone() &  a.clone()];
    [DeviceRConsumeBitXorAPI] [BitXor] [|a, b| *a = b.clone() ^  a.clone()];
    [DeviceRConsumeShlAPI   ] [Shl   ] [|a, b| *a = b.clone() << a.clone()];
    [DeviceRConsumeShrAPI   ] [Shr   ] [|a, b| *a = b.clone() >> a.clone()];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone + Op<TB, Output = TB>,
    TB: Clone,
    D: DimAPI,
{
    fn op_muta_refb(
        &self,
        b: &mut Vec<TB>,
        lb: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(b, lb, a, la, &mut func)
    }

    fn op_muta_numb(&self, b: &mut Vec<TB>, lb: &Layout<D>, a: TA) -> Result<()> {
        self.op_muta_numb_func(b, lb, a, &mut func)
    }
}

#[duplicate_item(
     DeviceOpAPI    Op    func                     func_inplace        ;
    [DeviceNegAPI] [Neg] [|a, b| *a = -b.clone()] [|a| *a = -a.clone()];
    [DeviceNotAPI] [Not] [|a, b| *a = !b.clone()] [|a| *a = !a.clone()];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    D: DimAPI,
{
    fn op_muta_refb(
        &self,
        a: &mut Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()>
    where
        TB: Op<Output = TA>,
    {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<TA>, la: &Layout<D>) -> Result<()>
    where
        TA: Op<Output = TA>,
    {
        self.op_muta_func(a, la, &mut func_inplace)
    }
}
