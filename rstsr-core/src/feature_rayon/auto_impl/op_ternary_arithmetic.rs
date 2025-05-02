use crate::prelude_dev::*;

#[duplicate_item(
     DeviceOpAPI       Op       func                                  ;
    [DeviceAddAPI   ] [Add   ] [|c, a, b| *c = a.clone() +  b.clone()];
    [DeviceSubAPI   ] [Sub   ] [|c, a, b| *c = a.clone() -  b.clone()];
    [DeviceMulAPI   ] [Mul   ] [|c, a, b| *c = a.clone() *  b.clone()];
    [DeviceDivAPI   ] [Div   ] [|c, a, b| *c = a.clone() /  b.clone()];
    [DeviceRemAPI   ] [Rem   ] [|c, a, b| *c = a.clone() %  b.clone()];
    [DeviceBitOrAPI ] [BitOr ] [|c, a, b| *c = a.clone() |  b.clone()];
    [DeviceBitAndAPI] [BitAnd] [|c, a, b| *c = a.clone() &  b.clone()];
    [DeviceBitXorAPI] [BitXor] [|c, a, b| *c = a.clone() ^  b.clone()];
    [DeviceShlAPI   ] [Shl   ] [|c, a, b| *c = a.clone() << b.clone()];
    [DeviceShrAPI   ] [Shr   ] [|c, a, b| *c = a.clone() >> b.clone()];
)]
impl<TA, TB, TC, D> DeviceOpAPI<TA, TB, TC, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync + Op<TB, Output = TC>,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync,
    D: DimAPI,
{
    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<TC>,
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
        c: &mut Vec<TC>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut func)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut func)
    }
}
