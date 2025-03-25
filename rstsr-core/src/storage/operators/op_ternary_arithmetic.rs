use crate::prelude_dev::*;
use core::ops::*;

#[duplicate_item(
     DeviceOpAPI       Op     ;
    [DeviceAddAPI   ] [Add   ];
    [DeviceSubAPI   ] [Sub   ];
    [DeviceMulAPI   ] [Mul   ];
    [DeviceDivAPI   ] [Div   ];
    [DeviceRemAPI   ] [Rem   ];
    [DeviceBitOrAPI ] [BitOr ];
    [DeviceBitAndAPI] [BitAnd];
    [DeviceBitXorAPI] [BitXor];
    [DeviceShlAPI   ] [Shl   ];
    [DeviceShrAPI   ] [Shr   ];
)]
pub trait DeviceOpAPI<TA, TB, TC, D>
where
    TA: Op<TB, Output = TC>,
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn op_mutc_refa_refb(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()>;

    fn op_mutc_refa_numb(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()>;

    fn op_mutc_numa_refb(
        &self,
        c: &mut <Self as DeviceRawAPI<TC>>::Raw,
        lc: &Layout<D>,
        a: TA,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()>;
}
