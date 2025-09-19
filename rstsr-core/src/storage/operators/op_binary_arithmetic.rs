use crate::prelude_dev::*;

#[duplicate_item(
     DeviceOpAPI             Op           ;
    [DeviceAddAssignAPI   ] [AddAssign   ];
    [DeviceSubAssignAPI   ] [SubAssign   ];
    [DeviceMulAssignAPI   ] [MulAssign   ];
    [DeviceDivAssignAPI   ] [DivAssign   ];
    [DeviceRemAssignAPI   ] [RemAssign   ];
    [DeviceBitOrAssignAPI ] [BitOrAssign ];
    [DeviceBitAndAssignAPI] [BitAndAssign];
    [DeviceBitXorAssignAPI] [BitXorAssign];
    [DeviceShlAssignAPI   ] [ShlAssign   ];
    [DeviceShrAssignAPI   ] [ShrAssign   ];
)]
pub trait DeviceOpAPI<TA, TB, D>
where
    TA: Op<TB>,
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB>,
{
    fn op_muta_refb(
        &self,
        a: &mut <Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()>;

    fn op_muta_numb(&self, a: &mut <Self as DeviceRawAPI<TA>>::Raw, la: &Layout<D>, b: TB) -> Result<()>;
}

#[duplicate_item(
     DeviceOpAPI               Op     ;
    [DeviceLConsumeAddAPI   ] [Add   ];
    [DeviceLConsumeSubAPI   ] [Sub   ];
    [DeviceLConsumeMulAPI   ] [Mul   ];
    [DeviceLConsumeDivAPI   ] [Div   ];
    [DeviceLConsumeRemAPI   ] [Rem   ];
    [DeviceLConsumeBitOrAPI ] [BitOr ];
    [DeviceLConsumeBitAndAPI] [BitAnd];
    [DeviceLConsumeBitXorAPI] [BitXor];
    [DeviceLConsumeShlAPI   ] [Shl   ];
    [DeviceLConsumeShrAPI   ] [Shr   ];
)]
pub trait DeviceOpAPI<TA, TB, D>
where
    TA: Op<TB, Output = TA>,
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB>,
{
    fn op_muta_refb(
        &self,
        a: &mut <Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()>;

    fn op_muta_numb(&self, a: &mut <Self as DeviceRawAPI<TA>>::Raw, la: &Layout<D>, b: TB) -> Result<()>;
}

#[duplicate_item(
     DeviceOpAPI               Op     ;
    [DeviceRConsumeAddAPI   ] [Add   ];
    [DeviceRConsumeSubAPI   ] [Sub   ];
    [DeviceRConsumeMulAPI   ] [Mul   ];
    [DeviceRConsumeDivAPI   ] [Div   ];
    [DeviceRConsumeRemAPI   ] [Rem   ];
    [DeviceRConsumeBitOrAPI ] [BitOr ];
    [DeviceRConsumeBitAndAPI] [BitAnd];
    [DeviceRConsumeBitXorAPI] [BitXor];
    [DeviceRConsumeShlAPI   ] [Shl   ];
    [DeviceRConsumeShrAPI   ] [Shr   ];
)]
pub trait DeviceOpAPI<TA, TB, D>
where
    TA: Op<TB, Output = TB>,
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB>,
{
    fn op_muta_refb(
        &self,
        b: &mut <Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
    ) -> Result<()>;

    fn op_muta_numb(&self, b: &mut <Self as DeviceRawAPI<TB>>::Raw, lb: &Layout<D>, a: TA) -> Result<()>;
}

#[duplicate_item(
     DeviceOpAPI    Op  ;
    [DeviceNegAPI] [Neg];
    [DeviceNotAPI] [Not];
)]
pub trait DeviceOpAPI<TA, TB, D>
where
    D: DimAPI,
    Self: DeviceAPI<MaybeUninit<TA>> + DeviceAPI<TA> + DeviceAPI<TB>,
{
    fn op_muta_refb(
        &self,
        a: &mut <Self as DeviceRawAPI<MaybeUninit<TA>>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()>
    where
        TB: Op<Output = TA>;

    fn op_muta(&self, a: &mut <Self as DeviceRawAPI<TA>>::Raw, la: &Layout<D>) -> Result<()>
    where
        TA: Op<Output = TA>;
}
