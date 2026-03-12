use crate::prelude_dev::*;

#[duplicate_item(
     OpAPI               Op           ;
    [OpAddAssignAPI   ] [AddAssign   ];
    [OpSubAssignAPI   ] [SubAssign   ];
    [OpMulAssignAPI   ] [MulAssign   ];
    [OpDivAssignAPI   ] [DivAssign   ];
    [OpRemAssignAPI   ] [RemAssign   ];
    [OpBitOrAssignAPI ] [BitOrAssign ];
    [OpBitAndAssignAPI] [BitAndAssign];
    [OpBitXorAssignAPI] [BitXorAssign];
    [OpShlAssignAPI   ] [ShlAssign   ];
    [OpShrAssignAPI   ] [ShrAssign   ];
)]
pub trait OpAPI<TA, TB, D>
where
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
     OpAPI                 Op     ;
    [OpLConsumeAddAPI   ] [Add   ];
    [OpLConsumeSubAPI   ] [Sub   ];
    [OpLConsumeMulAPI   ] [Mul   ];
    [OpLConsumeDivAPI   ] [Div   ];
    [OpLConsumeRemAPI   ] [Rem   ];
    [OpLConsumeBitOrAPI ] [BitOr ];
    [OpLConsumeBitAndAPI] [BitAnd];
    [OpLConsumeBitXorAPI] [BitXor];
    [OpLConsumeShlAPI   ] [Shl   ];
    [OpLConsumeShrAPI   ] [Shr   ];
)]
pub trait OpAPI<TA, TB, D>
where
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
     OpAPI                 Op     ;
    [OpRConsumeAddAPI   ] [Add   ];
    [OpRConsumeSubAPI   ] [Sub   ];
    [OpRConsumeMulAPI   ] [Mul   ];
    [OpRConsumeDivAPI   ] [Div   ];
    [OpRConsumeRemAPI   ] [Rem   ];
    [OpRConsumeBitOrAPI ] [BitOr ];
    [OpRConsumeBitAndAPI] [BitAnd];
    [OpRConsumeBitXorAPI] [BitXor];
    [OpRConsumeShlAPI   ] [Shl   ];
    [OpRConsumeShrAPI   ] [Shr   ];
)]
pub trait OpAPI<TA, TB, D>
where
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
     OpAPI      Op  ;
    [OpNegAPI] [Neg];
    [OpNotAPI] [Not];
)]
pub trait OpAPI<TA, TB, D>
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
    ) -> Result<()>;

    fn op_muta(&self, a: &mut <Self as DeviceRawAPI<TA>>::Raw, la: &Layout<D>) -> Result<()>;
}
