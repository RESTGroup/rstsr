use crate::prelude_dev::*;

#[duplicate_item(
    DeviceOpAPI           ;
   [DeviceATan2API       ];
   [DeviceCopySignAPI    ];
   [DeviceEqualAPI       ];
   [DeviceFloorDivideAPI ];
   [DeviceGreaterAPI     ];
   [DeviceGreaterEqualAPI];
   [DeviceHypotAPI       ];
   [DeviceLessAPI        ];
   [DeviceLessEqualAPI   ];
   [DeviceLogAddExpAPI   ];
   [DeviceMaximumAPI     ];
   [DeviceMinimumAPI     ];
   [DeviceNotEqualAPI    ];
   [DevicePowAPI         ];
   [DeviceNextAfterAPI   ];
)]
pub trait DeviceOpAPI<TA, TB, D>
where
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<Self::TOut>>,
{
    type TOut;

    fn op_mutc_refa_refb(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()>;

    fn op_mutc_refa_numb(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()>;

    fn op_mutc_numa_refb(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        lc: &Layout<D>,
        a: TA,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()>;
}

// Python Array API specifications (2023.1)

// not implemented types (defined in arithmetics)
// add, bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift,
// bitwise_xor, divide, logical_and, logical_or, multiply, remainder, subtract

// other common ternary operations

use rstsr_dtype_traits::IsCloseArgs;

pub trait DeviceIsCloseAPI<TA, TB, D, TE>
where
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<bool>>,
{
    fn op_mutc_refa_refb(
        &self,
        out: &mut <Self as DeviceRawAPI<MaybeUninit<bool>>>::Raw,
        lout: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        isclose_args: &IsCloseArgs<TE>,
    ) -> Result<()>;

    fn op_mutc_refa_numb(
        &self,
        out: &mut <Self as DeviceRawAPI<MaybeUninit<bool>>>::Raw,
        lout: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: TB,
        isclose_args: &IsCloseArgs<TE>,
    ) -> Result<()>;

    fn op_mutc_numa_refb(
        &self,
        out: &mut <Self as DeviceRawAPI<MaybeUninit<bool>>>::Raw,
        lout: &Layout<D>,
        a: TA,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        isclose_args: &IsCloseArgs<TE>,
    ) -> Result<()>;
}
