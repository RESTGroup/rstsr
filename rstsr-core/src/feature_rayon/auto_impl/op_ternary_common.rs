use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::{pow::Pow, Float};
use rstsr_dtype_traits::{ExtFloat, ExtReal, PromotionAPI, PromotionSpecialAPI};

// output with special promotion
#[duplicate_item(
     DeviceOpAPI             TraitT           func_inner;
    [DeviceATan2API       ] [Float         ] [Float::atan2(a, b)            ];
    [DeviceCopySignAPI    ] [Float         ] [Float::copysign(a, b)         ];
    [DeviceHypotAPI       ] [Float         ] [Float::hypot(a, b)            ];
    [DeviceNextAfterAPI   ] [ExtFloat      ] [ExtFloat::ext_nextafter(a, b) ];
    [DeviceLogAddExpAPI   ] [ComplexFloat  ] [(a.exp() + b.exp()).ln()      ];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync + PromotionAPI<TB, Res: PromotionSpecialAPI<FloatType: TraitT + Send + Sync>>,
    TB: Clone + Send + Sync,
    D: DimAPI,
{
    type TOut = <TA::Res as PromotionSpecialAPI>::FloatType;

    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            let (a, b) = (a.to_float_type(), b.to_float_type());
            c.write(func_inner);
        };
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut func)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            let (a, b) = (a.to_float_type(), b.to_float_type());
            c.write(func_inner);
        };
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut func)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            let (a, b) = (a.to_float_type(), b.to_float_type());
            c.write(func_inner);
        };
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut func)
    }
}

// general promotion
#[duplicate_item(
     DeviceOpAPI             TO        TraitT           func_inner;
    [DeviceMaximumAPI     ] [TA::Res] [ExtReal       ] [ExtReal::ext_max(a, b)         ];
    [DeviceMinimumAPI     ] [TA::Res] [ExtReal       ] [ExtReal::ext_min(a, b)         ];
    [DeviceFloorDivideAPI ] [TA::Res] [ExtReal       ] [ExtReal::ext_floor_divide(a, b)];
    [DeviceEqualAPI       ] [bool   ] [PartialEq     ] [a == b                         ];
    [DeviceGreaterAPI     ] [bool   ] [PartialOrd    ] [a > b                          ];
    [DeviceGreaterEqualAPI] [bool   ] [PartialOrd    ] [a >= b                         ];
    [DeviceLessAPI        ] [bool   ] [PartialOrd    ] [a < b                          ];
    [DeviceLessEqualAPI   ] [bool   ] [PartialOrd    ] [a <= b                         ];
)]
impl<TA, TB, D> DeviceOpAPI<TA, TB, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync + PromotionAPI<TB, Res: TraitT + Send + Sync>,
    TB: Clone + Send + Sync,
    D: DimAPI,
{
    type TOut = TO;

    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            c.write(func_inner);
        };
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut func)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            c.write(func_inner);
        };
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut func)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            c.write(func_inner);
        };
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut func)
    }
}

// Special case for pow
impl<TA, TB, D> DevicePowAPI<TA, TB, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TA: Pow<TB>,
    TA::Output: Clone + Send + Sync,
    D: DimAPI,
{
    type TOut = TA::Output;

    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut |c, a, b| {
            c.write(a.clone().pow(b.clone()));
        })
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut |c, a, b| {
            c.write(a.clone().pow(b.clone()));
        })
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        lc: &Layout<D>,
        a: TA,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut |c, a, b| {
            c.write(a.clone().pow(b.clone()));
        })
    }
}
