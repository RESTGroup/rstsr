use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::{pow::Pow, Float};
use rstsr_dtype_traits::{FloorDivideAPI, MinMaxAPI, NextAfterAPI};

#[duplicate_item(
     DeviceOpAPI             TO     TraitT           func;
    [DeviceATan2API       ] [T   ] [Float         ] [|c, &a, &b| { c.write(a.atan2(b)                       ); }];
    [DeviceCopySignAPI    ] [T   ] [Float         ] [|c, &a, &b| { c.write(a.copysign(b)                    ); }];
    [DeviceEqualAPI       ] [bool] [PartialEq     ] [|c,  a,  b| { c.write(a == b                           ); }];
    [DeviceGreaterAPI     ] [bool] [PartialOrd    ] [|c,  a,  b| { c.write(a > b                            ); }];
    [DeviceGreaterEqualAPI] [bool] [PartialOrd    ] [|c,  a,  b| { c.write(a >= b                           ); }];
    [DeviceHypotAPI       ] [T   ] [Float         ] [|c, &a, &b| { c.write(a.hypot(b)                       ); }];
    [DeviceLessAPI        ] [bool] [PartialOrd    ] [|c,  a,  b| { c.write(a < b                            ); }];
    [DeviceLessEqualAPI   ] [bool] [PartialOrd    ] [|c,  a,  b| { c.write(a <= b                           ); }];
    [DeviceLogAddExpAPI   ] [T   ] [ComplexFloat  ] [|c, &a, &b| { c.write((a.exp() + b.exp()).ln()         ); }];
    [DeviceMaximumAPI     ] [T   ] [MinMaxAPI     ] [|c,  a,  b| { c.write(a.clone().max(b.clone())         ); }];
    [DeviceMinimumAPI     ] [T   ] [MinMaxAPI     ] [|c,  a,  b| { c.write(a.clone().min(b.clone())         ); }];
    [DeviceNotEqualAPI    ] [bool] [PartialEq     ] [|c,  a,  b| { c.write(a != b                           ); }];
    [DeviceFloorDivideAPI ] [T   ] [FloorDivideAPI] [|c,  a,  b| { c.write(a.clone().floor_divide(b.clone())); }];
    [DeviceNextAfterAPI   ] [T   ] [NextAfterAPI  ] [|c,  a,  b| { c.write(a.clone().nextafter(b.clone())   ); }];
)]
impl<T, D> DeviceOpAPI<T, T, D> for DeviceCpuSerial
where
    T: Clone + TraitT,
    D: DimAPI,
{
    type TOut = TO;

    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<TO>>,
        lc: &Layout<D>,
        a: &Vec<T>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut func)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut Vec<MaybeUninit<TO>>,
        lc: &Layout<D>,
        a: &Vec<T>,
        la: &Layout<D>,
        b: T,
    ) -> Result<()> {
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut func)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<MaybeUninit<TO>>,
        lc: &Layout<D>,
        a: T,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut func)
    }
}

impl<TA, TB, D> DevicePowAPI<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TA: Pow<TB>,
    TA::Output: Clone,
    D: DimAPI,
{
    type TOut = <TA as Pow<TB>>::Output;

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
