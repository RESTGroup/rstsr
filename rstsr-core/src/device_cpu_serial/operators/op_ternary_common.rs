use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::{pow::Pow, Float};
use rstsr_dtype_traits::{FloorDivideAPI, MinMaxAPI};

#[duplicate_item(
     DeviceOpAPI             TO     TraitT           func;
    [DeviceATan2API       ] [T   ] [Float         ] [|c, &a, &b| *c = a.atan2(b)                       ];
    [DeviceCopySignAPI    ] [T   ] [Float         ] [|c, &a, &b| *c = a.copysign(b)                    ];
    [DeviceEqualAPI       ] [bool] [PartialEq     ] [|c,  a,  b| *c = a == b                           ];
    [DeviceGreaterAPI     ] [bool] [PartialOrd    ] [|c,  a,  b| *c = a > b                            ];
    [DeviceGreaterEqualAPI] [bool] [PartialOrd    ] [|c,  a,  b| *c = a >= b                           ];
    [DeviceHypotAPI       ] [T   ] [Float         ] [|c, &a, &b| *c = a.hypot(b)                       ];
    [DeviceLessAPI        ] [bool] [PartialOrd    ] [|c,  a,  b| *c = a < b                            ];
    [DeviceLessEqualAPI   ] [bool] [PartialOrd    ] [|c,  a,  b| *c = a <= b                           ];
    [DeviceLogAddExpAPI   ] [T   ] [ComplexFloat  ] [|c, &a, &b| *c = (a.exp() + b.exp()).ln()         ];
    [DeviceMaximumAPI     ] [T   ] [MinMaxAPI     ] [|c,  a,  b| *c = a.clone().max(b.clone())         ];
    [DeviceMinimumAPI     ] [T   ] [MinMaxAPI     ] [|c,  a,  b| *c = a.clone().min(b.clone())         ];
    [DeviceNotEqualAPI    ] [bool] [PartialEq     ] [|c,  a,  b| *c = a != b                           ];
    [DeviceFloorDivideAPI ] [T   ] [FloorDivideAPI] [|c,  a,  b| *c = a.clone().floor_divide(b.clone())];
)]
impl<T, D> DeviceOpAPI<T, T, D> for DeviceCpuSerial
where
    T: Clone + TraitT,
    D: DimAPI,
{
    type TOut = TO;

    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<TO>,
        lc: &Layout<D>,
        a: &Vec<T>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut func)
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
        c: &mut Vec<Self::TOut>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut |c, a, b| {
            *c = a.clone().pow(b.clone())
        })
    }
}
