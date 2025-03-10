use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::{pow::Pow, Float};
use rstsr_dtype_traits::{FloorDivideAPI, MinMaxAPI};

macro_rules! impl_same_binary {
    ($DeviceOpAPI: ident, $TOut: ident, $TraitT: ident, $func:expr) => {
        impl<T, D> $DeviceOpAPI<T, T, D> for DeviceRayonAutoImpl
        where
            T: Clone + Send + Sync + $TraitT,
            D: DimAPI,
        {
            type TOut = $TOut;

            fn op_mutc_refa_refb(
                &self,
                c: &mut Vec<$TOut>,
                lc: &Layout<D>,
                a: &Vec<T>,
                la: &Layout<D>,
                b: &Vec<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut $func)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_same_binary {

    use super::*;
    impl_same_binary!(DeviceATan2API        , T    , Float          , |c, &a, &b| *c = a.atan2(b)               );
    impl_same_binary!(DeviceCopySignAPI     , T    , Float          , |c, &a, &b| *c = a.copysign(b)            );
    impl_same_binary!(DeviceEqualAPI        , bool , PartialEq      , |c,  a,  b| *c = a == b                   );
    impl_same_binary!(DeviceGreaterAPI      , bool , PartialOrd     , |c,  a,  b| *c = a > b                    );
    impl_same_binary!(DeviceGreaterEqualAPI , bool , PartialOrd     , |c,  a,  b| *c = a >= b                   );
    impl_same_binary!(DeviceHypotAPI        , T    , Float          , |c, &a, &b| *c = a.hypot(b)               );
    impl_same_binary!(DeviceLessAPI         , bool , PartialOrd     , |c,  a,  b| *c = a < b                    );
    impl_same_binary!(DeviceLessEqualAPI    , bool , PartialOrd     , |c,  a,  b| *c = a <= b                   );
    impl_same_binary!(DeviceLogAddExpAPI    , T    , ComplexFloat   , |c, &a, &b| *c = (a.exp() + b.exp()).ln() );
    impl_same_binary!(DeviceMaximumAPI      , T    , MinMaxAPI      , |c,  a,  b| *c = a.clone().max(b.clone()) );
    impl_same_binary!(DeviceMinimumAPI      , T    , MinMaxAPI      , |c,  a,  b| *c = a.clone().min(b.clone()) );
    impl_same_binary!(DeviceNotEqualAPI     , bool , PartialEq      , |c,  a,  b| *c = a != b                   );
    impl_same_binary!(DeviceFloorDivideAPI  , T    , FloorDivideAPI , |c,  a,  b| *c = a.clone().floor_divide(b.clone()));
}

impl<TA, TB, D> DevicePowAPI<TA, TB, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TA: Pow<TB>,
    TA::Output: Clone + Send + Sync,
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
