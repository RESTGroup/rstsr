#[macro_export]
macro_rules! macro_impl_rayon_op_ternary_common {
    ($Device: ident) => {

use $crate::prelude_dev::*;
use core::cmp::Ord;
use num::{pow::Pow, Float, Integer};

macro_rules! impl_same_binary {
    ($DeviceOpAPI: ident, $TOut: ident, $TraitT: ident, $func:expr) => {
        impl<T, D> $DeviceOpAPI<T, T, D> for $Device
        where
            T: Clone + Send + Sync + $TraitT,
            D: DimAPI,
        {
            type TOut = $TOut;

            fn op_mutc_refa_refb(
                &self,
                c: &mut Storage<$TOut, Self>,
                lc: &Layout<D>,
                a: &Storage<T, Self>,
                la: &Layout<D>,
                b: &Storage<T, Self>,
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
    impl_same_binary!(DeviceATan2API            , T    , Float      , |c, &a, &b| *c = a.atan2(b)               );
    impl_same_binary!(DeviceCopySignAPI         , T    , Float      , |c, &a, &b| *c = a.copysign(b)            );
    impl_same_binary!(DeviceEqualAPI            , bool , PartialEq  , |c,  a,  b| *c = a == b                   );
    impl_same_binary!(DeviceGreaterAPI          , bool , PartialOrd , |c,  a,  b| *c = a > b                    );
    impl_same_binary!(DeviceGreaterEqualAPI     , bool , PartialOrd , |c,  a,  b| *c = a >= b                   );
    impl_same_binary!(DeviceHypotAPI            , T    , Float      , |c, &a, &b| *c = a.hypot(b)               );
    impl_same_binary!(DeviceLessAPI             , bool , PartialOrd , |c,  a,  b| *c = a < b                    );
    impl_same_binary!(DeviceLessEqualAPI        , bool , PartialOrd , |c,  a,  b| *c = a <= b                   );
    impl_same_binary!(DeviceLogAddExpAPI        , T    , Float      , |c, &a, &b| *c = (a.exp() + b.exp()).ln() );
    impl_same_binary!(DeviceMaximumAPI          , T    , Ord        , |c,  a,  b| *c = a.clone().max(b.clone()) );
    impl_same_binary!(DeviceMinimumAPI          , T    , Ord        , |c,  a,  b| *c = a.clone().min(b.clone()) );
    impl_same_binary!(DeviceNotEqualAPI         , bool , PartialEq  , |c,  a,  b| *c = a != b                   );

    impl_same_binary!(DeviceFloatFloorDivideAPI , T    , Float      , |c, &a, &b| *c = a.div(b).floor()         );
    impl_same_binary!(DeviceIntFloorDivideAPI   , T    , Integer    , |c,  a,  b| *c = a.div_floor(b)           );
}

impl<TA, TB, D> DevicePowAPI<TA, TB, D> for $Device
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
        c: &mut Storage<Self::TOut, Self>,
        lc: &Layout<D>,
        a: &Storage<TA, Self>,
        la: &Layout<D>,
        b: &Storage<TB, Self>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut |c, a, b| {
            *c = a.clone().pow(b.clone())
        })
    }
}
    };
}
