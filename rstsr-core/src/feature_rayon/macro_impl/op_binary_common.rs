#[macro_export]
macro_rules! macro_impl_rayon_op_binary_common {
    ($Device: ident) => {

use $crate::prelude_dev::*;
use core::ops::Div;
use num::complex::ComplexFloat;
use num::{Float, Num};

// TODO: log1p

/* #region same type */

macro_rules! impl_same_type {
    ($DeviceOpAPI: ident, $NumTrait: ident, $func:expr, $func_inplace:expr) => {
        impl<T, D> $DeviceOpAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + $NumTrait,
            D: DimAPI,
        {
            type TOut = T;

            fn op_muta_refb(
                &self,
                a: &mut Vec<T>,
                la: &Layout<D>,
                b: &Vec<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(a, la, b, lb, &mut $func)
            }

            fn op_muta(&self, a: &mut Vec<T>, la: &Layout<D>) -> Result<()> {
                self.op_muta_func(a, la, &mut $func_inplace)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_same_type {
    use super::*;

    impl_same_type!(DeviceAcosAPI         , ComplexFloat , |a, b| *a = b.acos()              , |a| *a = a.acos()             );
    impl_same_type!(DeviceAcoshAPI        , ComplexFloat , |a, b| *a = b.acosh()             , |a| *a = a.acosh()            );
    impl_same_type!(DeviceAsinAPI         , ComplexFloat , |a, b| *a = b.asin()              , |a| *a = a.asin()             );
    impl_same_type!(DeviceAsinhAPI        , ComplexFloat , |a, b| *a = b.asinh()             , |a| *a = a.asinh()            );
    impl_same_type!(DeviceAtanAPI         , ComplexFloat , |a, b| *a = b.atan()              , |a| *a = a.atan()             );
    impl_same_type!(DeviceAtanhAPI        , ComplexFloat , |a, b| *a = b.atanh()             , |a| *a = a.atanh()            );
    impl_same_type!(DeviceCeilAPI         , Float        , |a, b| *a = b.ceil()              , |a| *a = a.ceil()             );
    impl_same_type!(DeviceConjAPI         , ComplexFloat , |a, b| *a = b.conj()              , |a| *a = a.conj()             );
    impl_same_type!(DeviceCosAPI          , ComplexFloat , |a, b| *a = b.cos()               , |a| *a = a.cos()              );
    impl_same_type!(DeviceCoshAPI         , ComplexFloat , |a, b| *a = b.cosh()              , |a| *a = a.cosh()             );
    impl_same_type!(DeviceExpAPI          , ComplexFloat , |a, b| *a = b.exp()               , |a| *a = a.exp()              );
    impl_same_type!(DeviceExpm1API        , Float        , |a, b| *a = b.exp_m1()            , |a| *a = a.exp_m1()           );
    impl_same_type!(DeviceFloorAPI        , Float        , |a, b| *a = b.floor()             , |a| *a = a.floor()            );
    impl_same_type!(DeviceInvAPI          , ComplexFloat , |a, b| *a = b.recip()             , |a| *a = a.recip()            );
    impl_same_type!(DeviceLogAPI          , ComplexFloat , |a, b| *a = b.ln()                , |a| *a = a.ln()               );
    impl_same_type!(DeviceLog2API         , ComplexFloat , |a, b| *a = b.log2()              , |a| *a = a.log2()             );
    impl_same_type!(DeviceLog10API        , ComplexFloat , |a, b| *a = b.log10()             , |a| *a = a.log10()            );
    impl_same_type!(DeviceRoundAPI        , Float        , |a, b| *a = b.round()             , |a| *a = a.round()            );
    impl_same_type!(DeviceSinAPI          , ComplexFloat , |a, b| *a = b.sin()               , |a| *a = a.sin()              );
    impl_same_type!(DeviceSinhAPI         , ComplexFloat , |a, b| *a = b.sinh()              , |a| *a = a.sinh()             );
    impl_same_type!(DeviceSqrtAPI         , ComplexFloat , |a, b| *a = b.sqrt()              , |a| *a = a.sqrt()             );
    impl_same_type!(DeviceSquareAPI       , Num          , |a, b| *a = b.clone() * b.clone() , |a| *a = a.clone() * a.clone());
    impl_same_type!(DeviceTanAPI          , ComplexFloat , |a, b| *a = b.tan()               , |a| *a = a.tan()              );
    impl_same_type!(DeviceTanhAPI         , ComplexFloat , |a, b| *a = b.tanh()              , |a| *a = a.tanh()             );
    impl_same_type!(DeviceTruncAPI        , Float        , |a, b| *a = b.trunc()             , |a| *a = a.trunc()            );
}

/* #endregion */

/* #region boolean output */

macro_rules! impl_boolean_output {
    ($DeviceOpAPI: ident, $NumTrait: ident, $func:expr) => {
        impl<T, D> $DeviceOpAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + $NumTrait,
            D: DimAPI,
        {
            type TOut = bool;

            fn op_muta_refb(
                &self,
                a: &mut Vec<bool>,
                la: &Layout<D>,
                b: &Vec<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(a, la, b, lb, &mut $func)
            }

            fn op_muta(&self, _a: &mut Vec<bool>, _la: &Layout<D>) -> Result<()> {
                let type_b = core::any::type_name::<T>();
                unreachable!("{:?} is not supported in this function.", type_b);
            }
        }
    };
}

#[rustfmt::skip]
mod impl_bool_output{
    use super::*;
    use num::{Float, Signed};
    impl_boolean_output!(DeviceSignBitAPI  , Signed , |a, b| *a = b.is_positive() );
    impl_boolean_output!(DeviceIsFiniteAPI , Float  , |a, b| *a = b.is_finite()   );
    impl_boolean_output!(DeviceIsInfAPI    , Float  , |a, b| *a = b.is_infinite() );
    impl_boolean_output!(DeviceIsNanAPI    , Float  , |a, b| *a = b.is_nan()      );
}

/* #endregion */

/* #region complex specific implementation */

macro_rules! impl_complex_diff_type {
    ($DeviceOpAPI: ident, $func:expr, $func_inplace: expr) => {
        impl<T, D> $DeviceOpAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + ComplexFloat,
            T::Real: Clone + Send + Sync,
            D: DimAPI,
        {
            type TOut = T::Real;

            fn op_muta_refb(
                &self,
                a: &mut Vec<T::Real>,
                la: &Layout<D>,
                b: &Vec<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(a, la, b, lb, &mut $func)
            }

            fn op_muta(&self, a: &mut Vec<T::Real>, la: &Layout<D>) -> Result<()> {
                let type_b = core::any::type_name::<T::Real>();
                let type_a = core::any::type_name::<T>();
                if type_b == type_a {
                    self.op_muta_func(a, la, &mut $func_inplace)
                } else {
                    unreachable!("{:?} is not supported in this function.", type_b);
                }
            }
        }
    };
}

#[rustfmt::skip]
mod impl_complex_diff_type {
    use super::*;
    impl_complex_diff_type!(DeviceAbsAPI  , |a, b| *a = b.abs(), |a| *a = ComplexFloat::abs(a.clone()) );
    impl_complex_diff_type!(DeviceImagAPI , |a, b| *a = b.im() , |a| *a = a.im() );
    impl_complex_diff_type!(DeviceRealAPI , |a, b| *a = b.re() , |a| *a = a.re() );
}

impl<T, D> DeviceSignAPI<T, D> for $Device
where
    T: Clone + Send + Sync + ComplexFloat + Div<T::Real, Output = T>,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut Vec<T>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = *b / b.abs())
    }

    fn op_muta(&self, a: &mut Vec<T>, la: &Layout<D>) -> Result<()> {
        self.op_muta_func(a, la, &mut |a| *a = *a / a.abs())
    }
}

/* #endregion */


    };
}
