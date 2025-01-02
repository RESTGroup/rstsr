use crate::prelude_dev::*;
use num::complex::{Complex, ComplexFloat};
use num::{Num, Zero};

// TODO: log1p

/* #region same type */

macro_rules! impl_same_type {
    ($DeviceOpAPI: ident, $NumTrait: ident, $func:expr, $func_inplace:expr) => {
        impl<T, D> $DeviceOpAPI<T, D> for DeviceCpuSerial
        where
            T: Clone + $NumTrait,
            D: DimAPI,
        {
            type TOut = T;

            fn op_muta_refb(
                &self,
                a: &mut Storage<T, Self>,
                la: &Layout<D>,
                b: &Storage<T, Self>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(a, la, b, lb, &mut $func)
            }

            fn op_muta(&self, a: &mut Storage<T, Self>, la: &Layout<D>) -> Result<()> {
                self.op_muta_func(a, la, &mut $func_inplace)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_same_type {
    use super::*;
    use num::{Float, Num};
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
        impl<T, D> $DeviceOpAPI<T, D> for DeviceCpuSerial
        where
            T: Clone + $NumTrait,
            D: DimAPI,
        {
            type TOut = bool;

            fn op_muta_refb(
                &self,
                a: &mut Storage<bool, Self>,
                la: &Layout<D>,
                b: &Storage<T, Self>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(a, la, b, lb, &mut $func)
            }

            fn op_muta(&self, _a: &mut Storage<bool, Self>, _la: &Layout<D>) -> Result<()> {
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

/* #region complex Imag, Real, Abs, Sign */

impl<T, D> DeviceImagAPI<Complex<T>, D> for DeviceCpuSerial
where
    Complex<T>: Clone,
    T: Clone,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut Storage<T, Self>,
        la: &Layout<D>,
        b: &Storage<Complex<T>, Self>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = b.im.clone())
    }

    fn op_muta(&self, _a: &mut Storage<T, Self>, _la: &Layout<D>) -> Result<()> {
        let type_b = core::any::type_name::<Complex<T>>();
        unreachable!("{:?} is not supported in this function", type_b);
    }
}

impl<T, D> DeviceRealAPI<Complex<T>, D> for DeviceCpuSerial
where
    Complex<T>: Clone,
    T: Clone,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut Storage<T, Self>,
        la: &Layout<D>,
        b: &Storage<Complex<T>, Self>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = b.re.clone())
    }

    fn op_muta(&self, _a: &mut Storage<T, Self>, _la: &Layout<D>) -> Result<()> {
        let type_b = core::any::type_name::<Complex<T>>();
        unreachable!("{:?} is not supported in this function", type_b);
    }
}

impl<T, D> DeviceAbsAPI<Complex<T>, D> for DeviceCpuSerial
where
    Complex<T>: Clone + ComplexFloat<Real = T>,
    T: Clone,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut Storage<T, Self>,
        la: &Layout<D>,
        b: &Storage<Complex<T>, Self>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = b.abs())
    }

    fn op_muta(&self, _a: &mut Storage<T, Self>, _la: &Layout<D>) -> Result<()> {
        let type_b = core::any::type_name::<Complex<T>>();
        unreachable!("{:?} is not supported in this function", type_b);
    }
}

impl<T, D> DeviceSignAPI<Complex<T>, D> for DeviceCpuSerial
where
    Complex<T>: Clone + Num + ComplexFloat<Real = T>,
    T: Clone + Zero + Num,
    D: DimAPI,
{
    type TOut = Complex<T>;

    fn op_muta_refb(
        &self,
        a: &mut Storage<Complex<T>, Self>,
        la: &Layout<D>,
        b: &Storage<Complex<T>, Self>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            if *b == Complex::from(T::zero()) {
                *a = Complex::from(T::zero());
            } else {
                *a = b / Complex::<T>::new(b.abs(), T::zero());
            }
        })
    }

    fn op_muta(&self, a: &mut Storage<Complex<T>, Self>, la: &Layout<D>) -> Result<()> {
        self.op_muta_func(a, la, &mut |a| {
            if *a == Complex::from(T::zero()) {
                *a = Complex::from(T::zero());
            } else {
                *a = *a / Complex::<T>::new(a.abs(), T::zero());
            }
        })
    }
}

/* #endregion */

/* #region real Imag, Real, Abs, Sign */

macro_rules! impl_real_specialized {
    ($($t: ty),*) => {
        $(
            impl<D> DeviceImagAPI<$t, D> for DeviceCpuSerial
            where
                $t: Clone,
                D: DimAPI,
            {
                type TOut = $t;

                fn op_muta_refb(
                    &self,
                    a: &mut Storage<$t, Self>,
                    la: &Layout<D>,
                    b: &Storage<$t, Self>,
                    lb: &Layout<D>,
                ) -> Result<()> {
                    self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = b.clone())
                }

                fn op_muta(&self, _a: &mut Storage<$t, Self>, _la: &Layout<D>) -> Result<()> {
                    Ok(())
                }
            }

            impl<D> DeviceRealAPI<$t, D> for DeviceCpuSerial
            where
                $t: Clone,
                D: DimAPI,
            {
                type TOut = $t;

                fn op_muta_refb(
                    &self,
                    a: &mut Storage<$t, Self>,
                    la: &Layout<D>,
                    b: &Storage<$t, Self>,
                    lb: &Layout<D>,
                ) -> Result<()> {
                    self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = b.clone())
                }

                fn op_muta(&self, _a: &mut Storage<$t, Self>, _la: &Layout<D>) -> Result<()> {
                    Ok(())
                }
            }

            impl<D> DeviceAbsAPI<$t, D> for DeviceCpuSerial
            where
                D: DimAPI,
            {
                type TOut = $t;

                fn op_muta_refb(
                    &self,
                    a: &mut Storage<$t, Self>,
                    la: &Layout<D>,
                    b: &Storage<$t, Self>,
                    lb: &Layout<D>,
                ) -> Result<()> {
                    self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = b.abs())
                }

                fn op_muta(&self, a: &mut Storage<$t, Self>, la: &Layout<D>) -> Result<()> {
                    self.op_muta_func(a, la, &mut |a| *a = a.abs())
                }
            }

            impl<D> DeviceSignAPI<$t, D> for DeviceCpuSerial
            where
                D: DimAPI,
            {
                type TOut = $t;

                fn op_muta_refb(
                    &self,
                    a: &mut Storage<$t, Self>,
                    la: &Layout<D>,
                    b: &Storage<$t, Self>,
                    lb: &Layout<D>,
                ) -> Result<()> {
                    self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
                        if *b == <$t>::zero() {
                            *a = <$t>::zero();
                        } else {
                            *a = <$t>::signum(*b);
                        }
                    })
                }

                fn op_muta(&self, a: &mut Storage<$t, Self>, la: &Layout<D>) -> Result<()> {
                    self.op_muta_func(a, la, &mut |a| {
                        if *a != <$t>::zero() {
                            *a = <$t>::signum(*a);
                        }
                    })
                }
            }
        )*
    };
}

impl_real_specialized!(f32, f64, half::bf16, half::f16, i8, i16, i32, i64, i128, isize);

/* #endregion */
