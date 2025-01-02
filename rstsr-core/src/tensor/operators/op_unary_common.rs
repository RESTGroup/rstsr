use crate::prelude_dev::*;

/* Structure of implementation

Most unary functions are of the same type. However, there are some exceptions, and some of them are very common used functions.

- `same type`: Input and Output are of the same type. They can be implemented in an inplace manner.
- `boolean output`: Output is boolean. Not able for inplace operation.
- `Imag, Real, Abs`:
    - complex: generalized, not for inplace.
    - real: specialized, for inplace.
- `Sign`:
    - complex: generalized, for inplace.
    - real: specialized, for inplace.

*/

/* #region tensor traits */

macro_rules! trait_unary {
    ($op: ident, $op_f: ident, $TensorOpAPI: ident) => {
        pub trait $TensorOpAPI: Sized {
            type Output;
            fn $op_f(self) -> Result<Self::Output>;
            fn $op(self) -> Self::Output {
                self.$op_f().unwrap()
            }
        }

        pub fn $op_f<TRA, TRB>(a: TRA) -> Result<TRB>
        where
            TRA: $TensorOpAPI<Output = TRB>,
        {
            TRA::$op_f(a)
        }

        pub fn $op<TRA, TRB>(a: TRA) -> TRB
        where
            TRA: $TensorOpAPI<Output = TRB>,
        {
            TRA::$op(a)
        }
    };
}

#[rustfmt::skip]
#[allow(clippy::wrong_self_convention)]
mod trait_unary {
    use super::*;
    trait_unary!(abs       , abs_f       , TensorAbsAPI      );
    trait_unary!(acos      , acos_f      , TensorAcosAPI     );
    trait_unary!(acosh     , acosh_f     , TensorAcoshAPI    );
    trait_unary!(asin      , asin_f      , TensorAsinAPI     );
    trait_unary!(asinh     , asinh_f     , TensorAsinhAPI    );
    trait_unary!(atan      , atan_f      , TensorAtanAPI     );
    trait_unary!(atanh     , atanh_f     , TensorAtanhAPI    );
    trait_unary!(ceil      , ceil_f      , TensorCeilAPI     );
    trait_unary!(conj      , conj_f      , TensorConjAPI     );
    trait_unary!(cos       , cos_f       , TensorCosAPI      );
    trait_unary!(cosh      , cosh_f      , TensorCoshAPI     );
    trait_unary!(exp       , exp_f       , TensorExpAPI      );
    trait_unary!(expm1     , expm1_f     , TensorExpm1API    );
    trait_unary!(floor     , floor_f     , TensorFloorAPI    );
    trait_unary!(inv       , inv_f       , TensorInvAPI      );
    trait_unary!(log       , log_f       , TensorLogAPI      );
    trait_unary!(log1p     , log1p_f     , TensorLog1pAPI    );
    trait_unary!(log2      , log2_f      , TensorLog2API     );
    trait_unary!(log10     , log10_f     , TensorLog10API    );
    trait_unary!(round     , round_f     , TensorRoundAPI    );
    trait_unary!(sign      , sign_f      , TensorSignAPI     );
    trait_unary!(signbit   , signbit_f   , TensorSignBitAPI  );
    trait_unary!(sin       , sin_f       , TensorSinAPI      );
    trait_unary!(sinh      , sinh_f      , TensorSinhAPI     );
    trait_unary!(square    , square_f    , TensorSquareAPI   );
    trait_unary!(sqrt      , sqrt_f      , TensorSqrtAPI     );
    trait_unary!(tan       , tan_f       , TensorTanAPI      );
    trait_unary!(tanh      , tanh_f      , TensorTanhAPI     );
    trait_unary!(trunc     , trunc_f     , TensorTruncAPI    );
    trait_unary!(imag      , imag_f      , TensorImagAPI     );
    trait_unary!(is_finite , is_finite_f , TensorIsFiniteAPI );
    trait_unary!(is_inf    , is_inf_f    , TensorIsInfAPI    );
    trait_unary!(is_nan    , is_nan_f    , TensorIsNanAPI    );
    trait_unary!(real      , real_f      , TensorRealAPI     );
}

use num::Complex;
pub use trait_unary::*;

/* #endregion */

/* #region same type */

macro_rules! impl_same_type {
    ($op_f: ident, $TensorOpAPI: ident, $DeviceOpAPI: ident) => {
        impl<R, T, D, B> $TensorOpAPI for &TensorBase<R, D>
        where
            D: DimAPI,
            R: DataAPI<Data = Storage<T, B>>,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<T, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                let lb = self.layout();
                let storage_b = self.data().storage();
                // generate empty output tensor
                let device = self.device();
                let la = layout_for_array_copy(lb, TensorIterOrder::K)?;
                let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
                // compute and return
                device.op_muta_refb(&mut storage_a, &la, storage_b, lb)?;
                return Tensor::new_f(DataOwned::from(storage_a), la);
            }
        }

        impl<'l, T, D, B> $TensorOpAPI for TensorView<'l, T, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<T, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                $TensorOpAPI::$op_f(&self)
            }
        }

        impl<T, D, B> $TensorOpAPI for Tensor<T, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<T, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op_f(mut self) -> Result<Self::Output> {
                let layout = self.layout().clone();
                let device = self.device().clone();
                let storage = self.data_mut().storage_mut();
                // generate empty output tensor
                device.op_muta(storage, &layout)?;
                return Ok(self);
            }
        }
    };
}

#[rustfmt::skip]
#[allow(clippy::wrong_self_convention)]
mod impl_same_type {
    use super::*;
    impl_same_type!(acos_f      , TensorAcosAPI     , DeviceAcosAPI     );
    impl_same_type!(acosh_f     , TensorAcoshAPI    , DeviceAcoshAPI    );
    impl_same_type!(asin_f      , TensorAsinAPI     , DeviceAsinAPI     );
    impl_same_type!(asinh_f     , TensorAsinhAPI    , DeviceAsinhAPI    );
    impl_same_type!(atan_f      , TensorAtanAPI     , DeviceAtanAPI     );
    impl_same_type!(atanh_f     , TensorAtanhAPI    , DeviceAtanhAPI    );
    impl_same_type!(ceil_f      , TensorCeilAPI     , DeviceCeilAPI     );
    impl_same_type!(conj_f      , TensorConjAPI     , DeviceConjAPI     );
    impl_same_type!(cos_f       , TensorCosAPI      , DeviceCosAPI      );
    impl_same_type!(cosh_f      , TensorCoshAPI     , DeviceCoshAPI     );
    impl_same_type!(exp_f       , TensorExpAPI      , DeviceExpAPI      );
    impl_same_type!(expm1_f     , TensorExpm1API    , DeviceExpm1API    );
    impl_same_type!(floor_f     , TensorFloorAPI    , DeviceFloorAPI    );
    impl_same_type!(inv_f       , TensorInvAPI      , DeviceInvAPI      );
    impl_same_type!(log_f       , TensorLogAPI      , DeviceLogAPI      );
    impl_same_type!(log1p_f     , TensorLog1pAPI    , DeviceLog1pAPI    );
    impl_same_type!(log2_f      , TensorLog2API     , DeviceLog2API     );
    impl_same_type!(log10_f     , TensorLog10API    , DeviceLog10API    );
    impl_same_type!(round_f     , TensorRoundAPI    , DeviceRoundAPI    );
    impl_same_type!(sign_f      , TensorSignAPI     , DeviceSignAPI     );
    impl_same_type!(sin_f       , TensorSinAPI      , DeviceSinAPI      );
    impl_same_type!(sinh_f      , TensorSinhAPI     , DeviceSinhAPI     );
    impl_same_type!(square_f    , TensorSquareAPI   , DeviceSquareAPI   );
    impl_same_type!(sqrt_f      , TensorSqrtAPI     , DeviceSqrtAPI     );
    impl_same_type!(tan_f       , TensorTanAPI      , DeviceTanAPI      );
    impl_same_type!(tanh_f      , TensorTanhAPI     , DeviceTanhAPI     );
    impl_same_type!(trunc_f     , TensorTruncAPI    , DeviceTruncAPI    );
}

/* #endregion */

/* #region boolean output */

macro_rules! impl_boolean_output {
    ($op_f: ident, $TensorOpAPI: ident, $DeviceOpAPI: ident) => {
        impl<R, T, D, B> $TensorOpAPI for &TensorBase<R, D>
        where
            D: DimAPI,
            R: DataAPI<Data = Storage<T, B>>,
            B: DeviceAPI<T> + DeviceAPI<B::TOut>,
            B: $DeviceOpAPI<T, D> + DeviceCreationAnyAPI<B::TOut>,
        {
            type Output = Tensor<B::TOut, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                let lb = self.layout();
                let storage_b = self.data().storage();
                // generate empty output tensor
                let device = self.device();
                let la = layout_for_array_copy(lb, TensorIterOrder::K)?;
                let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
                // compute and return
                device.op_muta_refb(&mut storage_a, &la, storage_b, lb)?;
                return Tensor::new_f(DataOwned::from(storage_a), la);
            }
        }

        impl<'l, T, D, B> $TensorOpAPI for TensorView<'l, T, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T> + DeviceAPI<B::TOut>,
            B: $DeviceOpAPI<T, D> + DeviceCreationAnyAPI<B::TOut>,
        {
            type Output = Tensor<B::TOut, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                $TensorOpAPI::$op_f(&self)
            }
        }

        impl<T, D, B> $TensorOpAPI for Tensor<T, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<T, D> + DeviceCreationAnyAPI<B::TOut>,
        {
            type Output = Tensor<B::TOut, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }
    };
}

#[rustfmt::skip]
#[allow(clippy::wrong_self_convention)]
mod impl_boolean_output {
    use super::*;
    impl_boolean_output!(signbit_f   , TensorSignBitAPI  , DeviceSignBitAPI  );
    impl_boolean_output!(is_finite_f , TensorIsFiniteAPI , DeviceIsFiniteAPI );
    impl_boolean_output!(is_inf_f    , TensorIsInfAPI    , DeviceIsInfAPI    );
    impl_boolean_output!(is_nan_f    , TensorIsNanAPI    , DeviceIsNanAPI    );
}

/* #endregion */

/* #region complex Real, Imag, Abs */

macro_rules! impl_complex_specialization {
    ($op_f: ident, $TensorOpAPI: ident, $DeviceOpAPI: ident) => {
        impl<'l, T, D, B> $TensorOpAPI for TensorView<'l, Complex<T>, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<Complex<T>, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                let lb = self.layout();
                let storage_b = self.data().storage();
                // generate empty output tensor
                let device = self.device();
                let la = layout_for_array_copy(lb, TensorIterOrder::K)?;
                let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
                // compute and return
                device.op_muta_refb(&mut storage_a, &la, storage_b, lb)?;
                return Tensor::new_f(DataOwned::from(storage_a), la);
            }
        }

        impl<T, D, B> $TensorOpAPI for Tensor<Complex<T>, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<Complex<T>, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }

        impl<T, D, B> $TensorOpAPI for &TensorView<'_, Complex<T>, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<Complex<T>, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }

        impl<T, D, B> $TensorOpAPI for &TensorCow<'_, Complex<T>, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<Complex<T>, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }

        impl<T, D, B> $TensorOpAPI for &Tensor<Complex<T>, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<Complex<T>, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }

        impl<T, D, B> $TensorOpAPI for &TensorArc<Complex<T>, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<Complex<T>, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }
    };
}

impl_complex_specialization!(abs_f, TensorAbsAPI, DeviceAbsAPI);
impl_complex_specialization!(imag_f, TensorImagAPI, DeviceImagAPI);
impl_complex_specialization!(real_f, TensorRealAPI, DeviceRealAPI);

/* #endregion */

/* #region real Real, Imag, Abs */

macro_rules! impl_real_specialized {
    ($t: ty, $op_f: ident, $TensorOpAPI: ident, $DeviceOpAPI: ident) => {
        impl<'l, D, B> $TensorOpAPI for TensorView<'l, $t, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<$t>,
            B: $DeviceOpAPI<$t, D, TOut = $t> + DeviceCreationAnyAPI<$t>,
        {
            type Output = Tensor<$t, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                let lb = self.layout();
                let storage_b = self.data().storage();
                // generate empty output tensor
                let device = self.device();
                let la = layout_for_array_copy(lb, TensorIterOrder::K)?;
                let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
                // compute and return
                device.op_muta_refb(&mut storage_a, &la, storage_b, lb)?;
                return Tensor::new_f(DataOwned::from(storage_a), la);
            }
        }

        impl<D, B> $TensorOpAPI for Tensor<$t, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<$t>,
            B: $DeviceOpAPI<$t, D, TOut = $t> + DeviceCreationAnyAPI<$t>,
        {
            type Output = Tensor<$t, D, B>;
            fn $op_f(mut self) -> Result<Self::Output> {
                let layout = self.layout().clone();
                let device = self.device().clone();
                let storage = self.data_mut().storage_mut();
                // generate empty output tensor
                device.op_muta(storage, &layout)?;
                return Ok(self);
            }
        }

        impl<D, B> $TensorOpAPI for &TensorView<'_, $t, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<$t>,
            B: $DeviceOpAPI<$t, D, TOut = $t> + DeviceCreationAnyAPI<$t>,
        {
            type Output = Tensor<$t, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }

        impl<D, B> $TensorOpAPI for &TensorCow<'_, $t, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<$t>,
            B: $DeviceOpAPI<$t, D, TOut = $t> + DeviceCreationAnyAPI<$t>,
        {
            type Output = Tensor<$t, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }

        impl<D, B> $TensorOpAPI for &Tensor<$t, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<$t>,
            B: $DeviceOpAPI<$t, D, TOut = $t> + DeviceCreationAnyAPI<$t>,
        {
            type Output = Tensor<$t, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }

        impl<D, B> $TensorOpAPI for &TensorArc<$t, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<$t>,
            B: $DeviceOpAPI<$t, D, TOut = $t> + DeviceCreationAnyAPI<$t>,
        {
            type Output = Tensor<$t, D, B>;
            fn $op_f(self) -> Result<Self::Output> {
                self.view().$op_f()
            }
        }
    };
}

macro_rules! impl_outer_real_specialized {
    ($($t: ty),*) => {
        $(
            impl_real_specialized!($t, abs_f, TensorAbsAPI, DeviceAbsAPI);
            impl_real_specialized!($t, imag_f, TensorImagAPI, DeviceImagAPI);
            impl_real_specialized!($t, real_f, TensorRealAPI, DeviceRealAPI);
        )*
    };
}

impl_outer_real_specialized!(f32, f64, half::bf16, half::f16, i8, i16, i32, i64, i128, isize);

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_same_type() {
        let a = arange(6.0).into_shape([2, 3]).into_owned();
        let b = sin(&a);
        println!("{:}", b);
        let b = a.view().sin();
        println!("{:}", b);

        let ptr_a = a.rawvec().as_ptr();
        let b = a.sin();
        let ptr_b = b.rawvec().as_ptr();
        assert_eq!(ptr_a, ptr_b);
    }

    #[test]
    fn test_sign() {
        use num::complex::c64;
        let a = linspace((c64(1.0, 2.0), c64(5.0, 6.0), 6)).into_shape([2, 3]);
        let b = sign(&a);
        let vec_b = b.reshape([6]).to_vec();
        let b_abs_sum = vec_b.iter().map(|x| x.norm()).sum::<f64>();
        println!("{:}", b);
        assert!(b_abs_sum - 6.0 < 1e-6);
    }

    #[test]
    fn test_abs() {
        use num::complex::c32;
        let a = linspace((c32(1.0, 2.0), c32(5.0, 6.0), 6)).into_shape([2, 3]);
        let ptr_a = a.rawvec().as_ptr();
        let b = a.abs();
        let ptr_b = b.rawvec().as_ptr();
        println!("{:}", b);
        println!("{:?}", ptr_a);
        println!("{:?}", ptr_b);

        let a = linspace((-3.0f64, 3.0f64, 6)).into_shape([2, 3]).into_owned();
        let ptr_a = a.rawvec().as_ptr();
        let b = a.abs();
        let ptr_b = b.rawvec().as_ptr();
        println!("{:}", b);
        assert_eq!(ptr_a, ptr_b);
    }

    #[test]
    fn test_hetrogeneous_type() {
        use num::complex::c32;
        let a = linspace((c32(1.0, 2.0), c32(5.0, 6.0), 6)).into_shape([2, 3]).into_owned();
        let b = imag(&a);
        println!("{:}", b);
    }

    #[test]
    fn test_cpuserial() {
        let a = linspace((1.0, 5.0, 5, &DeviceCpuSerial));
        let b = a.sin();
        println!("{:}", b);
    }
}
