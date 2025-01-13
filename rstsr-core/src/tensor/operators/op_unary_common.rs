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

macro_rules! trait_unary_without_func {
    ($op: ident, $op_f: ident, $TensorOpAPI: ident) => {
        pub trait $TensorOpAPI: Sized {
            type Output;
            fn $op_f(self) -> Result<Self::Output>;
            fn $op(self) -> Self::Output {
                self.$op_f().unwrap()
            }
        }
    };
}

#[rustfmt::skip]
#[allow(clippy::wrong_self_convention)]
mod trait_unary {
    use super::*;
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
    trait_unary!(signbit   , signbit_f   , TensorSignBitAPI  );
    trait_unary!(sin       , sin_f       , TensorSinAPI      );
    trait_unary!(sinh      , sinh_f      , TensorSinhAPI     );
    trait_unary!(square    , square_f    , TensorSquareAPI   );
    trait_unary!(sqrt      , sqrt_f      , TensorSqrtAPI     );
    trait_unary!(tan       , tan_f       , TensorTanAPI      );
    trait_unary!(tanh      , tanh_f      , TensorTanhAPI     );
    trait_unary!(trunc     , trunc_f     , TensorTruncAPI    );
    trait_unary!(is_finite , is_finite_f , TensorIsFiniteAPI );
    trait_unary!(is_inf    , is_inf_f    , TensorIsInfAPI    );
    trait_unary!(is_nan    , is_nan_f    , TensorIsNanAPI    );

    trait_unary_without_func!(abs  , abs_f  , TensorRealAbsAPI     );
    trait_unary_without_func!(real , real_f , TensorRealRealAPI    );
    trait_unary_without_func!(imag , imag_f , TensorRealImagAPI    );
    trait_unary_without_func!(sign , sign_f , TensorRealSignAPI    );
    trait_unary_without_func!(abs  , abs_f  , TensorComplexAbsAPI  );
    trait_unary_without_func!(real , real_f , TensorComplexRealAPI );
    trait_unary_without_func!(imag , imag_f , TensorComplexImagAPI );
    trait_unary_without_func!(sign , sign_f , TensorComplexSignAPI );
}

pub use trait_unary::*;

/* #endregion */

/* #region impl tensor unary common */

macro_rules! impl_tensor_unary_common {
    ($op_f: ident, $TensorOpAPI: ident, $DeviceOpAPI: ident) => {
        // any types allowed
        impl<R, T, B, D> $TensorOpAPI for &TensorAny<R, T, B, D>
        where
            D: DimAPI,
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<T, D> + DeviceCreationAnyAPI<B::TOut>,
        {
            type Output = Tensor<B::TOut, B, D>;
            fn $op_f(self) -> Result<Self::Output> {
                let lb = self.layout();
                // generate empty output tensor
                let device = self.device();
                let la = layout_for_array_copy(lb, TensorIterOrder::K)?;
                let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
                // compute and return
                device.op_muta_refb(storage_a.raw_mut(), &la, self.raw(), lb)?;
                return Tensor::new_f(storage_a, la);
            }
        }

        // any types allowed
        impl<'l, T, B, D> $TensorOpAPI for TensorView<'l, T, B, D>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<T, D> + DeviceCreationAnyAPI<B::TOut>,
        {
            type Output = Tensor<B::TOut, B, D>;
            fn $op_f(self) -> Result<Self::Output> {
                $TensorOpAPI::$op_f(&self)
            }
        }

        // same types allowed
        impl<T, B, D> $TensorOpAPI for Tensor<T, B, D>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceOpAPI<T, D, TOut = T> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, B, D>;
            fn $op_f(mut self) -> Result<Self::Output> {
                let layout = self.layout().clone();
                let device = self.device().clone();
                // generate empty output tensor
                device.op_muta(self.raw_mut(), &layout)?;
                return Ok(self);
            }
        }
    };
}

#[rustfmt::skip]
#[allow(clippy::wrong_self_convention)]
mod impl_tensor_unary_common {
    use super::*;
    impl_tensor_unary_common!(acos_f      , TensorAcosAPI     , DeviceAcosAPI     );
    impl_tensor_unary_common!(acosh_f     , TensorAcoshAPI    , DeviceAcoshAPI    );
    impl_tensor_unary_common!(asin_f      , TensorAsinAPI     , DeviceAsinAPI     );
    impl_tensor_unary_common!(asinh_f     , TensorAsinhAPI    , DeviceAsinhAPI    );
    impl_tensor_unary_common!(atan_f      , TensorAtanAPI     , DeviceAtanAPI     );
    impl_tensor_unary_common!(atanh_f     , TensorAtanhAPI    , DeviceAtanhAPI    );
    impl_tensor_unary_common!(ceil_f      , TensorCeilAPI     , DeviceCeilAPI     );
    impl_tensor_unary_common!(conj_f      , TensorConjAPI     , DeviceConjAPI     );
    impl_tensor_unary_common!(cos_f       , TensorCosAPI      , DeviceCosAPI      );
    impl_tensor_unary_common!(cosh_f      , TensorCoshAPI     , DeviceCoshAPI     );
    impl_tensor_unary_common!(exp_f       , TensorExpAPI      , DeviceExpAPI      );
    impl_tensor_unary_common!(expm1_f     , TensorExpm1API    , DeviceExpm1API    );
    impl_tensor_unary_common!(floor_f     , TensorFloorAPI    , DeviceFloorAPI    );
    impl_tensor_unary_common!(inv_f       , TensorInvAPI      , DeviceInvAPI      );
    impl_tensor_unary_common!(is_finite_f , TensorIsFiniteAPI , DeviceIsFiniteAPI );
    impl_tensor_unary_common!(is_inf_f    , TensorIsInfAPI    , DeviceIsInfAPI    );
    impl_tensor_unary_common!(is_nan_f    , TensorIsNanAPI    , DeviceIsNanAPI    );
    impl_tensor_unary_common!(log_f       , TensorLogAPI      , DeviceLogAPI      );
    impl_tensor_unary_common!(log1p_f     , TensorLog1pAPI    , DeviceLog1pAPI    );
    impl_tensor_unary_common!(log2_f      , TensorLog2API     , DeviceLog2API     );
    impl_tensor_unary_common!(log10_f     , TensorLog10API    , DeviceLog10API    );
    impl_tensor_unary_common!(round_f     , TensorRoundAPI    , DeviceRoundAPI    );
    impl_tensor_unary_common!(signbit_f   , TensorSignBitAPI  , DeviceSignBitAPI  );
    impl_tensor_unary_common!(sin_f       , TensorSinAPI      , DeviceSinAPI      );
    impl_tensor_unary_common!(sinh_f      , TensorSinhAPI     , DeviceSinhAPI     );
    impl_tensor_unary_common!(square_f    , TensorSquareAPI   , DeviceSquareAPI   );
    impl_tensor_unary_common!(sqrt_f      , TensorSqrtAPI     , DeviceSqrtAPI     );
    impl_tensor_unary_common!(tan_f       , TensorTanAPI      , DeviceTanAPI      );
    impl_tensor_unary_common!(tanh_f      , TensorTanhAPI     , DeviceTanhAPI     );
    impl_tensor_unary_common!(trunc_f     , TensorTruncAPI    , DeviceTruncAPI    );

    impl_tensor_unary_common!(abs_f       , TensorRealAbsAPI     , DeviceRealAbsAPI     );
    impl_tensor_unary_common!(imag_f      , TensorRealImagAPI    , DeviceRealImagAPI    );
    impl_tensor_unary_common!(real_f      , TensorRealRealAPI    , DeviceRealRealAPI    );
    impl_tensor_unary_common!(sign_f      , TensorRealSignAPI    , DeviceRealSignAPI    );
    impl_tensor_unary_common!(abs_f       , TensorComplexAbsAPI  , DeviceComplexAbsAPI  );
    impl_tensor_unary_common!(imag_f      , TensorComplexImagAPI , DeviceComplexImagAPI );
    impl_tensor_unary_common!(real_f      , TensorComplexRealAPI , DeviceComplexRealAPI );
    impl_tensor_unary_common!(sign_f      , TensorComplexSignAPI , DeviceComplexSignAPI );
}

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

        let ptr_a = a.raw().as_ptr();
        let b = a.sin();
        let ptr_b = b.raw().as_ptr();
        assert_eq!(ptr_a, ptr_b);
    }

    #[test]
    fn test_sign() {
        use num::complex::c64;
        let a = linspace((c64(1.0, 2.0), c64(5.0, 6.0), 6)).into_shape([2, 3]);
        let b = (&a).sign();
        let vec_b = b.reshape([6]).to_vec();
        let b_abs_sum = vec_b.iter().map(|x| x.norm()).sum::<f64>();
        println!("{:}", b);
        assert!(b_abs_sum - 6.0 < 1e-6);
    }

    #[test]
    fn test_abs() {
        use num::complex::c32;
        let a = linspace((c32(1.0, 2.0), c32(5.0, 6.0), 6)).into_shape([2, 3]);
        let ptr_a = a.raw().as_ptr();
        let b = a.abs();
        let ptr_b = b.raw().as_ptr();
        println!("{:}", b);
        println!("{:?}", ptr_a);
        println!("{:?}", ptr_b);

        let a = linspace((-3.0f64, 3.0f64, 6)).into_shape([2, 3]);
        let ptr_a = a.raw().as_ptr();
        let b = a.abs();
        let ptr_b = b.raw().as_ptr();
        println!("{:}", b);
        assert_eq!(ptr_a, ptr_b);
    }

    #[test]
    fn test_hetrogeneous_type() {
        use num::complex::c32;
        let a = linspace((c32(1.0, 2.0), c32(5.0, 6.0), 6)).into_shape([2, 3]);
        let b = (&a).imag();
        println!("{:}", b);
    }

    #[test]
    fn test_cpuserial() {
        let a = linspace((1.0, 5.0, 5, &DeviceCpuSerial));
        let b = a.sin();
        println!("{:}", b);
    }
}
