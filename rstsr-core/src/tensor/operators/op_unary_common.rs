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
        pub trait $TensorOpAPI {
            type Output;
            fn $op_f(self) -> Result<Self::Output>;
            fn $op(self) -> Self::Output
            where
                Self: Sized,
            {
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

    trait_unary!(abs  , abs_f  , TensorAbsAPI  );
    trait_unary!(real , real_f , TensorRealAPI );
    trait_unary!(imag , imag_f , TensorImagAPI );
    trait_unary!(sign , sign_f , TensorSignAPI );
}

pub use trait_unary::*;

/* #endregion */

/* #region impl tensor unary common */

#[duplicate_item(
    op_f          TensorOpAPI         DeviceOpAPI       ;
   [acos_f     ] [TensorAcosAPI    ] [DeviceAcosAPI    ];
   [acosh_f    ] [TensorAcoshAPI   ] [DeviceAcoshAPI   ];
   [asin_f     ] [TensorAsinAPI    ] [DeviceAsinAPI    ];
   [asinh_f    ] [TensorAsinhAPI   ] [DeviceAsinhAPI   ];
   [atan_f     ] [TensorAtanAPI    ] [DeviceAtanAPI    ];
   [atanh_f    ] [TensorAtanhAPI   ] [DeviceAtanhAPI   ];
   [ceil_f     ] [TensorCeilAPI    ] [DeviceCeilAPI    ];
   [conj_f     ] [TensorConjAPI    ] [DeviceConjAPI    ];
   [cos_f      ] [TensorCosAPI     ] [DeviceCosAPI     ];
   [cosh_f     ] [TensorCoshAPI    ] [DeviceCoshAPI    ];
   [exp_f      ] [TensorExpAPI     ] [DeviceExpAPI     ];
   [expm1_f    ] [TensorExpm1API   ] [DeviceExpm1API   ];
   [floor_f    ] [TensorFloorAPI   ] [DeviceFloorAPI   ];
   [inv_f      ] [TensorInvAPI     ] [DeviceInvAPI     ];
   [is_finite_f] [TensorIsFiniteAPI] [DeviceIsFiniteAPI];
   [is_inf_f   ] [TensorIsInfAPI   ] [DeviceIsInfAPI   ];
   [is_nan_f   ] [TensorIsNanAPI   ] [DeviceIsNanAPI   ];
   [log_f      ] [TensorLogAPI     ] [DeviceLogAPI     ];
   [log1p_f    ] [TensorLog1pAPI   ] [DeviceLog1pAPI   ];
   [log2_f     ] [TensorLog2API    ] [DeviceLog2API    ];
   [log10_f    ] [TensorLog10API   ] [DeviceLog10API   ];
   [round_f    ] [TensorRoundAPI   ] [DeviceRoundAPI   ];
   [signbit_f  ] [TensorSignBitAPI ] [DeviceSignBitAPI ];
   [sin_f      ] [TensorSinAPI     ] [DeviceSinAPI     ];
   [sinh_f     ] [TensorSinhAPI    ] [DeviceSinhAPI    ];
   [square_f   ] [TensorSquareAPI  ] [DeviceSquareAPI  ];
   [sqrt_f     ] [TensorSqrtAPI    ] [DeviceSqrtAPI    ];
   [tan_f      ] [TensorTanAPI     ] [DeviceTanAPI     ];
   [tanh_f     ] [TensorTanhAPI    ] [DeviceTanhAPI    ];
   [trunc_f    ] [TensorTruncAPI   ] [DeviceTruncAPI   ];
   [abs_f      ] [TensorAbsAPI     ] [DeviceAbsAPI     ];
   [imag_f     ] [TensorImagAPI    ] [DeviceImagAPI    ];
   [real_f     ] [TensorRealAPI    ] [DeviceRealAPI    ];
   [sign_f     ] [TensorSignAPI    ] [DeviceSignAPI    ];
)]
mod impl_tensor_unary_common {
    use super::*;

    // any types allowed
    impl<R, T, B, D> TensorOpAPI for &TensorAny<R, T, B, D>
    where
        D: DimAPI,
        R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        B: DeviceAPI<T>,
        B: DeviceOpAPI<T, D> + DeviceCreationAnyAPI<B::TOut>,
    {
        type Output = Tensor<B::TOut, B, D>;
        fn op_f(self) -> Result<Self::Output> {
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
    impl<T, B, D> TensorOpAPI for TensorView<'_, T, B, D>
    where
        D: DimAPI,
        B: DeviceAPI<T>,
        B: DeviceOpAPI<T, D> + DeviceCreationAnyAPI<B::TOut>,
    {
        type Output = Tensor<B::TOut, B, D>;
        fn op_f(self) -> Result<Self::Output> {
            TensorOpAPI::op_f(&self)
        }
    }

    // same types allowed
    impl<T, B, D> TensorOpAPI for Tensor<T, B, D>
    where
        D: DimAPI,
        B: DeviceAPI<T>,
        B: DeviceOpAPI<T, D, TOut = T> + DeviceCreationAnyAPI<T>,
    {
        type Output = Tensor<T, B, D>;
        fn op_f(mut self) -> Result<Self::Output> {
            let layout = self.layout().clone();
            let device = self.device().clone();
            // generate empty output tensor
            device.op_muta(self.raw_mut(), &layout)?;
            return Ok(self);
        }
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_same_type() {
        let a = arange(6.0).into_shape([2, 3]).into_owned();
        let b = sin(&a);
        println!("{b:}");
        let b = a.view().sin();
        println!("{b:}");

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
        println!("{b:}");
        assert!(b_abs_sum - 6.0 < 1e-6);
    }

    #[test]
    fn test_abs() {
        use num::complex::c32;
        let a = linspace((c32(1.0, 2.0), c32(5.0, 6.0), 6)).into_shape([2, 3]);
        let ptr_a = a.raw().as_ptr();
        let b = a.abs();
        let ptr_b = b.raw().as_ptr();
        println!("{b:}");
        println!("{ptr_a:?}");
        println!("{ptr_b:?}");
        // for complex case, only abs(&a) is valid
        println!("{a:}");

        let a = linspace((-3.0f64, 3.0f64, 6)).into_shape([2, 3]);
        let ptr_a = a.raw().as_ptr();
        let b = a.abs();
        let ptr_b = b.raw().as_ptr();
        println!("{b:}");
        assert_eq!(ptr_a, ptr_b);
        // for f64 case, `a.abs()` will try to consume variable `a`
        // println!("{:?}", a);
    }

    #[test]
    fn test_hetrogeneous_type() {
        use num::complex::c32;
        let a = linspace((c32(1.0, 2.0), c32(5.0, 6.0), 6)).into_shape([2, 3]);
        let b = (&a).imag();
        println!("{b:}");
    }

    #[test]
    fn test_cpuserial() {
        let a = linspace((1.0, 5.0, 5, &DeviceCpuSerial::default()));
        let b = a.sin();
        println!("{b:}");
    }
}
