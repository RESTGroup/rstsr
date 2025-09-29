use crate::prelude_dev::*;

/* Structure of implementation

Exception functions:
- floor_divide: integer and float are different, so we need to implement two functions
- pow: different input types occurs

*/

/* #region tensor traits */

#[duplicate_item(
    op              op_f              TensorOpAPI           ;
   [atan2        ] [atan2_f        ] [TensorATan2API       ];
   [copysign     ] [copysign_f     ] [TensorCopySignAPI    ];
   [equal        ] [equal_f        ] [TensorEqualAPI       ];
   [floor_divide ] [floor_divide_f ] [TensorFloorDivideAPI ];
   [greater      ] [greater_f      ] [TensorGreaterAPI     ];
   [greater_equal] [greater_equal_f] [TensorGreaterEqualAPI];
   [hypot        ] [hypot_f        ] [TensorHypotAPI       ];
   [less         ] [less_f         ] [TensorLessAPI        ];
   [less_equal   ] [less_equal_f   ] [TensorLessEqualAPI   ];
   [log_add_exp  ] [log_add_exp_f  ] [TensorLogAddExpAPI   ];
   [maximum      ] [maximum_f      ] [TensorMaximumAPI     ];
   [minimum      ] [minimum_f      ] [TensorMinimumAPI     ];
   [not_equal    ] [not_equal_f    ] [TensorNotEqualAPI    ];
   [pow          ] [pow_f          ] [TensorPowAPI         ];
   [nextafter    ] [nextafter_f    ] [TensorNextAfterAPI   ];
)]
pub trait TensorOpAPI<TRB> {
    type Output;
    fn op_f(self, b: TRB) -> Result<Self::Output>;
    fn op(self, b: TRB) -> Self::Output
    where
        Self: Sized,
    {
        self.op_f(b).unwrap()
    }
}

#[duplicate_item(
    op_f              TensorOpAPI             DeviceOpAPI           ;
   [atan2_f        ] [TensorATan2API       ] [DeviceATan2API       ];
   [copysign_f     ] [TensorCopySignAPI    ] [DeviceCopySignAPI    ];
   [equal_f        ] [TensorEqualAPI       ] [DeviceEqualAPI       ];
   [floor_divide_f ] [TensorFloorDivideAPI ] [DeviceFloorDivideAPI ];
   [greater_f      ] [TensorGreaterAPI     ] [DeviceGreaterAPI     ];
   [greater_equal_f] [TensorGreaterEqualAPI] [DeviceGreaterEqualAPI];
   [hypot_f        ] [TensorHypotAPI       ] [DeviceHypotAPI       ];
   [less_f         ] [TensorLessAPI        ] [DeviceLessAPI        ];
   [less_equal_f   ] [TensorLessEqualAPI   ] [DeviceLessEqualAPI   ];
   [log_add_exp_f  ] [TensorLogAddExpAPI   ] [DeviceLogAddExpAPI   ];
   [maximum_f      ] [TensorMaximumAPI     ] [DeviceMaximumAPI     ];
   [minimum_f      ] [TensorMinimumAPI     ] [DeviceMinimumAPI     ];
   [not_equal_f    ] [TensorNotEqualAPI    ] [DeviceNotEqualAPI    ];
   [pow_f          ] [TensorPowAPI         ] [DevicePowAPI         ];
   [nextafter_f    ] [TensorNextAfterAPI   ] [DeviceNextAfterAPI   ];
)]
mod impl_trait_binary {
    use super::*;

    impl<RA, TA, DA, RB, TB, DB, B> TensorOpAPI<&TensorAny<RB, TB, B, DB>> for &TensorAny<RA, TA, B, DA>
    where
        RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        DA: DimAPI + DimMaxAPI<DB>,
        DB: DimAPI,
        DA::Max: DimAPI,
        B: DeviceOpAPI<TA, TB, DA::Max>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
    {
        type Output = Tensor<B::TOut, B, DA::Max>;

        fn op_f(self, b: &TensorAny<RB, TB, B, DB>) -> Result<Self::Output> {
            // check device
            rstsr_assert!(self.device().same_device(b.device()), DeviceMismatch)?;

            // check and broadcast layout
            let la = self.layout();
            let lb = b.layout();
            let default_order = self.device().default_order();
            let (la_b, lb_b) = broadcast_layout(la, lb, default_order)?;
            let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::default())?;
            let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::default())?;
            let lc = if lc_from_a == lc_from_b {
                lc_from_a
            } else {
                match self.device().default_order() {
                    RowMajor => la_b.shape().c(),
                    ColMajor => la_b.shape().f(),
                }
            };

            // perform operation and return
            let device = self.device();
            let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
            device.op_mutc_refa_refb(storage_c.raw_mut(), &lc, self.raw(), &la_b, b.raw(), &lb_b)?;
            let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
            Tensor::new_f(storage_c, lc)
        }
    }

    #[duplicate_item(
        ImplType                                                             TrA                         TrB                       ;
       [TA, DA, TB, DB, B, R: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>] [&TensorAny<R, TA, B, DA> ] [TensorView<'_, TB, B, DB>];
       [TA, DA, TB, DB, B, R: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>] [TensorView<'_, TA, B, DA>] [&TensorAny<R, TB, B, DB> ];
       [TA, DA, TB, DB, B                                                 ] [TensorView<'_, TA, B, DA>] [TensorView<'_, TB, B, DB>];
    )]
    impl<ImplType> TensorOpAPI<TrB> for TrA
    where
        DA: DimAPI + DimMaxAPI<DB>,
        DB: DimAPI,
        DA::Max: DimAPI,
        B: DeviceOpAPI<TA, TB, DA::Max>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
    {
        type Output = Tensor<B::TOut, B, DA::Max>;

        fn op_f(self, b: TrB) -> Result<Self::Output> {
            TensorOpAPI::op_f(&self.view(), &b.view())
        }
    }

    impl<RA, TA, DA, TB, B> TensorOpAPI<TB> for &TensorAny<RA, TA, B, DA>
    where
        RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
        DA: DimAPI,
        B: DeviceOpAPI<TA, TB, DA>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
        TB: num::Num,
    {
        type Output = Tensor<B::TOut, B, DA>;

        fn op_f(self, b: TB) -> Result<Self::Output> {
            // check and broadcast layout
            let la = self.layout();
            let lc = layout_for_array_copy(la, TensorIterOrder::default())?;

            // perform operation and return
            let device = self.device();
            let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
            device.op_mutc_refa_numb(storage_c.raw_mut(), &lc, self.raw(), la, b)?;
            let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
            Tensor::new_f(storage_c, lc)
        }
    }

    impl<TA, DA, TB, B> TensorOpAPI<TB> for TensorView<'_, TA, B, DA>
    where
        DA: DimAPI,
        B: DeviceOpAPI<TA, TB, DA>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
        TB: num::Num,
    {
        type Output = Tensor<B::TOut, B, DA>;

        fn op_f(self, b: TB) -> Result<Self::Output> {
            (&self).op_f(b)
        }
    }

    impl<RB, TA, DB, TB, B> TensorOpAPI<&TensorAny<RB, TB, B, DB>> for TA
    where
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        DB: DimAPI,
        B: DeviceOpAPI<TA, TB, DB>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
        TA: num::Num,
    {
        type Output = Tensor<B::TOut, B, DB>;

        fn op_f(self, b: &TensorAny<RB, TB, B, DB>) -> Result<Self::Output> {
            // check and broadcast layout
            let lb = b.layout();
            let lc = layout_for_array_copy(lb, TensorIterOrder::default())?;

            // perform operation and return
            let device = b.device();
            let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
            device.op_mutc_numa_refb(storage_c.raw_mut(), &lc, self, b.raw(), lb)?;
            let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
            Tensor::new_f(storage_c, lc)
        }
    }

    impl<TA, DB, TB, B> TensorOpAPI<TensorView<'_, TB, B, DB>> for TA
    where
        DB: DimAPI,
        B: DeviceOpAPI<TA, TB, DB>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
        TA: num::Num,
    {
        type Output = Tensor<B::TOut, B, DB>;

        fn op_f(self, b: TensorView<'_, TB, B, DB>) -> Result<Self::Output> {
            TensorOpAPI::op_f(self, &b.view())
        }
    }
}

/* #endregion */

/* #region function impl */

macro_rules! func_binary {
    ($op: ident, $op_f: ident, $TensorOpAPI: ident, $DeviceOpAPI: ident, $($op2: ident, $op2_f: ident),*) => {
        pub fn $op_f<TRA, TRB>(a: TRA, b: TRB) -> Result<TRA::Output>
        where
            TRA: $TensorOpAPI<TRB>,
        {
            a.$op_f(b)
        }

        pub fn $op<TRA, TRB>(a: TRA, b: TRB) -> TRA::Output
        where
            TRA: $TensorOpAPI<TRB>,
        {
            a.$op(b)
        }

        $(
            pub fn $op2_f<TRA, TRB>(a: TRA, b: TRB) -> Result<TRA::Output>
            where
                TRA: $TensorOpAPI<TRB>,
            {
                a.$op_f(b)
            }

            pub fn $op2<TRA, TRB>(a: TRA, b: TRB) -> TRA::Output
            where
                TRA: $TensorOpAPI<TRB>,
            {
                a.$op(b)
            }
        )*
    };
}

#[rustfmt::skip]
mod func_binary {
    use super::*;
    func_binary!(atan2         , atan2_f           , TensorATan2API            , DeviceATan2API            ,);
    func_binary!(copysign      , copysign_f        , TensorCopySignAPI         , DeviceCopySignAPI         ,);
    func_binary!(floor_divide  , floor_divide_f    , TensorFloorDivideAPI      , DeviceFloorDivideAPI      ,);
    func_binary!(hypot         , hypot_f           , TensorHypotAPI            , DeviceHypotAPI            ,);
    func_binary!(log_add_exp   , log_add_exp_f     , TensorLogAddExpAPI        , DeviceLogAddExpAPI        ,);
    func_binary!(pow           , pow_f             , TensorPowAPI              , DevicePowAPI              ,);
    func_binary!(maximum       , maximum_f         , TensorMaximumAPI          , DeviceMaximumAPI          , max, max_f);
    func_binary!(minimum       , minimum_f         , TensorMinimumAPI          , DeviceMinimumAPI          , min, min_f);
    func_binary!(equal         , equal_f           , TensorEqualAPI            , DeviceEqualAPI            , eq, eq_f, equal_than      , equal_than_f      );
    func_binary!(less          , less_f            , TensorLessAPI             , DeviceLessAPI             , lt, lt_f, less_than       , less_than_f       );
    func_binary!(greater       , greater_f         , TensorGreaterAPI          , DeviceGreaterAPI          , gt, gt_f, greater_than    , greater_than_f    );
    func_binary!(less_equal    , less_equal_f      , TensorLessEqualAPI        , DeviceLessEqualAPI        , le, le_f, less_equal_to   , less_equal_to_f   );
    func_binary!(greater_equal , greater_equal_f   , TensorGreaterEqualAPI     , DeviceGreaterEqualAPI     , ge, ge_f, greater_equal_to, greater_equal_to_f);
    func_binary!(not_equal     , not_equal_f       , TensorNotEqualAPI         , DeviceNotEqualAPI         , ne, ne_f, not_equal_to    , not_equal_to_f    );
    func_binary!(nextafter     , nextafter_f       , TensorNextAfterAPI        , DeviceNextAfterAPI        ,);
}

pub use func_binary::*;

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pow() {
        #[cfg(not(feature = "col_major"))]
        {
            let a = arange(6u32).into_shape([2, 3]);
            let b = arange(3u32);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1, 1, 4, 1, 4, 25]);

            let a = arange(6.0).into_shape([2, 3]);

            let b = arange(3.0);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1.0, 1.0, 4.0, 1.0, 4.0, 25.0]);

            let b = arange(3);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1.0, 1.0, 4.0, 1.0, 4.0, 25.0]);
        }
        #[cfg(feature = "col_major")]
        {
            let a = arange(6u32).into_shape([3, 2]);
            let b = arange(3u32);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1, 1, 4, 1, 4, 25]);

            let a = arange(6.0).into_shape([3, 2]);

            let b = arange(3.0);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1.0, 1.0, 4.0, 1.0, 4.0, 25.0]);

            let b = arange(3);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1.0, 1.0, 4.0, 1.0, 4.0, 25.0]);
        }
    }

    #[test]
    fn test_floor_divide() {
        #[cfg(not(feature = "col_major"))]
        {
            let a = arange(6u32).into_shape([2, 3]);
            let b = asarray(vec![1, 2, 2]);
            let c = a.floor_divide(&b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![0, 0, 1, 3, 2, 2]);

            let a = arange(6.0).into_shape([2, 3]);

            let b = asarray(vec![1.0, 2.0, 2.0]);
            let c = a.floor_divide(&b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![0.0, 0.0, 1.0, 3.0, 2.0, 2.0]);

            let b = asarray(vec![0.0, 2.0, 2.0]);
            let c = a.floor_divide_f(&b);
            println!("{c:?}");
        }
        #[cfg(feature = "col_major")]
        {
            // [3, 2] + [3]
            let a = arange(6u32).into_shape([3, 2]);
            let b = asarray(vec![1, 2, 2]);
            let c = a.floor_divide(&b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![0, 0, 1, 3, 2, 2]);

            let a = arange(6.0).into_shape([3, 2]);

            let b = asarray(vec![1.0, 2.0, 2.0]);
            let c = a.floor_divide(&b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![0.0, 0.0, 1.0, 3.0, 2.0, 2.0]);

            let b = asarray(vec![0.0, 2.0, 2.0]);
            let c = a.floor_divide_f(&b);
            println!("{c:?}");
        }
    }

    #[test]
    fn test_ge_gt() {
        let a = asarray(vec![1., 2., 3., 4., 5., 6.]);
        let b = asarray(vec![1., 3., 2., 5., 5., 2.]);

        let c = gt(a.view(), &b);
        assert_eq!(c.raw(), &[false, false, true, false, false, true]);
        let c = ge(a.view(), &b);
        assert_eq!(c.raw(), &[true, false, true, false, true, true]);

        let c_sum = c.sum();
        assert_eq!(c_sum, 4);
    }

    #[test]
    fn test_refa_numb() {
        let a = asarray(vec![1., 3., 2., 5., 5., 2.]);
        let b = a.greater_equal(3.0);
        assert_eq!(b.raw(), &[false, true, false, true, true, false]);
        let b = a.pow(2);
        assert_eq!(b.raw(), &[1.0, 9.0, 4.0, 25.0, 25.0, 4.0]);
        let b = 2.0.pow(a.view());
        assert_eq!(b.raw(), &[2.0, 8.0, 4.0, 32.0, 32.0, 4.0]);
    }
}
