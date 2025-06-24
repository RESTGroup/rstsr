use crate::prelude_dev::*;

#[duplicate_item(
    op              op_f              TensorOpAPI           ;
   [add_assign   ] [add_assign_f   ] [TensorAddAssignAPI   ];
   [sub_assign   ] [sub_assign_f   ] [TensorSubAssignAPI   ];
   [mul_assign   ] [mul_assign_f   ] [TensorMulAssignAPI   ];
   [div_assign   ] [div_assign_f   ] [TensorDivAssignAPI   ];
   [rem_assign   ] [rem_assign_f   ] [TensorRemAssignAPI   ];
   [bitor_assign ] [bitor_assign_f ] [TensorBitOrAssignAPI ];
   [bitand_assign] [bitand_assign_f] [TensorBitAndAssignAPI];
   [bitxor_assign] [bitxor_assign_f] [TensorBitXorAssignAPI];
   [shl_assign   ] [shl_assign_f   ] [TensorShlAssignAPI   ];
   [shr_assign   ] [shr_assign_f   ] [TensorShrAssignAPI   ];
)]
pub trait TensorOpAPI<TRB> {
    fn op_f(a: Self, b: TRB) -> Result<()>;
    fn op(a: Self, b: TRB)
    where
        Self: Sized,
    {
        Self::op_f(a, b).unwrap()
    }
}

#[duplicate_item(
    op              op_f              TensorOpAPI           ;
   [add_assign   ] [add_assign_f   ] [TensorAddAssignAPI   ];
   [sub_assign   ] [sub_assign_f   ] [TensorSubAssignAPI   ];
   [mul_assign   ] [mul_assign_f   ] [TensorMulAssignAPI   ];
   [div_assign   ] [div_assign_f   ] [TensorDivAssignAPI   ];
   [rem_assign   ] [rem_assign_f   ] [TensorRemAssignAPI   ];
   [bitor_assign ] [bitor_assign_f ] [TensorBitOrAssignAPI ];
   [bitand_assign] [bitand_assign_f] [TensorBitAndAssignAPI];
   [bitxor_assign] [bitxor_assign_f] [TensorBitXorAssignAPI];
   [shl_assign   ] [shl_assign_f   ] [TensorShlAssignAPI   ];
   [shr_assign   ] [shr_assign_f   ] [TensorShrAssignAPI   ];
)]
pub fn op_f<TRA, TRB>(a: TRA, b: TRB) -> Result<()>
where
    TRA: TensorOpAPI<TRB>,
{
    TRA::op_f(a, b)
}

#[duplicate_item(
    op              op_f              TensorOpAPI           ;
   [add_assign   ] [add_assign_f   ] [TensorAddAssignAPI   ];
   [sub_assign   ] [sub_assign_f   ] [TensorSubAssignAPI   ];
   [mul_assign   ] [mul_assign_f   ] [TensorMulAssignAPI   ];
   [div_assign   ] [div_assign_f   ] [TensorDivAssignAPI   ];
   [rem_assign   ] [rem_assign_f   ] [TensorRemAssignAPI   ];
   [bitor_assign ] [bitor_assign_f ] [TensorBitOrAssignAPI ];
   [bitand_assign] [bitand_assign_f] [TensorBitAndAssignAPI];
   [bitxor_assign] [bitxor_assign_f] [TensorBitXorAssignAPI];
   [shl_assign   ] [shl_assign_f   ] [TensorShlAssignAPI   ];
   [shr_assign   ] [shr_assign_f   ] [TensorShrAssignAPI   ];
)]
pub fn op<TRA, TRB>(a: TRA, b: TRB)
where
    TRA: TensorOpAPI<TRB>,
{
    TRA::op(a, b)
}

#[duplicate_item(
    op              TensorOpAPI             Op           ;
   [add_assign   ] [TensorAddAssignAPI   ] [AddAssign   ];
   [sub_assign   ] [TensorSubAssignAPI   ] [SubAssign   ];
   [mul_assign   ] [TensorMulAssignAPI   ] [MulAssign   ];
   [div_assign   ] [TensorDivAssignAPI   ] [DivAssign   ];
   [rem_assign   ] [TensorRemAssignAPI   ] [RemAssign   ];
   [bitor_assign ] [TensorBitOrAssignAPI ] [BitOrAssign ];
   [bitand_assign] [TensorBitAndAssignAPI] [BitAndAssign];
   [bitxor_assign] [TensorBitXorAssignAPI] [BitXorAssign];
   [shl_assign   ] [TensorShlAssignAPI   ] [ShlAssign   ];
   [shr_assign   ] [TensorShrAssignAPI   ] [ShrAssign   ];
)]
impl<TRB, RA, TA, DA, B> Op<TRB> for TensorAny<RA, TA, B, DA>
where
    RA: DataMutAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    DA: DimAPI,
    B: DeviceAPI<TA>,
    for<'a> &'a mut Self: TensorOpAPI<TRB>,
{
    fn op(&mut self, b: TRB) {
        TensorOpAPI::op(self, b)
    }
}

#[duplicate_item(
    op_f              TensorOpAPI             Op             DeviceOpAPI           ;
   [add_assign_f   ] [TensorAddAssignAPI   ] [AddAssign   ] [DeviceAddAssignAPI   ];
   [sub_assign_f   ] [TensorSubAssignAPI   ] [SubAssign   ] [DeviceSubAssignAPI   ];
   [mul_assign_f   ] [TensorMulAssignAPI   ] [MulAssign   ] [DeviceMulAssignAPI   ];
   [div_assign_f   ] [TensorDivAssignAPI   ] [DivAssign   ] [DeviceDivAssignAPI   ];
   [rem_assign_f   ] [TensorRemAssignAPI   ] [RemAssign   ] [DeviceRemAssignAPI   ];
   [bitor_assign_f ] [TensorBitOrAssignAPI ] [BitOrAssign ] [DeviceBitOrAssignAPI ];
   [bitand_assign_f] [TensorBitAndAssignAPI] [BitAndAssign] [DeviceBitAndAssignAPI];
   [bitxor_assign_f] [TensorBitXorAssignAPI] [BitXorAssign] [DeviceBitXorAssignAPI];
   [shl_assign_f   ] [TensorShlAssignAPI   ] [ShlAssign   ] [DeviceShlAssignAPI   ];
   [shr_assign_f   ] [TensorShrAssignAPI   ] [ShrAssign   ] [DeviceShrAssignAPI   ];
)]
mod impl_binary_assign {
    use super::*;

    #[doc(hidden)]
    impl<RA, RB, TA, TB, DA, DB, B> TensorOpAPI<&TensorAny<RB, TB, B, DB>> for &mut TensorAny<RA, TA, B, DA>
    where
        // tensor types
        RA: DataMutAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        // data constraints
        DA: DimAPI,
        DB: DimAPI,
        B: DeviceAPI<TA> + DeviceAPI<TB>,
        // operation constraints
        TA: Op<TB>,
        B: DeviceOpAPI<TA, TB, DA>,
    {
        fn op_f(a: Self, b: &TensorAny<RB, TB, B, DB>) -> Result<()> {
            rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
            let la = a.layout();
            let lb = b.layout();
            let default_order = a.device().default_order();
            // check layout broadcast
            let (la_b, lb_b) = broadcast_layout_to_first(&la.to_dim::<IxD>()?, &lb.to_dim::<IxD>()?, default_order)?;
            rstsr_assert_eq!(la_b, la.to_dim::<IxD>()?, InvalidLayout)?;
            // op provided by device
            let device = a.device().clone();
            device.op_muta_refb(a.raw_mut(), &la_b.to_dim()?, b.raw(), &lb_b.to_dim()?)
        }
    }

    #[doc(hidden)]
    impl<RA, RB, TA, TB, DA, DB, B> TensorOpAPI<TensorAny<RB, TB, B, DB>> for &mut TensorAny<RA, TA, B, DA>
    where
        // tensor types
        RA: DataMutAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        // data constraints
        DA: DimAPI,
        DB: DimAPI,
        B: DeviceAPI<TA> + DeviceAPI<TB>,
        // broadcast constraints
        DA: DimMaxAPI<DB, Max = DA>,
        // operation constraints
        TA: Op<TB>,
        B: DeviceOpAPI<TA, TB, DA>,
    {
        fn op_f(a: Self, b: TensorAny<RB, TB, B, DB>) -> Result<()> {
            TensorOpAPI::op_f(a, &b)
        }
    }

    #[doc(hidden)]
    impl<RA, TA, TB, D, B> TensorOpAPI<TB> for &mut TensorAny<RA, TA, B, D>
    where
        // tensor types
        RA: DataMutAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
        // data constraints
        D: DimAPI,
        B: DeviceAPI<TA>,
        // operation constraints
        TA: Op<TB>,
        B: DeviceOpAPI<TA, TB, D>,
        // this constraint prohibits confliting impl to TensorBase<RB, D>
        TB: num::Num,
    {
        fn op_f(a: Self, b: TB) -> Result<()> {
            let la = a.layout().clone();
            let device = a.device().clone();
            device.op_muta_numb(a.raw_mut(), &la, b)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add_assign() {
        // contiguous
        let mut c = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        c += &b;
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        c += b;
        let c_ref = vec![5., 10., 15., 20., 25.].into();
        assert!(allclose_f64(&c, &c_ref));

        #[cfg(not(feature = "col_major"))]
        {
            // broadcast
            // [2, 3] + [3]
            let c = linspace((1.0, 6.0, 6));
            let mut c = c.into_shape_assume_contig([2, 3]);
            let b = linspace((2.0, 6.0, 3));
            *&mut c.view_mut() += &b;
            let c_ref = vec![3., 6., 9., 6., 9., 12.].into();
            assert!(allclose_f64(&c, &c_ref));

            // scalar
            c *= 2.0;
            let c_ref = vec![6., 12., 18., 12., 18., 24.].into();
            assert!(allclose_f64(&c, &c_ref));
        }
        #[cfg(feature = "col_major")]
        {
            // broadcast
            // [3, 2] + [3]
            let c = linspace((1.0, 6.0, 6));
            let mut c = c.into_shape_assume_contig([3, 2]);
            let b = linspace((2.0, 6.0, 3));
            *&mut c.view_mut() += &b;
            let c_ref = vec![3., 6., 9., 6., 9., 12.];
            assert!(allclose_f64(&c.raw().into(), &c_ref.into()));

            // scalar
            c *= 2.0;
            let c_ref = vec![6., 12., 18., 12., 18., 24.];
            assert!(allclose_f64(&c.raw().into(), &c_ref.into()));
        }
    }

    #[test]
    fn test_sub_assign() {
        // contiguous
        let mut c = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        c -= &b;
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));

        #[cfg(not(feature = "col_major"))]
        {
            // broadcast
            // [2, 3] + [3]
            let c = linspace((1.0, 6.0, 6));
            let mut c = c.into_shape_assume_contig([2, 3]);
            let b = linspace((2.0, 6.0, 3));
            // let mut c_mut = c.view_mut();
            // c_mut += &b;
            *&mut c.view_mut() -= &b;
            let c_ref = vec![-1., -2., -3., 2., 1., 0.].into();
            assert!(allclose_f64(&c, &c_ref));
        }
        #[cfg(feature = "col_major")]
        {
            // broadcast
            // [3, 2] + [3]
            let c = linspace((1.0, 6.0, 6));
            let mut c = c.into_shape_assume_contig([3, 2]);
            let b = linspace((2.0, 6.0, 3));
            // let mut c_mut = c.view_mut();
            // c_mut += &b;
            *&mut c.view_mut() -= &b;
            let c_ref = vec![-1., -2., -3., 2., 1., 0.];
            assert!(allclose_f64(&c.raw().into(), &c_ref.into()));
        }
    }
}
