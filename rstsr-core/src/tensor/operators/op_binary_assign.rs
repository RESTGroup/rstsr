use crate::prelude_dev::*;
use core::ops::*;

macro_rules! trait_binary_assign {
    ($op: ident, $TensorOpAPI: ident) => {
        pub trait $TensorOpAPI<TRB> {
            fn $op(a: Self, b: TRB) -> Result<()>;
        }

        pub fn $op<TRA, TRB>(a: TRA, b: TRB) -> Result<()>
        where
            TRA: $TensorOpAPI<TRB>,
        {
            TRA::$op(a, b)
        }
    };
}

#[rustfmt::skip]
mod trait_binary_assign {
    use super::*;
    trait_binary_assign!(add_assign   , TensorAddAssignAPI   );
    trait_binary_assign!(sub_assign   , TensorSubAssignAPI   );
    trait_binary_assign!(mul_assign   , TensorMulAssignAPI   );
    trait_binary_assign!(div_assign   , TensorDivAssignAPI   );
    trait_binary_assign!(rem_assign   , TensorRemAssignAPI   );
    trait_binary_assign!(bitor_assign , TensorBitOrAssignAPI );
    trait_binary_assign!(bitand_assign, TensorBitAndAssignAPI);
    trait_binary_assign!(bitxor_assign, TensorBitXorAssignAPI);
    trait_binary_assign!(shl_assign   , TensorShlAssignAPI   );
    trait_binary_assign!(shr_assign   , TensorShrAssignAPI   );
}
pub use trait_binary_assign::*;

macro_rules! impl_assign_ops {
    ($op: ident, $TensorOpAPI: ident, $Op: ident) => {
        impl<TRB, RA, TA, DA, B> $Op<TRB> for TensorBase<RA, DA>
        where
            RA: DataMutAPI<Data = Storage<TA, B>>,
            DA: DimAPI,
            B: DeviceAPI<TA>,
            for<'a> &'a mut Self: $TensorOpAPI<TRB>,
        {
            fn $op(&mut self, b: TRB) -> () {
                $TensorOpAPI::$op(self, b).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
mod impl_assign_ops {
    use super::*;
    impl_assign_ops!(add_assign   , TensorAddAssignAPI   , AddAssign   );
    impl_assign_ops!(sub_assign   , TensorSubAssignAPI   , SubAssign   );
    impl_assign_ops!(mul_assign   , TensorMulAssignAPI   , MulAssign   );
    impl_assign_ops!(div_assign   , TensorDivAssignAPI   , DivAssign   );
    impl_assign_ops!(rem_assign   , TensorRemAssignAPI   , RemAssign   );
    impl_assign_ops!(bitor_assign , TensorBitOrAssignAPI , BitOrAssign );
    impl_assign_ops!(bitand_assign, TensorBitAndAssignAPI, BitAndAssign);
    impl_assign_ops!(bitxor_assign, TensorBitXorAssignAPI, BitXorAssign);
    impl_assign_ops!(shl_assign   , TensorShlAssignAPI   , ShlAssign   );
    impl_assign_ops!(shr_assign   , TensorShrAssignAPI   , ShrAssign   );
}

macro_rules! impl_binary_assign {
    ($op: ident, $TensorOpAPI: ident, $Op: ident, $DeviceOpAPI: ident) => {
        impl<RA, RB, TA, TB, DA, DB, B> $TensorOpAPI<&TensorBase<RB, DB>>
            for &mut TensorBase<RA, DA>
        where
            // tensor types
            RA: DataMutAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DA>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, DA>,
        {
            fn $op(a: Self, b: &TensorBase<RB, DB>) -> Result<()> {
                rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
                let la = a.layout();
                let lb = b.layout();
                // check layout broadcast
                let (la_b, lb_b) = broadcast_layout_to_first(la, lb)?;
                rstsr_assert_eq!(la_b, *la, InvalidLayout)?;
                // op provided by device
                let device = a.device().clone();
                let storage_a = a.data_mut().storage_mut();
                let storage_b = b.data().storage();
                device.op_muta_refb(storage_a, &la_b, storage_b, &lb_b)
            }
        }

        impl<RA, RB, TA, TB, DA, DB, B> $TensorOpAPI<TensorBase<RB, DB>> for &mut TensorBase<RA, DA>
        where
            // tensor types
            RA: DataMutAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DA>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, DA>,
        {
            fn $op(a: Self, b: TensorBase<RB, DB>) -> Result<()> {
                $TensorOpAPI::$op(a, &b)
            }
        }

        impl<RA, TA, TB, D, B> $TensorOpAPI<TB> for &mut TensorBase<RA, D>
        where
            // tensor types
            RA: DataMutAPI<Data = Storage<TA, B>>,
            // data constraints
            D: DimAPI,
            B: DeviceAPI<TA>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, D>,
            // this constraint prohibits confliting impl to TensorBase<RB, D>
            TB: num::Num,
        {
            fn $op(a: Self, b: TB) -> Result<()> {
                let la = a.layout().clone();
                let device = a.device().clone();
                let storage_a = a.data_mut().storage_mut();
                device.op_muta_numb(storage_a, &la, b)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_binary_assign {
    use super::*;
    impl_binary_assign!(add_assign   , TensorAddAssignAPI   , AddAssign   , DeviceAddAssignAPI   );
    impl_binary_assign!(sub_assign   , TensorSubAssignAPI   , SubAssign   , DeviceSubAssignAPI   );
    impl_binary_assign!(mul_assign   , TensorMulAssignAPI   , MulAssign   , DeviceMulAssignAPI   );
    impl_binary_assign!(div_assign   , TensorDivAssignAPI   , DivAssign   , DeviceDivAssignAPI   );
    impl_binary_assign!(rem_assign   , TensorRemAssignAPI   , RemAssign   , DeviceRemAssignAPI   );
    impl_binary_assign!(bitor_assign , TensorBitOrAssignAPI , BitOrAssign , DeviceBitOrAssignAPI );
    impl_binary_assign!(bitand_assign, TensorBitAndAssignAPI, BitAndAssign, DeviceBitAndAssignAPI);
    impl_binary_assign!(bitxor_assign, TensorBitXorAssignAPI, BitXorAssign, DeviceBitXorAssignAPI);
    impl_binary_assign!(shl_assign   , TensorShlAssignAPI   , ShlAssign   , DeviceShlAssignAPI   );
    impl_binary_assign!(shr_assign   , TensorShrAssignAPI   , ShrAssign   , DeviceShrAssignAPI   );
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[allow(clippy::deref_addrof)]
    fn test_add_assign() {
        // contiguous
        let mut c = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        c += &b;
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        c += b;
        let c_ref = vec![5., 10., 15., 20., 25.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [2, 3] + [3]
        let c = Tensor::linspace_cpu(1.0, 6.0, 6);
        let mut c = c.into_shape_assume_contig_f::<Ix2>([2, 3]).unwrap();
        let b = Tensor::linspace_cpu(2.0, 6.0, 3);
        // let mut c_mut = c.view_mut();
        // c_mut += &b;
        *&mut c.view_mut() += &b;
        let c_ref = vec![3., 6., 9., 6., 9., 12.].into();
        assert!(allclose_f64(&c, &c_ref));

        // scalar
        c *= 2.0;
        let c_ref = vec![6., 12., 18., 12., 18., 24.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    #[allow(clippy::deref_addrof)]
    fn test_sub_assign() {
        // contiguous
        let mut c = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        c -= &b;
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [2, 3] + [3]
        let c = Tensor::linspace_cpu(1.0, 6.0, 6);
        let mut c = c.into_shape_assume_contig_f::<Ix2>([2, 3]).unwrap();
        let b = Tensor::linspace_cpu(2.0, 6.0, 3);
        // let mut c_mut = c.view_mut();
        // c_mut += &b;
        *&mut c.view_mut() -= &b;
        let c_ref = vec![-1., -2., -3., 2., 1., 0.].into();
        assert!(allclose_f64(&c, &c_ref));
    }
}
