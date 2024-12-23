use crate::prelude_dev::*;

macro_rules! trait_binary_arithmetic {
    ($op: ident, $TensorOpAPI: ident) => {
        pub trait $TensorOpAPI<TRB> {
            type Output;
            fn $op(a: Self, b: TRB) -> Result<Self::Output>;
        }

        pub fn $op<TRA, TRB>(a: TRA, b: TRB) -> Result<TRA::Output>
        where
            TRA: $TensorOpAPI<TRB>,
        {
            TRA::$op(a, b)
        }
    };
}

#[rustfmt::skip]
mod trait_binary_arithmetic {
    use super::*;
    trait_binary_arithmetic!(add   , TensorAddAPI   );
    trait_binary_arithmetic!(sub   , TensorSubAPI   );
    trait_binary_arithmetic!(mul   , TensorMulAPI   );
    trait_binary_arithmetic!(div   , TensorDivAPI   );
    trait_binary_arithmetic!(rem   , TensorRemAPI   );
    trait_binary_arithmetic!(bitor , TensorBitOrAPI );
    trait_binary_arithmetic!(bitand, TensorBitAndAPI);
    trait_binary_arithmetic!(bitxor, TensorBitXorAPI);
    trait_binary_arithmetic!(shl   , TensorShlAPI   );
    trait_binary_arithmetic!(shr   , TensorShrAPI   );
}
pub use trait_binary_arithmetic::*;

macro_rules! impl_binary_arithmetic_ref {
    ($op: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $Op: ident) => {
        impl<RA, RB, TA, TB, TC, DA, DB, DC, B> $TensorOpAPI<&TensorBase<RB, DB>>
            for &TensorBase<RA, DA>
        where
            // tensor types
            RA: DataAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
            B: DeviceCreationAnyAPI<TC>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            // operation constraints
            TA: $Op<TB, Output = TC>,
            B: $DeviceOpAPI<TA, TB, TC, DC>,
        {
            type Output = Tensor<TC, DC, B>;
            fn $op(a: Self, b: &TensorBase<RB, DB>) -> Result<Self::Output> {
                // get tensor views
                let a = a.view();
                let b = b.view();
                // check device and layout
                rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
                let la = a.layout();
                let lb = b.layout();
                let (la_b, lb_b) = broadcast_layout(la, lb)?;
                // generate output layout
                let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::default())?;
                let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::default())?;
                let lc = if lc_from_a == lc_from_b {
                    lc_from_a
                } else {
                    match TensorOrder::default() {
                        TensorOrder::C => la_b.shape().c(),
                        TensorOrder::F => la_b.shape().f(),
                    }
                };
                // generate empty c
                let device = a.device();
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                // add provided by device
                let storage_a = a.data().storage();
                let storage_b = b.data().storage();
                device.op_mutc_refa_refb(
                    &mut storage_c,
                    &lc,
                    storage_a,
                    &la_b,
                    storage_b,
                    &lb_b,
                )?;
                // return tensor
                Tensor::new_f(DataOwned::from(storage_c), lc)
            }
        }

        impl<'a, RB, TA, TB, TC, DA, DB, DC, B> $TensorOpAPI<&TensorBase<RB, DB>>
            for TensorView<'a, TA, DA, B>
        where
            // tensor types
            RB: DataAPI<Data = Storage<TB, B>>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
            B: DeviceCreationAnyAPI<TC>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            // operation constraints
            TA: $Op<TB, Output = TC>,
            B: $DeviceOpAPI<TA, TB, TC, DC>,
        {
            type Output = Tensor<TC, DC, B>;
            fn $op(a: Self, b: &TensorBase<RB, DB>) -> Result<Self::Output> {
                $TensorOpAPI::$op(&a, b)
            }
        }

        impl<'b, RA, TA, TB, TC, DA, DB, DC, B> $TensorOpAPI<TensorView<'b, TB, DB, B>>
            for &TensorBase<RA, DA>
        where
            // tensor types
            RA: DataAPI<Data = Storage<TA, B>>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
            B: DeviceCreationAnyAPI<TC>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            // operation constraints
            TA: $Op<TB, Output = TC>,
            B: $DeviceOpAPI<TA, TB, TC, DC>,
        {
            type Output = Tensor<TC, DC, B>;
            fn $op(a: Self, b: TensorView<'b, TB, DB, B>) -> Result<Self::Output> {
                $TensorOpAPI::$op(a, &b)
            }
        }

        impl<'a, 'b, TA, TB, TC, DA, DB, DC, B> $TensorOpAPI<TensorView<'b, TB, DB, B>>
            for TensorView<'a, TA, DA, B>
        where
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceCreationAnyAPI<TC>,
            B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            // operation constraints
            TA: $Op<TB, Output = TC>,
            B: $DeviceOpAPI<TA, TB, TC, DC>,
        {
            type Output = Tensor<TC, DC, B>;
            fn $op(a: Self, b: TensorView<'b, TB, DB, B>) -> Result<Self::Output> {
                $TensorOpAPI::$op(&a, &b)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_binary_arithmetic_ref {
    use super::*;
    use core::ops::*;
    impl_binary_arithmetic_ref!(add   , DeviceAddAPI   , TensorAddAPI   , Add   );
    impl_binary_arithmetic_ref!(sub   , DeviceSubAPI   , TensorSubAPI   , Sub   );
    impl_binary_arithmetic_ref!(mul   , DeviceMulAPI   , TensorMulAPI   , Mul   );
    impl_binary_arithmetic_ref!(div   , DeviceDivAPI   , TensorDivAPI   , Div   );
    impl_binary_arithmetic_ref!(rem   , DeviceRemAPI   , TensorRemAPI   , Rem   );
    impl_binary_arithmetic_ref!(bitor , DeviceBitOrAPI , TensorBitOrAPI , BitOr );
    impl_binary_arithmetic_ref!(bitand, DeviceBitAndAPI, TensorBitAndAPI, BitAnd);
    impl_binary_arithmetic_ref!(bitxor, DeviceBitXorAPI, TensorBitXorAPI, BitXor);
    impl_binary_arithmetic_ref!(shl   , DeviceShlAPI   , TensorShlAPI   , Shl   );
    impl_binary_arithmetic_ref!(shr   , DeviceShrAPI   , TensorShrAPI   , Shr   );
}

macro_rules! impl_binary_lr_consume {
    ($op: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $Op: ident, $DeviceLConsumeAPI: ident, $DeviceRConsumeAPI: ident) => {
        impl<RB, TA, TB, DA, DB, DC, B> $TensorOpAPI<&TensorBase<RB, DB>> for Tensor<TA, DA, B>
        where
            // tensor
            // types
            RB: DataAPI<Data = Storage<TB, B>>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            B: DeviceCreationAnyAPI<TA>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            DC: DimConvertAPI<DA>,
            DA: DimConvertAPI<DC>,
            // operation constraints
            TA: $Op<TB, Output = TA>,
            B: $DeviceOpAPI<TA, TB, TA, DC>,
            B: $DeviceLConsumeAPI<TA, TB, DA>,
        {
            type Output = Tensor<TA, DC, B>;
            fn $op(a: Self, b: &TensorBase<RB, DB>) -> Result<Self::Output> {
                rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
                let device = a.device().clone();
                let la = a.layout();
                let lb = b.layout();
                let broadcast_result = broadcast_layout_to_first(la, lb);
                if a.layout().is_broadcasted() || broadcast_result.is_err() {
                    // not broadcastable for output a
                    $TensorOpAPI::$op(&a, b)
                } else {
                    // check broadcast layouts
                    let (la_b, lb_b) = broadcast_result?;
                    if la_b != *la {
                        // output shape of c is not the same to input owned a
                        $TensorOpAPI::$op(&a, b)
                    } else {
                        // reuse a as c
                        let mut storage_a = a.data.into_storage();
                        let storage_b = b.data().storage();
                        device.op_muta_refb(&mut storage_a, &la_b, storage_b, &lb_b)?;
                        let c = unsafe { Tensor::new_unchecked(DataOwned::from(storage_a), la_b) };
                        c.into_dim::<DC>()
                    }
                }
            }
        }

        impl<RA, TA, TB, DA, DB, DC, B> $TensorOpAPI<Tensor<TB, DB, B>> for &TensorBase<RA, DA>
        where
            // tensor
            // types
            RA: DataAPI<Data = Storage<TA, B>>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            B: DeviceCreationAnyAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            DB: DimMaxAPI<DA, Max = DC>,
            DC: DimConvertAPI<DB>,
            DB: DimConvertAPI<DC>,
            // operation constraints
            TA: $Op<TB, Output = TB>,
            B: $DeviceOpAPI<TA, TB, TB, DC>,
            B: $DeviceRConsumeAPI<TA, TB, DB>,
        {
            type Output = Tensor<TB, DC, B>;
            fn $op(a: Self, b: Tensor<TB, DB, B>) -> Result<Self::Output> {
                rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
                let device = b.device().clone();
                let la = a.layout();
                let lb = b.layout();
                let broadcast_result = broadcast_layout_to_first(lb, la);
                if b.layout().is_broadcasted() || broadcast_result.is_err() {
                    // not broadcastable for output a
                    $TensorOpAPI::$op(a, &b)
                } else {
                    // check broadcast layouts
                    let (lb_b, la_b) = broadcast_result?;
                    if lb_b != *lb {
                        // output shape of c is not the same to input owned b
                        $TensorOpAPI::$op(a, &b)
                    } else {
                        // reuse b as c
                        let mut storage_b = b.data.into_storage();
                        let storage_a = a.data().storage();
                        device.op_muta_refb(&mut storage_b, &lb_b, storage_a, &la_b)?;
                        let c = unsafe { Tensor::new_unchecked(DataOwned::from(storage_b), lb_b) };
                        c.into_dim::<DC>()
                    }
                }
            }
        }

        impl<'b, TA, TB, DA, DB, DC, B> $TensorOpAPI<TensorView<'b, TB, DB, B>>
            for Tensor<TA, DA, B>
        where
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            B: DeviceCreationAnyAPI<TA>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            DC: DimConvertAPI<DA>,
            DA: DimConvertAPI<DC>,
            // operation constraints
            TA: $Op<TB, Output = TA>,
            B: $DeviceOpAPI<TA, TB, TA, DC>,
            B: $DeviceLConsumeAPI<TA, TB, DA>,
        {
            type Output = Tensor<TA, DC, B>;
            fn $op(a: Self, b: TensorView<'b, TB, DB, B>) -> Result<Self::Output> {
                $TensorOpAPI::$op(a, &b)
            }
        }

        impl<'a, TA, TB, DA, DB, DC, B> $TensorOpAPI<Tensor<TB, DB, B>>
            for TensorView<'a, TA, DA, B>
        where
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            B: DeviceCreationAnyAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            DB: DimMaxAPI<DA, Max = DC>,
            DC: DimConvertAPI<DB>,
            DB: DimConvertAPI<DC>,
            // operation constraints
            TA: $Op<TB, Output = TB>,
            B: $DeviceOpAPI<TA, TB, TB, DC>,
            B: $DeviceRConsumeAPI<TA, TB, DB>,
        {
            type Output = Tensor<TB, DC, B>;
            fn $op(a: Self, b: Tensor<TB, DB, B>) -> Result<Self::Output> {
                $TensorOpAPI::$op(&a, b)
            }
        }

        impl<T, DA, DB, DC, B> $TensorOpAPI<Tensor<T, DB, B>> for Tensor<T, DA, B>
        where
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<T>,
            B: DeviceCreationAnyAPI<T>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC> + DimConvertAPI<DC>,
            DB: DimMaxAPI<DA, Max = DC> + DimConvertAPI<DC>,
            DC: DimConvertAPI<DA> + DimConvertAPI<DB>,
            // operation constraints
            T: $Op<T, Output = T>,
            B: $DeviceOpAPI<T, T, T, DC>,
            B: $DeviceLConsumeAPI<T, T, DA>,
            B: $DeviceRConsumeAPI<T, T, DB>,
        {
            type Output = Tensor<T, DC, B>;
            fn $op(a: Self, b: Tensor<T, DB, B>) -> Result<Self::Output> {
                rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
                let la = a.layout();
                let lb = b.layout();
                let broadcast_result = broadcast_layout_to_first(la, lb);
                if !a.layout().is_broadcasted() && broadcast_result.is_ok() {
                    let (la_b, _) = broadcast_result?;
                    if la_b == *la {
                        return $TensorOpAPI::$op(a, &b);
                    }
                }
                let broadcast_result = broadcast_layout_to_first(lb, la);
                if !b.layout().is_broadcasted() && broadcast_result.is_ok() {
                    let (lb_b, _) = broadcast_result?;
                    if lb_b == *lb {
                        return $TensorOpAPI::$op(&a, b);
                    }
                }
                return $TensorOpAPI::$op(&a, &b);
            }
        }
    };
}

#[rustfmt::skip]
mod impl_binary_lr_consume {
    use super::*;
    use core::ops::*;
    impl_binary_lr_consume!(add   , DeviceAddAPI   , TensorAddAPI   , Add   , DeviceLConsumeAddAPI   , DeviceRConsumeAddAPI   );
    impl_binary_lr_consume!(sub   , DeviceSubAPI   , TensorSubAPI   , Sub   , DeviceLConsumeSubAPI   , DeviceRConsumeSubAPI   );
    impl_binary_lr_consume!(mul   , DeviceMulAPI   , TensorMulAPI   , Mul   , DeviceLConsumeMulAPI   , DeviceRConsumeMulAPI   );
    impl_binary_lr_consume!(div   , DeviceDivAPI   , TensorDivAPI   , Div   , DeviceLConsumeDivAPI   , DeviceRConsumeDivAPI   );
    impl_binary_lr_consume!(rem   , DeviceRemAPI   , TensorRemAPI   , Rem   , DeviceLConsumeRemAPI   , DeviceRConsumeRemAPI   );
    impl_binary_lr_consume!(bitor , DeviceBitOrAPI , TensorBitOrAPI , BitOr , DeviceLConsumeBitOrAPI , DeviceRConsumeBitOrAPI );
    impl_binary_lr_consume!(bitand, DeviceBitAndAPI, TensorBitAndAPI, BitAnd, DeviceLConsumeBitAndAPI, DeviceRConsumeBitAndAPI);
    impl_binary_lr_consume!(bitxor, DeviceBitXorAPI, TensorBitXorAPI, BitXor, DeviceLConsumeBitXorAPI, DeviceRConsumeBitXorAPI);
    impl_binary_lr_consume!(shl   , DeviceShlAPI   , TensorShlAPI   , Shl   , DeviceLConsumeShlAPI   , DeviceRConsumeShlAPI   );
    impl_binary_lr_consume!(shr   , DeviceShrAPI   , TensorShrAPI   , Shr   , DeviceLConsumeShrAPI   , DeviceRConsumeShrAPI   );
}

macro_rules! impl_core_ops {
    ($op: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $Op: ident) => {
        impl<RA, DA, TRB, TRC> $Op<TRB> for &TensorBase<RA, DA>
        where
            DA: DimAPI,
            Self: $TensorOpAPI<TRB, Output = TRC>,
        {
            type Output = TRC;
            fn $op(self, b: TRB) -> TRC {
                $TensorOpAPI::$op(self, b).unwrap()
            }
        }

        impl<'a, TA, DA, B, TRB, TRC> $Op<TRB> for TensorView<'a, TA, DA, B>
        where
            DA: DimAPI,
            B: DeviceAPI<TA>,
            Self: $TensorOpAPI<TRB, Output = TRC>,
        {
            type Output = TRC;
            fn $op(self, b: TRB) -> TRC {
                $TensorOpAPI::$op(self, b).unwrap()
            }
        }

        impl<TA, DA, B, TRB, TRC> $Op<TRB> for Tensor<TA, DA, B>
        where
            DA: DimAPI,
            B: DeviceAPI<TA>,
            Self: $TensorOpAPI<TRB, Output = TRC>,
        {
            type Output = TRC;
            fn $op(self, b: TRB) -> TRC {
                $TensorOpAPI::$op(self, b).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
mod impl_core_ops {
    use super::*;
    use core::ops::*;
    impl_core_ops!(add   , DeviceAddAPI   , TensorAddAPI   , Add   );
    impl_core_ops!(sub   , DeviceSubAPI   , TensorSubAPI   , Sub   );
    impl_core_ops!(mul   , DeviceMulAPI   , TensorMulAPI   , Mul   );
    impl_core_ops!(div   , DeviceDivAPI   , TensorDivAPI   , Div   );
//  impl_core_ops!(rem   , DeviceRemAPI   , TensorRemAPI   , Rem   );
    impl_core_ops!(bitor , DeviceBitOrAPI , TensorBitOrAPI , BitOr );
    impl_core_ops!(bitand, DeviceBitAndAPI, TensorBitAndAPI, BitAnd);
    impl_core_ops!(bitxor, DeviceBitXorAPI, TensorBitXorAPI, BitXor);
    impl_core_ops!(shl   , DeviceShlAPI   , TensorShlAPI   , Shl   );
    impl_core_ops!(shr   , DeviceShrAPI   , TensorShrAPI   , Shr   );
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add() {
        // contiguous
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let c = &a + &b;
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));

        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let c = add(&a, &b).unwrap();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [2, 3] + [3]
        let a = Tensor::linspace((1.0, 6.0, 6)).into_shape_assume_contig([2, 3]);
        let b = Tensor::linspace((2.0, 6.0, 3));
        let c = &a + &b;
        let c_ref = vec![3., 6., 9., 6., 9., 12.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [1, 2, 3] + [5, 1, 2, 1]
        // a = np.linspace(1, 6, 6).reshape(1, 2, 3)
        // b = np.linspace(1, 10, 10).reshape(5, 1, 2, 1)
        let a = Tensor::linspace((1.0, 6.0, 6));
        let a = a.into_shape_assume_contig([1, 2, 3]);
        let b = Tensor::linspace((1.0, 10.0, 10));
        let b = b.into_shape_assume_contig([5, 1, 2, 1]);
        let c = &a + &b;
        let c_ref = vec![
            2., 3., 4., 6., 7., 8., 4., 5., 6., 8., 9., 10., 6., 7., 8., 10., 11., 12., 8., 9.,
            10., 12., 13., 14., 10., 11., 12., 14., 15., 16.,
        ];
        let c_ref = c_ref.into();
        assert!(allclose_f64(&c, &c_ref));

        // transposed
        let a = Tensor::linspace((1.0, 9.0, 9));
        let a = a.into_shape_assume_contig([3, 3]);
        let b = Tensor::linspace((2.0, 18.0, 9));
        let b = b.into_shape_assume_contig([3, 3]).into_reverse_axes();
        let c = &a + &b;
        let c_ref = vec![3., 10., 17., 8., 15., 22., 13., 20., 27.].into();
        assert!(allclose_f64(&c, &c_ref));

        // negative strides
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let a = a.flip(0);
        let c = &a + &b;
        let c_ref = vec![7., 8., 9., 10., 11.].into();
        assert!(allclose_f64(&c, &c_ref));

        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let b = b.flip(0);
        let c = &a + &b;
        let c_ref = vec![11., 10., 9., 8., 7.].into();
        assert!(allclose_f64(&c, &c_ref));

        // view
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let c = a.view() + &b;
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));

        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let c = &a + b.view();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    fn test_sub() {
        // contiguous
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let c = &a - &b;
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    fn test_mul() {
        // contiguous
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let c = &a * &b;
        let c_ref = vec![2., 8., 18., 32., 50.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    fn test_add_consume() {
        // a + &b, same shape
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a + &b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
        // a + &b, broadcastable
        let a = Tensor::linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let b = Tensor::linspace((2.0, 10.0, 5));
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a + &b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15., 8., 11., 14., 17., 20.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
        // a + &b, non-broadcastable
        let a = Tensor::linspace((2.0, 10.0, 5));
        let b = Tensor::linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a + &b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15., 8., 11., 14., 17., 20.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_ne!(a_ptr, c_ptr);
        // &a + b
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let b_ptr = b.data().storage().rawvec().as_ptr();
        let c = &a + b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(b_ptr, c_ptr);
        // &a + b, non-broadcastable
        let a = Tensor::linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let b = Tensor::linspace((2.0, 10.0, 5));
        let b_ptr = b.data().storage().rawvec().as_ptr();
        let c = &a + b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15., 8., 11., 14., 17., 20.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_ne!(b_ptr, c_ptr);
        // a + b, same shape
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a + b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
    }

    #[test]
    fn test_sub_consume() {
        // &a - b
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let b_ptr = b.data().storage().rawvec().as_ptr();
        let c = &a - b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(b_ptr, c_ptr);
        // a - &b
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a - b.view();
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
        // a - b
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = Tensor::linspace((2.0, 10.0, 5));
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a - b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
    }
}
