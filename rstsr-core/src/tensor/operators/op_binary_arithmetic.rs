use crate::prelude_dev::*;

/* #region binary operation function and traits */

macro_rules! trait_binary_arithmetic {
    ($op: ident, $op_f: ident, $TensorOpAPI: ident) => {
        pub trait $TensorOpAPI<TRB> {
            type Output;
            fn $op_f(a: Self, b: TRB) -> Result<Self::Output>;
            fn $op(a: Self, b: TRB) -> Self::Output
            where
                Self: Sized,
            {
                Self::$op_f(a, b).unwrap()
            }
        }

        pub fn $op_f<TRA, TRB>(a: TRA, b: TRB) -> Result<TRA::Output>
        where
            TRA: $TensorOpAPI<TRB>,
        {
            TRA::$op_f(a, b)
        }

        pub fn $op<TRA, TRB>(a: TRA, b: TRB) -> TRA::Output
        where
            TRA: $TensorOpAPI<TRB>,
        {
            TRA::$op(a, b)
        }

        impl<S, D> TensorBase<S, D>
        where
            D: DimAPI,
        {
            pub fn $op_f<TRB>(&self, b: TRB) -> Result<<&Self as $TensorOpAPI<TRB>>::Output>
            where
                for<'a> &'a Self: $TensorOpAPI<TRB>,
            {
                <&Self as $TensorOpAPI<TRB>>::$op_f(self, b)
            }

            pub fn $op<TRB>(&self, b: TRB) -> <&Self as $TensorOpAPI<TRB>>::Output
            where
                for<'a> &'a Self: $TensorOpAPI<TRB>,
            {
                <&Self as $TensorOpAPI<TRB>>::$op(self, b)
            }
        }
    };
}

#[rustfmt::skip]
mod trait_binary_arithmetic {
    use super::*;
    trait_binary_arithmetic!(add   , add_f   , TensorAddAPI   );
    trait_binary_arithmetic!(sub   , sub_f   , TensorSubAPI   );
    trait_binary_arithmetic!(mul   , mul_f   , TensorMulAPI   );
    trait_binary_arithmetic!(div   , div_f   , TensorDivAPI   );
    trait_binary_arithmetic!(rem   , rem_f   , TensorRemAPI   );
    trait_binary_arithmetic!(bitor , bitor_f , TensorBitOrAPI );
    trait_binary_arithmetic!(bitand, bitand_f, TensorBitAndAPI);
    trait_binary_arithmetic!(bitxor, bitxor_f, TensorBitXorAPI);
    trait_binary_arithmetic!(shl   , shl_f   , TensorShlAPI   );
    trait_binary_arithmetic!(shr   , shr_f   , TensorShrAPI   );
}
pub use trait_binary_arithmetic::*;

/* #endregion */

/* #region binary core ops implementation */

macro_rules! impl_core_ops {
    ($op: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $Op: ident) => {
        impl<SA, DA, TRB> $Op<TRB> for &TensorBase<SA, DA>
        where
            DA: DimAPI,
            Self: $TensorOpAPI<TRB>,
        {
            type Output = <Self as $TensorOpAPI<TRB>>::Output;
            fn $op(self, b: TRB) -> Self::Output {
                $TensorOpAPI::$op(self, b)
            }
        }

        impl<'a, TA, DA, B, TRB> $Op<TRB> for TensorView<'a, TA, B, DA>
        where
            DA: DimAPI,
            B: DeviceAPI<TA>,
            Self: $TensorOpAPI<TRB>,
        {
            type Output = <Self as $TensorOpAPI<TRB>>::Output;
            fn $op(self, b: TRB) -> Self::Output {
                $TensorOpAPI::$op(self, b)
            }
        }

        impl<TA, DA, B, TRB> $Op<TRB> for Tensor<TA, B, DA>
        where
            DA: DimAPI,
            B: DeviceAPI<TA>,
            Self: $TensorOpAPI<TRB>,
        {
            type Output = <Self as $TensorOpAPI<TRB>>::Output;
            fn $op(self, b: TRB) -> Self::Output {
                $TensorOpAPI::$op(self, b)
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

/* #endregion */

/* #region binary implementation */

macro_rules! impl_binary_arithmetic_ref {
    ($op_f: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $Op: ident) => {
        #[doc(hidden)]
        impl<RA, RB, TA, TB, TC, DA, DB, DC, B> $TensorOpAPI<&TensorAny<RB, TB, B, DB>>
            for &TensorAny<RA, TA, B, DA>
        where
            // tensor types
            RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
            RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
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
            type Output = Tensor<TC, B, DC>;
            fn $op_f(a: Self, b: &TensorAny<RB, TB, B, DB>) -> Result<Self::Output> {
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
                    match FlagOrder::default() {
                        RowMajor => la_b.shape().c(),
                        ColMajor => la_b.shape().f(),
                    }
                };
                // generate empty c
                let device = a.device();
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                // add provided by device
                device.op_mutc_refa_refb(
                    storage_c.raw_mut(),
                    &lc,
                    a.raw(),
                    &la_b,
                    b.raw(),
                    &lb_b,
                )?;
                // return tensor
                Tensor::new_f(storage_c, lc)
            }
        }

        #[doc(hidden)]
        impl<'a, RB, TA, TB, TC, DA, DB, DC, B> $TensorOpAPI<&TensorAny<RB, TB, B, DB>>
            for TensorView<'a, TA, B, DA>
        where
            // tensor types
            RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
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
            type Output = Tensor<TC, B, DC>;
            fn $op_f(a: Self, b: &TensorAny<RB, TB, B, DB>) -> Result<Self::Output> {
                $TensorOpAPI::$op_f(&a, b)
            }
        }

        #[doc(hidden)]
        impl<'b, RA, TA, TB, TC, DA, DB, DC, B> $TensorOpAPI<TensorView<'b, TB, B, DB>>
            for &TensorAny<RA, TA, B, DA>
        where
            // tensor types
            RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
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
            type Output = Tensor<TC, B, DC>;
            fn $op_f(a: Self, b: TensorView<'b, TB, B, DB>) -> Result<Self::Output> {
                $TensorOpAPI::$op_f(a, &b)
            }
        }

        #[doc(hidden)]
        impl<'a, 'b, TA, TB, TC, DA, DB, DC, B> $TensorOpAPI<TensorView<'b, TB, B, DB>>
            for TensorView<'a, TA, B, DA>
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
            type Output = Tensor<TC, B, DC>;
            fn $op_f(a: Self, b: TensorView<'b, TB, B, DB>) -> Result<Self::Output> {
                $TensorOpAPI::$op_f(&a, &b)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_binary_arithmetic_ref {
    use super::*;
    use core::ops::*;
    impl_binary_arithmetic_ref!(add_f   , DeviceAddAPI   , TensorAddAPI   , Add   );
    impl_binary_arithmetic_ref!(sub_f   , DeviceSubAPI   , TensorSubAPI   , Sub   );
    impl_binary_arithmetic_ref!(mul_f   , DeviceMulAPI   , TensorMulAPI   , Mul   );
    impl_binary_arithmetic_ref!(div_f   , DeviceDivAPI   , TensorDivAPI   , Div   );
    impl_binary_arithmetic_ref!(rem_f   , DeviceRemAPI   , TensorRemAPI   , Rem   );
    impl_binary_arithmetic_ref!(bitor_f , DeviceBitOrAPI , TensorBitOrAPI , BitOr );
    impl_binary_arithmetic_ref!(bitand_f, DeviceBitAndAPI, TensorBitAndAPI, BitAnd);
    impl_binary_arithmetic_ref!(bitxor_f, DeviceBitXorAPI, TensorBitXorAPI, BitXor);
    impl_binary_arithmetic_ref!(shl_f   , DeviceShlAPI   , TensorShlAPI   , Shl   );
    impl_binary_arithmetic_ref!(shr_f   , DeviceShrAPI   , TensorShrAPI   , Shr   );
}

macro_rules! impl_binary_lr_consume {
    ($op_f: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $Op: ident, $DeviceLConsumeAPI: ident, $DeviceRConsumeAPI: ident) => {
        #[doc(hidden)]
        impl<RB, TA, TB, DA, DB, DC, B> $TensorOpAPI<&TensorAny<RB, TB, B, DB>>
            for Tensor<TA, B, DA>
        where
            // tensor types
            RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            B: DeviceCreationAnyAPI<TA>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            DC: DimIntoAPI<DA>,
            DA: DimIntoAPI<DC>,
            // operation constraints
            TA: $Op<TB, Output = TA>,
            B: $DeviceOpAPI<TA, TB, TA, DC>,
            B: $DeviceLConsumeAPI<TA, TB, DA>,
        {
            type Output = Tensor<TA, B, DC>;
            fn $op_f(a: Self, b: &TensorAny<RB, TB, B, DB>) -> Result<Self::Output> {
                rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
                let device = a.device().clone();
                let la = a.layout();
                let lb = b.layout();
                let broadcast_result = broadcast_layout_to_first(la, lb);
                if a.layout().is_broadcasted() || broadcast_result.is_err() {
                    // not broadcastable for output a
                    $TensorOpAPI::$op_f(&a, b)
                } else {
                    // check broadcast layouts
                    let (la_b, lb_b) = broadcast_result?;
                    if la_b != *la {
                        // output shape of c is not the same to input owned a
                        $TensorOpAPI::$op_f(&a, b)
                    } else {
                        // reuse a as c
                        let (mut storage_a, _) = a.into_raw_parts();
                        device.op_muta_refb(storage_a.raw_mut(), &la_b, b.raw(), &lb_b)?;
                        let c = unsafe { Tensor::new_unchecked(storage_a, la_b) };
                        c.into_dim_f::<DC>()
                    }
                }
            }
        }

        #[doc(hidden)]
        impl<RA, TA, TB, DA, DB, DC, B> $TensorOpAPI<Tensor<TB, B, DB>>
            for &TensorAny<RA, TA, B, DA>
        where
            // tensor
            // types
            RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            B: DeviceCreationAnyAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            DB: DimMaxAPI<DA, Max = DC>,
            DC: DimIntoAPI<DB>,
            DB: DimIntoAPI<DC>,
            // operation constraints
            TA: $Op<TB, Output = TB>,
            B: $DeviceOpAPI<TA, TB, TB, DC>,
            B: $DeviceRConsumeAPI<TA, TB, DB>,
        {
            type Output = Tensor<TB, B, DC>;
            fn $op_f(a: Self, b: Tensor<TB, B, DB>) -> Result<Self::Output> {
                rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
                let device = b.device().clone();
                let la = a.layout();
                let lb = b.layout();
                let broadcast_result = broadcast_layout_to_first(lb, la);
                if b.layout().is_broadcasted() || broadcast_result.is_err() {
                    // not broadcastable for output a
                    $TensorOpAPI::$op_f(a, &b)
                } else {
                    // check broadcast layouts
                    let (lb_b, la_b) = broadcast_result?;
                    if lb_b != *lb {
                        // output shape of c is not the same to input owned b
                        $TensorOpAPI::$op_f(a, &b)
                    } else {
                        // reuse b as c
                        let (mut storage_b, _) = b.into_raw_parts();
                        device.op_muta_refb(storage_b.raw_mut(), &lb_b, a.raw(), &la_b)?;
                        let c = unsafe { Tensor::new_unchecked(storage_b, lb_b) };
                        c.into_dim_f::<DC>()
                    }
                }
            }
        }

        #[doc(hidden)]
        impl<'b, TA, TB, DA, DB, DC, B> $TensorOpAPI<TensorView<'b, TB, B, DB>>
            for Tensor<TA, B, DA>
        where
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            B: DeviceCreationAnyAPI<TA>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC>,
            DC: DimIntoAPI<DA>,
            DA: DimIntoAPI<DC>,
            // operation constraints
            TA: $Op<TB, Output = TA>,
            B: $DeviceOpAPI<TA, TB, TA, DC>,
            B: $DeviceLConsumeAPI<TA, TB, DA>,
        {
            type Output = Tensor<TA, B, DC>;
            fn $op_f(a: Self, b: TensorView<'b, TB, B, DB>) -> Result<Self::Output> {
                $TensorOpAPI::$op_f(a, &b)
            }
        }

        #[doc(hidden)]
        impl<'a, TA, TB, DA, DB, DC, B> $TensorOpAPI<Tensor<TB, B, DB>>
            for TensorView<'a, TA, B, DA>
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
            DC: DimIntoAPI<DB>,
            DB: DimIntoAPI<DC>,
            // operation constraints
            TA: $Op<TB, Output = TB>,
            B: $DeviceOpAPI<TA, TB, TB, DC>,
            B: $DeviceRConsumeAPI<TA, TB, DB>,
        {
            type Output = Tensor<TB, B, DC>;
            fn $op_f(a: Self, b: Tensor<TB, B, DB>) -> Result<Self::Output> {
                $TensorOpAPI::$op_f(&a, b)
            }
        }

        #[doc(hidden)]
        impl<T, DA, DB, DC, B> $TensorOpAPI<Tensor<T, B, DB>> for Tensor<T, B, DA>
        where
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<T>,
            B: DeviceCreationAnyAPI<T>,
            // broadcast constraints
            DA: DimMaxAPI<DB, Max = DC> + DimIntoAPI<DC>,
            DB: DimMaxAPI<DA, Max = DC> + DimIntoAPI<DC>,
            DC: DimIntoAPI<DA> + DimIntoAPI<DB>,
            // operation constraints
            T: $Op<T, Output = T>,
            B: $DeviceOpAPI<T, T, T, DC>,
            B: $DeviceLConsumeAPI<T, T, DA>,
            B: $DeviceRConsumeAPI<T, T, DB>,
        {
            type Output = Tensor<T, B, DC>;
            fn $op_f(a: Self, b: Tensor<T, B, DB>) -> Result<Self::Output> {
                rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
                let la = a.layout();
                let lb = b.layout();
                let broadcast_result = broadcast_layout_to_first(la, lb);
                if !a.layout().is_broadcasted() && broadcast_result.is_ok() {
                    let (la_b, _) = broadcast_result?;
                    if la_b == *la {
                        return $TensorOpAPI::$op_f(a, &b);
                    }
                }
                let broadcast_result = broadcast_layout_to_first(lb, la);
                if !b.layout().is_broadcasted() && broadcast_result.is_ok() {
                    let (lb_b, _) = broadcast_result?;
                    if lb_b == *lb {
                        return $TensorOpAPI::$op_f(&a, b);
                    }
                }
                return $TensorOpAPI::$op_f(&a, &b);
            }
        }
    };
}

#[rustfmt::skip]
mod impl_binary_lr_consume {
    use super::*;
    use core::ops::*;
    impl_binary_lr_consume!(add_f   , DeviceAddAPI   , TensorAddAPI   , Add   , DeviceLConsumeAddAPI   , DeviceRConsumeAddAPI   );
    impl_binary_lr_consume!(sub_f   , DeviceSubAPI   , TensorSubAPI   , Sub   , DeviceLConsumeSubAPI   , DeviceRConsumeSubAPI   );
    impl_binary_lr_consume!(mul_f   , DeviceMulAPI   , TensorMulAPI   , Mul   , DeviceLConsumeMulAPI   , DeviceRConsumeMulAPI   );
    impl_binary_lr_consume!(div_f   , DeviceDivAPI   , TensorDivAPI   , Div   , DeviceLConsumeDivAPI   , DeviceRConsumeDivAPI   );
    impl_binary_lr_consume!(rem_f   , DeviceRemAPI   , TensorRemAPI   , Rem   , DeviceLConsumeRemAPI   , DeviceRConsumeRemAPI   );
    impl_binary_lr_consume!(bitor_f , DeviceBitOrAPI , TensorBitOrAPI , BitOr , DeviceLConsumeBitOrAPI , DeviceRConsumeBitOrAPI );
    impl_binary_lr_consume!(bitand_f, DeviceBitAndAPI, TensorBitAndAPI, BitAnd, DeviceLConsumeBitAndAPI, DeviceRConsumeBitAndAPI);
    impl_binary_lr_consume!(bitxor_f, DeviceBitXorAPI, TensorBitXorAPI, BitXor, DeviceLConsumeBitXorAPI, DeviceRConsumeBitXorAPI);
    impl_binary_lr_consume!(shl_f   , DeviceShlAPI   , TensorShlAPI   , Shl   , DeviceLConsumeShlAPI   , DeviceRConsumeShlAPI   );
    impl_binary_lr_consume!(shr_f   , DeviceShrAPI   , TensorShrAPI   , Shr   , DeviceLConsumeShrAPI   , DeviceRConsumeShrAPI   );
}

/* #endregion */

/* #region binary with output implementation */

macro_rules! impl_binary_with_output {
    ($op: ident, $op_f: ident, $DeviceOpAPI: ident, $Op: ident) => {
        pub fn $op_f<TRA, TRB, TRC, TA, TB, TC, DA, DB, DC, B>(
            a: TRA,
            b: TRB,
            mut c: TRC,
        ) -> Result<()>
        where
            // tensor types
            TRA: TensorViewAPI<TA, B, DA>,
            TRB: TensorViewAPI<TB, B, DB>,
            TRC: TensorViewMutAPI<TC, B, DC>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
            // broadcast constraints
            DC: DimMaxAPI<DA, Max = DC> + DimMaxAPI<DB, Max = DC>,
            // operation constraints
            TA: $Op<TB, Output = TC>,
            B: $DeviceOpAPI<TA, TB, TC, DC>,
        {
            // get tensor views
            let a = a.view();
            let b = b.view();
            let mut c = c.view_mut();
            // check device
            rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
            rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
            let lc = c.layout();
            let la = a.layout();
            let lb = b.layout();
            // all layouts should be broadcastable to lc
            // we can first generate broadcasted shape, then check this
            let (lc_b, la_b) = broadcast_layout_to_first(lc, la)?;
            rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
            let (lc_b, lb_b) = broadcast_layout_to_first(lc, lb)?;
            rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
            // op provided by device
            let device = c.device().clone();
            device.op_mutc_refa_refb(c.raw_mut(), &lc_b, a.raw(), &la_b, b.raw(), &lb_b)
        }

        pub fn $op<TRA, TRB, TRC, TA, TB, TC, DA, DB, DC, B>(a: TRA, b: TRB, c: TRC)
        where
            // tensor types
            TRA: TensorViewAPI<TA, B, DA>,
            TRB: TensorViewAPI<TB, B, DB>,
            TRC: TensorViewMutAPI<TC, B, DC>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
            // broadcast constraints
            DC: DimMaxAPI<DA, Max = DC> + DimMaxAPI<DB, Max = DC>,
            // operation constraints
            TA: $Op<TB, Output = TC>,
            B: $DeviceOpAPI<TA, TB, TC, DC>,
        {
            $op_f(a, b, c).unwrap()
        }
    };
}

#[rustfmt::skip]
mod impl_binary_with_output{
    use super::*;
    use core::ops::*;
    impl_binary_with_output!(   add_with_output,    add_with_output_f, DeviceAddAPI   , Add   );
    impl_binary_with_output!(   sub_with_output,    sub_with_output_f, DeviceSubAPI   , Sub   );
    impl_binary_with_output!(   mul_with_output,    mul_with_output_f, DeviceMulAPI   , Mul   );
    impl_binary_with_output!(   div_with_output,    div_with_output_f, DeviceDivAPI   , Div   );
    impl_binary_with_output!(   rem_with_output,    rem_with_output_f, DeviceRemAPI   , Rem   );
    impl_binary_with_output!( bitor_with_output,  bitor_with_output_f, DeviceBitOrAPI , BitOr );
    impl_binary_with_output!(bitand_with_output, bitand_with_output_f, DeviceBitAndAPI, BitAnd);
    impl_binary_with_output!(bitxor_with_output, bitxor_with_output_f, DeviceBitXorAPI, BitXor);
    impl_binary_with_output!(   shl_with_output,    shl_with_output_f, DeviceShlAPI   , Shl   );
    impl_binary_with_output!(   shr_with_output,    shr_with_output_f, DeviceShrAPI   , Shr   );
}
pub use impl_binary_with_output::*;

/* #endregion */

/* #region binary with scalar, num a op tsr b */

macro_rules! impl_arithmetic_scalar_lhs {
    ($ty: ty, $op: ident, $op_f: ident, $Op: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $DeviceRConsumeOpAPI: ident) => {
        #[doc(hidden)]
        impl<T, R, D, B> $TensorOpAPI<&TensorAny<R, T, B, D>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            R: DataAPI<Data = B::Raw>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
        {
            type Output = Tensor<T, B, D>;
            fn $op_f(a: Self, b: &TensorAny<R, T, B, D>) -> Result<Self::Output> {
                let a = T::from(a);
                let device = b.device();
                let lb = b.layout();
                let lc = layout_for_array_copy(lb, TensorIterOrder::default())?;
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                device.op_mutc_numa_refb(storage_c.raw_mut(), &lc, a, b.raw(), lb)?;
                Tensor::new_f(storage_c, lc)
            }
        }

        #[doc(hidden)]
        impl<T, R, D, B> $Op<&TensorAny<R, T, B, D>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            R: DataAPI<Data = B::Raw>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
        {
            type Output = Tensor<T, B, D>;
            fn $op(self, rhs: &TensorAny<R, T, B, D>) -> Self::Output {
                $TensorOpAPI::$op_f(self, rhs).unwrap()
            }
        }

        #[doc(hidden)]
        impl<'l, T, B, D> $TensorOpAPI<TensorView<'l, T, B, D>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
        {
            type Output = Tensor<T, B, D>;
            fn $op_f(a: Self, b: TensorView<'l, T, B, D>) -> Result<Self::Output> {
                let a = T::from(a);
                let device = b.device();
                let lb = b.layout();
                let lc = layout_for_array_copy(lb, TensorIterOrder::default())?;
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                device.op_mutc_numa_refb(storage_c.raw_mut(), &lc, a, b.raw(), lb)?;
                Tensor::new_f(storage_c, lc)
            }
        }

        #[doc(hidden)]
        impl<'l, T, B, D> $Op<TensorView<'l, T, B, D>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
        {
            type Output = Tensor<T, B, D>;
            fn $op(self, rhs: TensorView<'l, T, B, D>) -> Self::Output {
                $TensorOpAPI::$op_f(self, rhs).unwrap()
            }
        }

        #[doc(hidden)]
        impl<T, B, D> $TensorOpAPI<Tensor<T, B, D>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceRConsumeOpAPI<T, T, D>,
        {
            type Output = Tensor<T, B, D>;
            fn $op_f(a: Self, mut b: Tensor<T, B, D>) -> Result<Self::Output> {
                let a = T::from(a);
                let device = b.device().clone();
                let lb = b.layout().clone();
                device.op_muta_numb(b.raw_mut(), &lb, a)?;
                return Ok(b);
            }
        }

        #[doc(hidden)]
        impl<T, B, D> $Op<Tensor<T, B, D>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceRConsumeOpAPI<T, T, D>,
        {
            type Output = Tensor<T, B, D>;
            fn $op(self, rhs: Tensor<T, B, D>) -> Self::Output {
                $TensorOpAPI::$op_f(self, rhs).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
macro_rules! impl_arithmetic_scalar_lhs_all {
    ($ty: ty) => {
        impl_arithmetic_scalar_lhs!($ty, add   , add_f   , Add   , DeviceAddAPI   , TensorAddAPI   , DeviceRConsumeAddAPI   );
        impl_arithmetic_scalar_lhs!($ty, sub   , sub_f   , Sub   , DeviceSubAPI   , TensorSubAPI   , DeviceRConsumeSubAPI   );
        impl_arithmetic_scalar_lhs!($ty, mul   , mul_f   , Mul   , DeviceMulAPI   , TensorMulAPI   , DeviceRConsumeMulAPI   );
        impl_arithmetic_scalar_lhs!($ty, div   , div_f   , Div   , DeviceDivAPI   , TensorDivAPI   , DeviceRConsumeDivAPI   );
        impl_arithmetic_scalar_lhs!($ty, rem   , rem_f   , Rem   , DeviceRemAPI   , TensorRemAPI   , DeviceRConsumeRemAPI   );
        impl_arithmetic_scalar_lhs!($ty, bitor , bitor_f , BitOr , DeviceBitOrAPI , TensorBitOrAPI , DeviceRConsumeBitOrAPI );
        impl_arithmetic_scalar_lhs!($ty, bitand, bitand_f, BitAnd, DeviceBitAndAPI, TensorBitAndAPI, DeviceRConsumeBitAndAPI);
        impl_arithmetic_scalar_lhs!($ty, bitxor, bitxor_f, BitXor, DeviceBitXorAPI, TensorBitXorAPI, DeviceRConsumeBitXorAPI);
        impl_arithmetic_scalar_lhs!($ty, shl   , shl_f   , Shl   , DeviceShlAPI   , TensorShlAPI   , DeviceRConsumeShlAPI   );
        impl_arithmetic_scalar_lhs!($ty, shr   , shr_f   , Shr   , DeviceShrAPI   , TensorShrAPI   , DeviceRConsumeShrAPI   );
    };
}

#[rustfmt::skip]
macro_rules! impl_arithmetic_scalar_lhs_bool {
    ($ty: ty) => {
        impl_arithmetic_scalar_lhs!($ty, bitor , bitor_f , BitOr , DeviceBitOrAPI , TensorBitOrAPI , DeviceRConsumeBitOrAPI );
        impl_arithmetic_scalar_lhs!($ty, bitand, bitand_f, BitAnd, DeviceBitAndAPI, TensorBitAndAPI, DeviceRConsumeBitAndAPI);
        impl_arithmetic_scalar_lhs!($ty, bitxor, bitxor_f, BitXor, DeviceBitXorAPI, TensorBitXorAPI, DeviceRConsumeBitXorAPI);
    };
}

#[rustfmt::skip]
macro_rules! impl_arithmetic_scalar_lhs_float {
    ($ty: ty) => {
        impl_arithmetic_scalar_lhs!($ty, add , add_f , Add   , DeviceAddAPI   , TensorAddAPI   , DeviceRConsumeAddAPI   );
        impl_arithmetic_scalar_lhs!($ty, sub , sub_f , Sub   , DeviceSubAPI   , TensorSubAPI   , DeviceRConsumeSubAPI   );
        impl_arithmetic_scalar_lhs!($ty, mul , mul_f , Mul   , DeviceMulAPI   , TensorMulAPI   , DeviceRConsumeMulAPI   );
        impl_arithmetic_scalar_lhs!($ty, div , div_f , Div   , DeviceDivAPI   , TensorDivAPI   , DeviceRConsumeDivAPI   );
    };
}

mod impl_arithmetic_scalar_lhs {
    use super::*;
    use core::ops::*;
    use half::{bf16, f16};
    use num::complex::Complex;
    impl_arithmetic_scalar_lhs_all!(i8);
    impl_arithmetic_scalar_lhs_all!(u8);
    impl_arithmetic_scalar_lhs_all!(i16);
    impl_arithmetic_scalar_lhs_all!(u16);
    impl_arithmetic_scalar_lhs_all!(i32);
    impl_arithmetic_scalar_lhs_all!(u32);
    impl_arithmetic_scalar_lhs_all!(i64);
    impl_arithmetic_scalar_lhs_all!(u64);
    impl_arithmetic_scalar_lhs_all!(i128);
    impl_arithmetic_scalar_lhs_all!(u128);
    impl_arithmetic_scalar_lhs_all!(isize);
    impl_arithmetic_scalar_lhs_all!(usize);

    impl_arithmetic_scalar_lhs_bool!(bool);

    impl_arithmetic_scalar_lhs_float!(bf16);
    impl_arithmetic_scalar_lhs_float!(f16);
    impl_arithmetic_scalar_lhs_float!(f32);
    impl_arithmetic_scalar_lhs_float!(f64);
    impl_arithmetic_scalar_lhs_float!(Complex<bf16>);
    impl_arithmetic_scalar_lhs_float!(Complex<f16>);
    impl_arithmetic_scalar_lhs_float!(Complex<f32>);
    impl_arithmetic_scalar_lhs_float!(Complex<f64>);
}

/* #endregion */

/* #region binary with scalar, tsr a op num b */

// for this case, core::ops::* is not required to be re-implemented
// see macro_rule `impl_core_ops`

macro_rules! impl_arithmetic_scalar_rhs {
    ($op_f: ident, $Op: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $DeviceLConsumeOpAPI: ident) => {
        #[doc(hidden)]
        impl<T, TB, R, D, B> $TensorOpAPI<TB> for &TensorAny<R, T, B, D>
        where
            T: From<TB> + $Op<T, Output = T>,
            R: DataAPI<Data = B::Raw>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
            // this constraint prohibits confliting impl to TensorBase<RB, D>
            TB: num::Num,
        {
            type Output = Tensor<T, B, D>;
            fn $op_f(a: Self, b: TB) -> Result<Self::Output> {
                let b = T::from(b);
                let device = a.device();
                let la = a.layout();
                let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                device.op_mutc_refa_numb(storage_c.raw_mut(), &lc, a.raw(), la, b)?;
                Tensor::new_f(storage_c, lc)
            }
        }

        #[doc(hidden)]
        impl<'l, T, TB, D, B> $TensorOpAPI<TB> for TensorView<'l, T, B, D>
        where
            T: From<TB> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
            // this constraint prohibits confliting impl to TensorBase<RB, D>
            TB: num::Num,
        {
            type Output = Tensor<T, B, D>;
            fn $op_f(a: Self, b: TB) -> Result<Self::Output> {
                let b = T::from(b);
                let device = a.device();
                let la = a.layout();
                let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                device.op_mutc_refa_numb(storage_c.raw_mut(), &lc, a.raw(), la, b)?;
                Tensor::new_f(storage_c, lc)
            }
        }

        #[doc(hidden)]
        impl<T, TB, D, B> $TensorOpAPI<TB> for Tensor<T, B, D>
        where
            T: From<TB> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceLConsumeOpAPI<T, T, D>,
            // this constraint prohibits confliting impl to TensorBase<RB, D>
            TB: num::Num,
        {
            type Output = Tensor<T, B, D>;
            fn $op_f(mut a: Self, b: TB) -> Result<Self::Output> {
                let b = T::from(b);
                let device = a.device().clone();
                let la = a.layout().clone();
                device.op_muta_numb(a.raw_mut(), &la, b)?;
                return Ok(a);
            }
        }
    };
}

#[rustfmt::skip]
mod impl_arithmetic_scalar_rhs {
    use super::*;
    use core::ops::*;
    impl_arithmetic_scalar_rhs!(add_f   , Add   , DeviceAddAPI   , TensorAddAPI   , DeviceLConsumeAddAPI   );
    impl_arithmetic_scalar_rhs!(sub_f   , Sub   , DeviceSubAPI   , TensorSubAPI   , DeviceLConsumeSubAPI   );
    impl_arithmetic_scalar_rhs!(mul_f   , Mul   , DeviceMulAPI   , TensorMulAPI   , DeviceLConsumeMulAPI   );
    impl_arithmetic_scalar_rhs!(div_f   , Div   , DeviceDivAPI   , TensorDivAPI   , DeviceLConsumeDivAPI   );
    impl_arithmetic_scalar_rhs!(rem_f   , Rem   , DeviceRemAPI   , TensorRemAPI   , DeviceLConsumeRemAPI   );
    impl_arithmetic_scalar_rhs!(bitor_f , BitOr , DeviceBitOrAPI , TensorBitOrAPI , DeviceLConsumeBitOrAPI );
    impl_arithmetic_scalar_rhs!(bitand_f, BitAnd, DeviceBitAndAPI, TensorBitAndAPI, DeviceLConsumeBitAndAPI);
    impl_arithmetic_scalar_rhs!(bitxor_f, BitXor, DeviceBitXorAPI, TensorBitXorAPI, DeviceLConsumeBitXorAPI);
    impl_arithmetic_scalar_rhs!(shl_f   , Shl   , DeviceShlAPI   , TensorShlAPI   , DeviceLConsumeShlAPI   );
    impl_arithmetic_scalar_rhs!(shr_f   , Shr   , DeviceShrAPI   , TensorShrAPI   , DeviceLConsumeShrAPI   );
}

/* #endregion */

/* #region test */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add() {
        // contiguous
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let c = &a + &b;
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));

        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let c = add(&a, &b);
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [2, 3] + [3]
        let a = linspace((1.0, 6.0, 6)).into_shape_assume_contig([2, 3]);
        let b = linspace((2.0, 6.0, 3));
        let c = &a + &b;
        let c_ref = vec![3., 6., 9., 6., 9., 12.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [1, 2, 3] + [5, 1, 2, 1]
        // a = np.linspace(1, 6, 6).reshape(1, 2, 3)
        // b = np.linspace(1, 10, 10).reshape(5, 1, 2, 1)
        let a = linspace((1.0, 6.0, 6));
        let a = a.into_shape_assume_contig([1, 2, 3]);
        let b = linspace((1.0, 10.0, 10));
        let b = b.into_shape_assume_contig([5, 1, 2, 1]);
        let c = &a + &b;
        let c_ref = vec![
            2., 3., 4., 6., 7., 8., 4., 5., 6., 8., 9., 10., 6., 7., 8., 10., 11., 12., 8., 9.,
            10., 12., 13., 14., 10., 11., 12., 14., 15., 16.,
        ];
        let c_ref = c_ref.into();
        assert!(allclose_f64(&c, &c_ref));

        // transposed
        let a = linspace((1.0, 9.0, 9));
        let a = a.into_shape_assume_contig([3, 3]);
        let b = linspace((2.0, 18.0, 9));
        let b = b.into_shape_assume_contig([3, 3]).into_reverse_axes();
        let c = &a + &b;
        let c_ref = vec![3., 10., 17., 8., 15., 22., 13., 20., 27.].into();
        assert!(allclose_f64(&c, &c_ref));

        // negative strides
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let a = a.flip(0);
        let c = &a + &b;
        let c_ref = vec![7., 8., 9., 10., 11.].into();
        assert!(allclose_f64(&c, &c_ref));

        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let b = b.flip(0);
        let c = &a + &b;
        let c_ref = vec![11., 10., 9., 8., 7.].into();
        assert!(allclose_f64(&c, &c_ref));

        // view
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let c = a.view() + &b;
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));

        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let c = &a + b.view();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    fn test_sub() {
        // contiguous
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let c = &a - &b;
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    fn test_mul() {
        // contiguous
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let c = &a * &b;
        let c_ref = vec![2., 8., 18., 32., 50.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    fn test_add_consume() {
        // a + &b, same shape
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let a_ptr = a.raw().as_ptr();
        let c = a + &b;
        let c_ptr = c.raw().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
        // a + &b, broadcastable
        let a = linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let b = linspace((2.0, 10.0, 5));
        let a_ptr = a.raw().as_ptr();
        let c = a + &b;
        let c_ptr = c.raw().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15., 8., 11., 14., 17., 20.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
        // a + &b, non-broadcastable
        let a = linspace((2.0, 10.0, 5));
        let b = linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let a_ptr = a.raw().as_ptr();
        let c = a + &b;
        let c_ptr = c.raw().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15., 8., 11., 14., 17., 20.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_ne!(a_ptr, c_ptr);
        // &a + b
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let b_ptr = b.raw().as_ptr();
        let c = &a + b;
        let c_ptr = c.raw().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(b_ptr, c_ptr);
        // &a + b, non-broadcastable
        let a = linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let b = linspace((2.0, 10.0, 5));
        let b_ptr = b.raw().as_ptr();
        let c = &a + b;
        let c_ptr = c.raw().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15., 8., 11., 14., 17., 20.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_ne!(b_ptr, c_ptr);
        // a + b, same shape
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let a_ptr = a.raw().as_ptr();
        let c = a + b;
        let c_ptr = c.raw().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
    }

    #[test]
    fn test_sub_consume() {
        // &a - b
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let b_ptr = b.raw().as_ptr();
        let c = &a - b;
        let c_ptr = c.raw().as_ptr();
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(b_ptr, c_ptr);
        // a - &b
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let a_ptr = a.raw().as_ptr();
        let c = a - b.view();
        let c_ptr = c.raw().as_ptr();
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
        // a - b
        let a = linspace((1.0, 5.0, 5));
        let b = linspace((2.0, 10.0, 5));
        let a_ptr = a.raw().as_ptr();
        let c = a - b;
        let c_ptr = c.raw().as_ptr();
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
    }
}

#[cfg(test)]
mod test_with_output {
    use super::*;

    #[test]
    fn test_op_binary_with_output() {
        let a = linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let b = linspace((2.0, 10.0, 5)).into_layout([5].c());
        let mut c = linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let c_view = c.view_mut();
        add_with_output(&a, b, c_view);
        println!("{:?}", c);
    }
}

#[cfg(test)]
mod tests_with_scalar {
    use super::*;

    #[test]
    fn test_add() {
        // b - &a
        let a = linspace((1.0, 5.0, 5));
        let b = 1;
        let c = b - &a;
        let c_ref = vec![0., -1., -2., -3., -4.].into();
        assert!(allclose_f64(&c, &c_ref));

        // &a - b
        let a = linspace((1.0, 5.0, 5));
        let b = 1;
        let c = &a - b;
        let c_ref = vec![0., 1., 2., 3., 4.].into();
        assert!(allclose_f64(&c, &c_ref));

        // b * a
        let a = linspace((1.0, 5.0, 5));
        let a_ptr = a.raw().as_ptr();
        let b = 2;
        let c: Tensor<_> = -b * a;
        let c_ref = vec![-2., -4., -6., -8., -10.].into();
        assert!(allclose_f64(&c, &c_ref));
        let c_ptr = c.raw().as_ptr();
        assert_eq!(a_ptr, c_ptr);
    }

    #[test]
    fn test_scalar_consequent() {
        let a = linspace((1.0, 5.0, 5));
        let mut c = linspace((1.0, 5.0, 5));
        // TODO: currently `let b = 2 * a` will give compiler error
        // type must be known at this point
        // I'm not sure why this happens, maybe rust's type inference problem?
        let b = a * 2;
        *&mut c.i_mut(1) += b.i(1);
        println!("{:?}", c);
    }
}

/* #endregion */
