use crate::prelude_dev::*;
use crate::tensor::operators::*;
use core::ops::*;
use half::{bf16, f16};
use num::complex::Complex;

// this file is branched from `op_binary_arithmetic.rs`

/* #region num a op tsr b */

macro_rules! impl_arithmetic_scalar_lhs {
    ($ty: ty, $op: ident, $Op: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $DeviceRConsumeOpAPI: ident) => {
        impl<T, R, D, B> $TensorOpAPI<&TensorBase<R, D>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            R: DataAPI<Data = Storage<T, B>>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(a: Self, b: &TensorBase<R, D>) -> Result<Self::Output> {
                let a = T::from(a);
                let device = b.device();
                let lb = b.layout();
                let lc = layout_for_array_copy(lb, TensorIterOrder::default())?;
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                let storage_b = b.storage();
                device.op_mutc_numa_refb(&mut storage_c, &lc, a, storage_b, lb)?;
                Tensor::new_f(DataOwned::from(storage_c), lc)
            }
        }

        impl<T, R, D, B> $Op<&TensorBase<R, D>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            R: DataAPI<Data = Storage<T, B>>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(self, rhs: &TensorBase<R, D>) -> Self::Output {
                $TensorOpAPI::$op(self, rhs).unwrap()
            }
        }

        impl<'l, T, D, B> $TensorOpAPI<TensorView<'l, T, D, B>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(a: Self, b: TensorView<'l, T, D, B>) -> Result<Self::Output> {
                let a = T::from(a);
                let device = b.device();
                let lb = b.layout();
                let lc = layout_for_array_copy(lb, TensorIterOrder::default())?;
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                let storage_b = b.storage();
                device.op_mutc_numa_refb(&mut storage_c, &lc, a, storage_b, lb)?;
                Tensor::new_f(DataOwned::from(storage_c), lc)
            }
        }

        impl<'l, T, D, B> $Op<TensorView<'l, T, D, B>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(self, rhs: TensorView<'l, T, D, B>) -> Self::Output {
                $TensorOpAPI::$op(self, rhs).unwrap()
            }
        }

        impl<T, D, B> $TensorOpAPI<Tensor<T, D, B>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceRConsumeOpAPI<T, T, D>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(a: Self, mut b: Tensor<T, D, B>) -> Result<Self::Output> {
                let a = T::from(a);
                let device = b.device().clone();
                let lb = b.layout().clone();
                let storage_b = b.data_mut().storage_mut();
                device.op_muta_numb(storage_b, &lb, a)?;
                return Ok(b);
            }
        }

        impl<T, D, B> $Op<Tensor<T, D, B>> for $ty
        where
            T: From<$ty> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceRConsumeOpAPI<T, T, D>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(self, rhs: Tensor<T, D, B>) -> Self::Output {
                $TensorOpAPI::$op(self, rhs).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
macro_rules! impl_arithmetic_scalar_lhs_all {
    ($ty: ty) => {
        impl_arithmetic_scalar_lhs!($ty, add   , Add   , DeviceAddAPI   , TensorAddAPI   , DeviceRConsumeAddAPI   );
        impl_arithmetic_scalar_lhs!($ty, sub   , Sub   , DeviceSubAPI   , TensorSubAPI   , DeviceRConsumeSubAPI   );
        impl_arithmetic_scalar_lhs!($ty, mul   , Mul   , DeviceMulAPI   , TensorMulAPI   , DeviceRConsumeMulAPI   );
        impl_arithmetic_scalar_lhs!($ty, div   , Div   , DeviceDivAPI   , TensorDivAPI   , DeviceRConsumeDivAPI   );
        impl_arithmetic_scalar_lhs!($ty, rem   , Rem   , DeviceRemAPI   , TensorRemAPI   , DeviceRConsumeRemAPI   );
        impl_arithmetic_scalar_lhs!($ty, bitor , BitOr , DeviceBitOrAPI , TensorBitOrAPI , DeviceRConsumeBitOrAPI );
        impl_arithmetic_scalar_lhs!($ty, bitand, BitAnd, DeviceBitAndAPI, TensorBitAndAPI, DeviceRConsumeBitAndAPI);
        impl_arithmetic_scalar_lhs!($ty, bitxor, BitXor, DeviceBitXorAPI, TensorBitXorAPI, DeviceRConsumeBitXorAPI);
        impl_arithmetic_scalar_lhs!($ty, shl   , Shl   , DeviceShlAPI   , TensorShlAPI   , DeviceRConsumeShlAPI   );
        impl_arithmetic_scalar_lhs!($ty, shr   , Shr   , DeviceShrAPI   , TensorShrAPI   , DeviceRConsumeShrAPI   );
    };
}

#[rustfmt::skip]
macro_rules! impl_arithmetic_scalar_lhs_bool {
    ($ty: ty) => {
        impl_arithmetic_scalar_lhs!($ty, bitor , BitOr , DeviceBitOrAPI , TensorBitOrAPI , DeviceRConsumeBitOrAPI );
        impl_arithmetic_scalar_lhs!($ty, bitand, BitAnd, DeviceBitAndAPI, TensorBitAndAPI, DeviceRConsumeBitAndAPI);
        impl_arithmetic_scalar_lhs!($ty, bitxor, BitXor, DeviceBitXorAPI, TensorBitXorAPI, DeviceRConsumeBitXorAPI);
    };
}

#[rustfmt::skip]
macro_rules! impl_arithmetic_scalar_lhs_float {
    ($ty: ty) => {
        impl_arithmetic_scalar_lhs!($ty, add   , Add   , DeviceAddAPI   , TensorAddAPI   , DeviceRConsumeAddAPI   );
        impl_arithmetic_scalar_lhs!($ty, sub   , Sub   , DeviceSubAPI   , TensorSubAPI   , DeviceRConsumeSubAPI   );
        impl_arithmetic_scalar_lhs!($ty, mul   , Mul   , DeviceMulAPI   , TensorMulAPI   , DeviceRConsumeMulAPI   );
        impl_arithmetic_scalar_lhs!($ty, div   , Div   , DeviceDivAPI   , TensorDivAPI   , DeviceRConsumeDivAPI   );
    };
}

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
impl_arithmetic_scalar_lhs_float!(Complex<f32>);
impl_arithmetic_scalar_lhs_float!(Complex<f64>);

/* #endregion */

/* #region tsr a op num b */

// for this case, core::ops::* is not required to be re-implemented
// see macro_rule `impl_core_ops`

macro_rules! impl_arithmetic_scalar_rhs {
    ($op: ident, $Op: ident, $DeviceOpAPI: ident, $TensorOpAPI: ident, $DeviceLConsumeOpAPI: ident) => {
        impl<T, TB, R, D, B> $TensorOpAPI<TB> for &TensorBase<R, D>
        where
            T: From<TB> + $Op<T, Output = T>,
            R: DataAPI<Data = Storage<T, B>>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
            // this constraint prohibits confliting impl to TensorBase<RB, D>
            TB: num::Num,
        {
            type Output = Tensor<T, D, B>;
            fn $op(a: Self, b: TB) -> Result<Self::Output> {
                let b = T::from(b);
                let device = a.device();
                let la = a.layout();
                let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                let storage_a = a.storage();
                device.op_mutc_refa_numb(&mut storage_c, &lc, storage_a, la, b)?;
                Tensor::new_f(DataOwned::from(storage_c), lc)
            }
        }

        impl<'l, T, TB, D, B> $TensorOpAPI<TB> for TensorView<'l, T, D, B>
        where
            T: From<TB> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, D>,
            // this constraint prohibits confliting impl to TensorBase<RB, D>
            TB: num::Num,
        {
            type Output = Tensor<T, D, B>;
            fn $op(a: Self, b: TB) -> Result<Self::Output> {
                let b = T::from(b);
                let device = a.device();
                let la = a.layout();
                let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
                let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
                let storage_a = a.storage();
                device.op_mutc_refa_numb(&mut storage_c, &lc, storage_a, la, b)?;
                Tensor::new_f(DataOwned::from(storage_c), lc)
            }
        }

        impl<T, TB, D, B> $TensorOpAPI<TB> for Tensor<T, D, B>
        where
            T: From<TB> + $Op<T, Output = T>,
            D: DimAPI,
            B: DeviceAPI<T>,
            B: $DeviceLConsumeOpAPI<T, T, D>,
            // this constraint prohibits confliting impl to TensorBase<RB, D>
            TB: num::Num,
        {
            type Output = Tensor<T, D, B>;
            fn $op(mut a: Self, b: TB) -> Result<Self::Output> {
                let b = T::from(b);
                let device = a.device().clone();
                let la = a.layout().clone();
                let storage_a = a.data_mut().storage_mut();
                device.op_muta_numb(storage_a, &la, b)?;
                return Ok(a);
            }
        }
    };
}

#[rustfmt::skip]
mod impl_arithmetic_scalar_rhs {
    use super::*;
    impl_arithmetic_scalar_rhs!(add   , Add   , DeviceAddAPI   , TensorAddAPI   , DeviceLConsumeAddAPI   );
    impl_arithmetic_scalar_rhs!(sub   , Sub   , DeviceSubAPI   , TensorSubAPI   , DeviceLConsumeSubAPI   );
    impl_arithmetic_scalar_rhs!(mul   , Mul   , DeviceMulAPI   , TensorMulAPI   , DeviceLConsumeMulAPI   );
    impl_arithmetic_scalar_rhs!(div   , Div   , DeviceDivAPI   , TensorDivAPI   , DeviceLConsumeDivAPI   );
    impl_arithmetic_scalar_rhs!(rem   , Rem   , DeviceRemAPI   , TensorRemAPI   , DeviceLConsumeRemAPI   );
    impl_arithmetic_scalar_rhs!(bitor , BitOr , DeviceBitOrAPI , TensorBitOrAPI , DeviceLConsumeBitOrAPI );
    impl_arithmetic_scalar_rhs!(bitand, BitAnd, DeviceBitAndAPI, TensorBitAndAPI, DeviceLConsumeBitAndAPI);
    impl_arithmetic_scalar_rhs!(bitxor, BitXor, DeviceBitXorAPI, TensorBitXorAPI, DeviceLConsumeBitXorAPI);
    impl_arithmetic_scalar_rhs!(shl   , Shl   , DeviceShlAPI   , TensorShlAPI   , DeviceLConsumeShlAPI   );
    impl_arithmetic_scalar_rhs!(shr   , Shr   , DeviceShrAPI   , TensorShrAPI   , DeviceLConsumeShrAPI   );
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        // b - &a
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = 1;
        let c = b - &a;
        let c_ref = vec![0., -1., -2., -3., -4.].into();
        assert!(allclose_f64(&c, &c_ref));

        // &a - b
        let a = Tensor::linspace((1.0, 5.0, 5));
        let b = 1;
        let c = &a - b;
        let c_ref = vec![0., 1., 2., 3., 4.].into();
        assert!(allclose_f64(&c, &c_ref));

        // b * a
        let a = Tensor::linspace((1.0, 5.0, 5));
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let b = 2;
        let c: Tensor<_, _> = -b * a;
        let c_ref = vec![-2., -4., -6., -8., -10.].into();
        assert!(allclose_f64(&c, &c_ref));
        let c_ptr = c.data().storage().rawvec().as_ptr();
        assert_eq!(a_ptr, c_ptr);
    }
}
