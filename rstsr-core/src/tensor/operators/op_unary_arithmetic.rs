//! Unary arithmetic operators: [`neg`](neg()) (operator `-`, arithmetic
//! negation), [`not`](not()) (operator `!`, logical/bitwise not for boolean
//! and integer tensors), and [`positive`](positive()) (identity function; no
//! rust-operator counterpart).
//!
//! # Examples
//!
//! ```rust
//! # use rstsr::prelude::*;
//! # let mut device = DeviceCpu::default();
//! # device.set_default_order(RowMajor);
//! let a = rt::tensor_from_nested!([-1, 2, -3], &device);
//! println!("{}", rt::neg(&a));
//! // [ 1 -2 3]
//! # assert_eq!(format!("{}", rt::neg(&a)), "[ 1 -2 3]");
//! ```

use crate::prelude_dev::*;

#[duplicate_item(
    op    op_f    TensorOpAPI    ;
   [neg] [neg_f] [TensorNegAPI];
   [not] [not_f] [TensorNotAPI];
   [positive] [positive_f] [TensorPositiveAPI];
)]
pub trait TensorOpAPI {
    type Output;
    fn op_f(self) -> Result<Self::Output>;
    fn op(self) -> Self::Output
    where
        Self: Sized,
    {
        Self::op_f(self).rstsr_unwrap()
    }
}

#[duplicate_item(
    op    op_f    TensorOpAPI    ;
   [neg] [neg_f] [TensorNegAPI];
   [not] [not_f] [TensorNotAPI];
   [positive] [positive_f] [TensorPositiveAPI];
)]
pub fn op_f<TRA, TRB>(a: TRA) -> Result<TRB>
where
    TRA: TensorOpAPI<Output = TRB>,
{
    TRA::op_f(a)
}

#[duplicate_item(
    op    op_f    TensorOpAPI    ;
   [neg] [neg_f] [TensorNegAPI];
   [not] [not_f] [TensorNotAPI];
   [positive] [positive_f] [TensorPositiveAPI];
)]
pub fn op<TRA, TRB>(a: TRA) -> TRB
where
    TRA: TensorOpAPI<Output = TRB>,
{
    TRA::op(a)
}

#[duplicate_item(
    op    op_f    TensorOpAPI    ;
   [neg] [neg_f] [TensorNegAPI];
   [not] [not_f] [TensorNotAPI];
   [positive] [positive_f] [TensorPositiveAPI];
)]
impl<S, D> TensorBase<S, D>
where
    D: DimAPI,
{
    pub fn op_f(&self) -> Result<<&Self as TensorOpAPI>::Output>
    where
        for<'a> &'a Self: TensorOpAPI,
    {
        op_f(self)
    }

    pub fn op(&self) -> <&Self as TensorOpAPI>::Output
    where
        for<'a> &'a Self: TensorOpAPI,
    {
        op(self)
    }
}

#[duplicate_item(
    op    Op    TensorOpAPI    ;
   [neg] [Neg] [TensorNegAPI];
   [not] [Not] [TensorNotAPI];
)]
mod impl_unary_core_ops {
    use super::*;

    impl<R, T, B, D> Op for &TensorAny<R, T, B, D>
    where
        D: DimAPI,
        for<'a> &'a TensorAny<R, T, B, D>: TensorOpAPI,
        R: DataAPI<Data = B::Raw>,
        B: DeviceAPI<T>,
    {
        type Output = <Self as TensorOpAPI>::Output;
        fn op(self) -> Self::Output {
            TensorOpAPI::op(self)
        }
    }

    impl<R, T, B, D> Op for TensorAny<R, T, B, D>
    where
        D: DimAPI,
        TensorAny<R, T, B, D>: TensorOpAPI,
        R: DataAPI<Data = B::Raw>,
        B: DeviceAPI<T>,
    {
        type Output = <Self as TensorOpAPI>::Output;
        fn op(self) -> Self::Output {
            TensorOpAPI::op(self)
        }
    }
}

#[duplicate_item(
    op_f    Op    TensorOpAPI      OpAPI    ;
   [neg_f] [Neg] [TensorNegAPI] [OpNegAPI];
   [not_f] [Not] [TensorNotAPI] [OpNotAPI];
)]
mod impl_unary {
    use super::*;

    #[doc(hidden)]
    impl<R, T, TB, D, B> TensorOpAPI for &TensorAny<R, TB, B, D>
    where
        D: DimAPI,
        R: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        B: DeviceAPI<T>,
        TB: Op<Output = T>,
        B: OpAPI<T, TB, D> + DeviceCreationAnyAPI<T>,
    {
        type Output = Tensor<T, B, D>;
        fn op_f(self) -> Result<Self::Output> {
            let lb = self.layout();
            // generate empty output tensor
            let device = self.device();
            let la = layout_for_array_copy(lb, TensorIterOrder::K)?;
            let mut storage_a = device.uninit_impl(la.bounds_index()?.1)?;
            // compute and return
            device.op_muta_refb(storage_a.raw_mut(), &la, self.raw(), lb)?;
            // SAFETY: the op above wrote every element of the fresh `storage_a`.
            let storage_a = unsafe { B::assume_init_impl(storage_a) }?;
            return Tensor::new_f(storage_a, la);
        }
    }

    #[doc(hidden)]
    impl<T, TB, D, B> TensorOpAPI for TensorView<'_, TB, B, D>
    where
        D: DimAPI,
        B: DeviceAPI<T>,
        TB: Op<Output = T>,
        B: OpAPI<T, TB, D> + DeviceCreationAnyAPI<T>,
    {
        type Output = Tensor<T, B, D>;
        fn op_f(self) -> Result<Self::Output> {
            TensorOpAPI::op_f(&self)
        }
    }

    #[doc(hidden)]
    impl<T, B, D> TensorOpAPI for Tensor<T, B, D>
    where
        D: DimAPI,
        B: DeviceAPI<T>,
        T: Op<Output = T>,
        B: OpAPI<T, T, D> + DeviceCreationAnyAPI<T>,
    {
        type Output = Tensor<T, B, D>;
        fn op_f(mut self) -> Result<Self::Output> {
            if self.layout().is_broadcasted() {
                // an owned broadcasted tensor cannot be negated in place
                // (elements alias); fall back to producing a fresh packed
                // output instead, same policy as binary op reuse
                return TensorOpAPI::op_f(&self);
            }
            let layout = self.layout().clone();
            let device = self.device().clone();
            // generate empty output tensor
            device.op_muta(self.raw_mut(), &layout)?;
            return Ok(self);
        }
    }
}

#[doc(hidden)]
mod impl_unary_positive {
    use super::*;

    // `positive` follows the `neg`/`not` machinery (device kernel behind
    // `OpPositiveAPI`), but with no operator trait bound on the element type:
    // the kernel only clones, and the in-place form is the identity.

    #[doc(hidden)]
    impl<R, T, B, D> TensorPositiveAPI for &TensorAny<R, T, B, D>
    where
        D: DimAPI,
        R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        T: Clone,
        B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpPositiveAPI<T, D>,
    {
        type Output = Tensor<T, B, D>;
        fn positive_f(self) -> Result<Self::Output> {
            let lb = self.layout();
            // generate empty output tensor
            let device = self.device();
            let la = layout_for_array_copy(lb, TensorIterOrder::K)?;
            let mut storage_a = device.uninit_impl(la.bounds_index()?.1)?;
            // compute and return
            device.op_muta_refb(storage_a.raw_mut(), &la, self.raw(), lb)?;
            // SAFETY: the op above wrote every element of the fresh `storage_a`.
            let storage_a = unsafe { B::assume_init_impl(storage_a) }?;
            return Tensor::new_f(storage_a, la);
        }
    }

    #[doc(hidden)]
    impl<T, B, D> TensorPositiveAPI for TensorView<'_, T, B, D>
    where
        D: DimAPI,
        T: Clone,
        B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpPositiveAPI<T, D>,
    {
        type Output = Tensor<T, B, D>;
        fn positive_f(self) -> Result<Self::Output> {
            TensorPositiveAPI::positive_f(&self)
        }
    }

    #[doc(hidden)]
    impl<T, B, D> TensorPositiveAPI for Tensor<T, B, D>
    where
        D: DimAPI,
        B: DeviceAPI<T>,
    {
        type Output = Tensor<T, B, D>;
        fn positive_f(self) -> Result<Self::Output> {
            // the identity in place is a no-op: an owned tensor is returned
            // as-is, without a device call or a copy
            Ok(self)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_neg() {
        let a = linspace((1.0, 5.0, 5));
        let b = -&a;
        let b_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&b, &b_ref));
        let b = -a;
        let b_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&b, &b_ref));
    }

    #[test]
    fn test_positive() {
        let a = linspace((1.0, 5.0, 5));
        let b_ref = vec![1., 2., 3., 4., 5.].into();
        // borrowed input: fresh owned copy
        let b = positive(&a);
        assert!(allclose_f64(&b, &b_ref));
        assert_ne!(a.raw().as_ptr(), b.raw().as_ptr());
        // view input: fresh owned copy as well
        let c = positive(a.view());
        assert!(allclose_f64(&c, &b_ref));
        // owned input: identity in place, no copy
        let ptr_a = a.raw().as_ptr();
        let d = positive(a);
        assert!(allclose_f64(&d, &b_ref));
        assert_eq!(ptr_a, d.raw().as_ptr());
    }

    #[test]
    fn test_neg_broadcast_owned_fallback() {
        // an owned broadcasted tensor cannot be negated in place (elements
        // alias); the op falls back to a fresh output instead of erroring
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let a = arange((3.0, &device));
        let (storage, _) = a.into_raw_parts();
        let c = Tensor::new(storage, Layout::new([2, 3], [0, 1], 0).unwrap());
        let d = -c;
        let v: Vec<_> = d.view().iter().cloned().collect();
        assert_eq!(v, vec![-0., -1., -2., -0., -1., -2.]);
    }
}
