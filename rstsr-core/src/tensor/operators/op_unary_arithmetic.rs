use crate::prelude_dev::*;

macro_rules! trait_unary {
    ($op: ident, $op_f: ident, $TensorOpAPI: ident) => {
        pub trait $TensorOpAPI {
            type Output;
            fn $op_f(self) -> Result<Self::Output>;
            fn $op(self) -> Self::Output
            where
                Self: Sized,
            {
                Self::$op_f(self).unwrap()
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

        impl<S, D> TensorBase<S, D>
        where
            D: DimAPI,
        {
            pub fn $op_f(&self) -> Result<<&Self as $TensorOpAPI>::Output>
            where
                for<'a> &'a Self: $TensorOpAPI,
            {
                $op_f(self)
            }

            pub fn $op(&self) -> <&Self as $TensorOpAPI>::Output
            where
                for<'a> &'a Self: $TensorOpAPI,
            {
                $op(self)
            }
        }
    };
}

#[rustfmt::skip]
mod trait_unary {
    use super::*;
    trait_unary!(neg, neg_f, TensorNegAPI);
    trait_unary!(not, not_f, TensorNotAPI);
}
pub use trait_unary::*;

macro_rules! impl_unary_core_ops {
    ($op: ident, $Op: ident, $TensorOpAPI: ident) => {
        impl<R, T, B, D> $Op for &TensorAny<R, T, B, D>
        where
            D: DimAPI,
            for<'a> &'a TensorAny<R, T, B, D>: $TensorOpAPI,
            R: DataAPI<Data = B::Raw>,
            B: DeviceAPI<T>,
        {
            type Output = <Self as $TensorOpAPI>::Output;
            fn $op(self) -> Self::Output {
                $TensorOpAPI::$op(self)
            }
        }

        impl<R, T, B, D> $Op for TensorAny<R, T, B, D>
        where
            D: DimAPI,
            TensorAny<R, T, B, D>: $TensorOpAPI,
            R: DataAPI<Data = B::Raw>,
            B: DeviceAPI<T>,
        {
            type Output = <Self as $TensorOpAPI>::Output;
            fn $op(self) -> Self::Output {
                $TensorOpAPI::$op(self)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_unary_core_ops {
    use super::*;
    impl_unary_core_ops!(neg, Neg, TensorNegAPI);
    impl_unary_core_ops!(not, Not, TensorNotAPI);
}

macro_rules! impl_unary {
    ($op_f: ident, $Op: ident, $TensorOpAPI: ident, $DeviceOpAPI: ident) => {
        #[doc(hidden)]
        impl<R, T, TB, D, B> $TensorOpAPI for &TensorAny<R, TB, B, D>
        where
            D: DimAPI,
            R: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
            B: DeviceAPI<T>,
            TB: $Op<Output = T>,
            B: $DeviceOpAPI<T, TB, D> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, B, D>;
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

        #[doc(hidden)]
        impl<'l, T, TB, D, B> $TensorOpAPI for TensorView<'l, TB, B, D>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            TB: $Op<Output = T>,
            B: $DeviceOpAPI<T, TB, D> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, B, D>;
            fn $op_f(self) -> Result<Self::Output> {
                $TensorOpAPI::$op_f(&self)
            }
        }

        #[doc(hidden)]
        impl<T, B, D> $TensorOpAPI for Tensor<T, B, D>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            T: $Op<Output = T>,
            B: $DeviceOpAPI<T, T, D> + DeviceCreationAnyAPI<T>,
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
mod impl_unary {
    use super::*;
    impl_unary!(neg_f, Neg, TensorNegAPI, DeviceNegAPI);
    impl_unary!(not_f, Not, TensorNotAPI, DeviceNotAPI);
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
}
