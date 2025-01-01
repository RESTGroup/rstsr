use crate::prelude_dev::*;
use core::ops::*;

macro_rules! trait_unary {
    ($op: ident, $TensorOpAPI: ident) => {
        pub trait $TensorOpAPI {
            type Output;
            fn $op(self) -> Result<Self::Output>;
        }

        pub fn $op<TRA, TRB>(a: TRA) -> Result<TRB>
        where
            TRA: $TensorOpAPI<Output = TRB>,
        {
            TRA::$op(a)
        }
    };
}

#[rustfmt::skip]
mod trait_unary {
    use super::*;
    trait_unary!(neg, TensorNegAPI);
    trait_unary!(not, TensorNotAPI);
}
pub use trait_unary::*;

macro_rules! impl_unary_core_ops {
    ($op: ident, $Op: ident, $TensorOpAPI: ident) => {
        impl<R, D> $Op for &TensorBase<R, D>
        where
            D: DimAPI,
            for<'a> &'a TensorBase<R, D>: $TensorOpAPI,
        {
            type Output = <Self as $TensorOpAPI>::Output;
            fn $op(self) -> Self::Output {
                $TensorOpAPI::$op(self).unwrap()
            }
        }

        impl<R, D> $Op for TensorBase<R, D>
        where
            D: DimAPI,
            TensorBase<R, D>: $TensorOpAPI,
        {
            type Output = <Self as $TensorOpAPI>::Output;
            fn $op(self) -> Self::Output {
                $TensorOpAPI::$op(self).unwrap()
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
    ($op: ident, $Op: ident, $TensorOpAPI: ident, $DeviceOpAPI: ident) => {
        impl<R, T, TB, D, B> $TensorOpAPI for &TensorBase<R, D>
        where
            D: DimAPI,
            R: DataAPI<Data = Storage<TB, B>>,
            B: DeviceAPI<T>,
            TB: $Op<Output = T>,
            B: $DeviceOpAPI<T, TB, D> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(self) -> Result<Self::Output> {
                let lb = self.layout();
                let storage_b = self.data().storage();
                // generate empty output tensor
                let device = self.device();
                let la = layout_for_array_copy(lb, TensorIterOrder::K)?;
                let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
                // compute and return
                device.op_muta_refb(&mut storage_a, &la, storage_b, lb)?;
                return Tensor::new_f(DataOwned::from(storage_a), la);
            }
        }

        impl<'l, T, TB, D, B> $TensorOpAPI for TensorView<'l, TB, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            TB: $Op<Output = T>,
            B: $DeviceOpAPI<T, TB, D> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(self) -> Result<Self::Output> {
                $TensorOpAPI::$op(&self)
            }
        }

        impl<T, D, B> $TensorOpAPI for Tensor<T, D, B>
        where
            D: DimAPI,
            B: DeviceAPI<T>,
            T: $Op<Output = T>,
            B: $DeviceOpAPI<T, T, D> + DeviceCreationAnyAPI<T>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(mut self) -> Result<Self::Output> {
                let layout = self.layout().clone();
                let device = self.device().clone();
                let storage = self.data_mut().storage_mut();
                // generate empty output tensor
                device.op_muta(storage, &layout)?;
                return Ok(self);
            }
        }
    };
}

#[rustfmt::skip]
mod impl_unary {
    use super::*;
    impl_unary!(neg, Neg, TensorNegAPI, DeviceNegAPI);
    impl_unary!(not, Not, TensorNotAPI, DeviceNotAPI);
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
