use crate::prelude_dev::*;
use core::ops::{Add, Mul};
use num::complex::ComplexFloat;
use num::{Bounded, FromPrimitive, One, Zero};
use rstsr_dtype_traits::MinMaxAPI;

impl<T, D> OpSumAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + Zero + Add<Output = T>,
    D: DimAPI,
{
    type TOut = T;

    fn sum_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        let pool = self.get_current_pool();

        let f_init = T::zero;
        let f = |acc, x| acc + x;
        let f_sum = |acc1, acc2| acc1 + acc2;
        let f_out = |acc| acc;

        reduce_all_cpu_rayon(a, la, f_init, f, f_sum, f_out, pool)
    }

    fn sum_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let f_init = T::zero;
        let f = |acc, x| acc + x;
        let f_sum = |acc1, acc2| acc1 + acc2;
        let f_out = |acc| acc;

        let (out, layout_out) =
            reduce_axes_cpu_rayon(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out, pool)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpMinAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + MinMaxAPI + Bounded,
    D: DimAPI,
{
    type TOut = T;

    fn min_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
        }

        let pool = self.get_current_pool();

        let f_init = T::max_value;
        let f = |acc: T, x: T| acc.min(x);
        let f_sum = |acc1: T, acc2: T| acc1.min(acc2);
        let f_out = |acc| acc;

        reduce_all_cpu_rayon(a, la, f_init, f, f_sum, f_out, pool)
    }

    fn min_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
        }

        let pool = self.get_current_pool();

        let f_init = T::max_value;
        let f = |acc: T, x: T| acc.min(x);
        let f_sum = |acc1: T, acc2: T| acc1.min(acc2);
        let f_out = |acc| acc;

        let (out, layout_out) =
            reduce_axes_cpu_rayon(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out, pool)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpMaxAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + MinMaxAPI + Bounded,
    D: DimAPI,
{
    type TOut = T;

    fn max_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
        }

        let pool = self.get_current_pool();

        let f_init = T::min_value;
        let f = |acc: T, x: T| acc.max(x);
        let f_sum = |acc1: T, acc2: T| acc1.max(acc2);
        let f_out = |acc| acc;

        reduce_all_cpu_rayon(a, la, f_init, f, f_sum, f_out, pool)
    }

    fn max_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
        }

        let pool = self.get_current_pool();

        let f_init = T::min_value;
        let f = |acc: T, x: T| acc.max(x);
        let f_sum = |acc1: T, acc2: T| acc1.max(acc2);
        let f_out = |acc| acc;

        let (out, layout_out) =
            reduce_axes_cpu_rayon(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out, pool)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpProdAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + One + Mul<Output = T>,
    D: DimAPI,
{
    type TOut = T;

    fn prod_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        let pool = self.get_current_pool();

        let f_init = T::one;
        let f = |acc, x| acc * x;
        let f_sum = |acc1, acc2| acc1 * acc2;
        let f_out = |acc| acc;

        reduce_all_cpu_rayon(a, la, f_init, f, f_sum, f_out, pool)
    }

    fn prod_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let f_init = T::one;
        let f = |acc, x| acc * x;
        let f_sum = |acc1, acc2| acc1 * acc2;
        let f_out = |acc| acc;

        let (out, layout_out) =
            reduce_axes_cpu_rayon(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out, pool)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpMeanAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ComplexFloat + FromPrimitive,
    D: DimAPI,
{
    type TOut = T;

    fn mean_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        let pool = self.get_current_pool();

        let size = la.size();
        let f_init = T::zero;
        let f = |acc, x| acc + x;
        let f_sum = |acc, x| acc + x;
        let f_out = |acc| acc / T::from_usize(size).unwrap();

        let sum = reduce_all_cpu_rayon(a, la, f_init, f, f_sum, f_out, pool)?;
        Ok(sum)
    }

    fn mean_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let (layout_axes, _) = la.dim_split_axes(axes)?;
        let size = layout_axes.size();
        let f_init = T::zero;
        let f = |acc, x| acc + x;
        let f_sum = |acc, x| acc + x;
        let f_out = |acc| acc / T::from_usize(size).unwrap();

        let (out, layout_out) =
            reduce_axes_cpu_rayon(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out, pool)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpVarAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ComplexFloat + FromPrimitive,
    T::Real: Clone + Send + Sync + ComplexFloat + FromPrimitive,
    D: DimAPI,
{
    type TOut = T::Real;

    fn var_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T::Real> {
        let pool = self.get_current_pool();

        let size = la.size();

        let f_init = || (T::zero(), T::Real::zero());
        let f = |(acc_1, acc_2): (T, T::Real), x: T| (acc_1 + x, acc_2 + (x * x.conj()).re());
        let f_sum = |(acc_1, acc_2): (T, T::Real), (x_1, x_2)| (acc_1 + x_1, acc_2 + x_2);
        let f_out = |(acc_1, acc_2): (T, T::Real)| {
            let size_1 = T::from_usize(size).unwrap();
            let size_2 = T::Real::from_usize(size).unwrap();
            let mean = acc_1 / size_1;
            acc_2 / size_2 - (mean * mean.conj()).re()
        };

        let result = reduce_all_cpu_rayon(a, la, f_init, f, f_sum, f_out, pool)?;
        Ok(result)
    }

    fn var_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T::Real>>, T::Real, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let (layout_axes, _) = la.dim_split_axes(axes)?;
        let size = layout_axes.size();

        let f_init = || (T::zero(), T::Real::zero());
        let f = |(acc_1, acc_2): (T, T::Real), x: T| (acc_1 + x, acc_2 + (x * x.conj()).re());
        let f_sum = |(acc_1, acc_2): (T, T::Real), (x_1, x_2)| (acc_1 + x_1, acc_2 + x_2);
        let f_out = |(acc_1, acc_2): (T, T::Real)| {
            let size_1 = T::from_usize(size).unwrap();
            let size_2 = T::Real::from_usize(size).unwrap();
            let mean = acc_1 / size_1;
            acc_2 / size_2 - (mean * mean.conj()).re()
        };

        let (out, layout_out) =
            reduce_axes_difftype_cpu_rayon(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out, pool)?;

        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpStdAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ComplexFloat + FromPrimitive,
    T::Real: Clone + Send + Sync + ComplexFloat + FromPrimitive,
    D: DimAPI,
{
    type TOut = T::Real;

    fn std_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T::Real> {
        let pool = self.get_current_pool();

        let size = la.size();

        let f_init = || (T::zero(), T::Real::zero());
        let f = |(acc_1, acc_2): (T, T::Real), x: T| (acc_1 + x, acc_2 + (x * x.conj()).re());
        let f_sum = |(acc_1, acc_2): (T, T::Real), (x_1, x_2)| (acc_1 + x_1, acc_2 + x_2);
        let f_out = |(acc_1, acc_2): (T, T::Real)| {
            let size_1 = T::from_usize(size).unwrap();
            let size_2 = T::Real::from_usize(size).unwrap();
            let mean = acc_1 / size_1;
            let var = acc_2 / size_2 - (mean * mean.conj()).re();
            var.sqrt()
        };

        let result = reduce_all_cpu_rayon(a, la, f_init, f, f_sum, f_out, pool)?;
        Ok(result)
    }

    fn std_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T::Real>>, T::Real, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let (layout_axes, _) = la.dim_split_axes(axes)?;
        let size = layout_axes.size();

        let f_init = || (T::zero(), T::Real::zero());
        let f = |(acc_1, acc_2): (T, T::Real), x: T| (acc_1 + x, acc_2 + (x * x.conj()).re());
        let f_sum = |(acc_1, acc_2): (T, T::Real), (x_1, x_2)| (acc_1 + x_1, acc_2 + x_2);
        let f_out = |(acc_1, acc_2): (T, T::Real)| {
            let size_1 = T::from_usize(size).unwrap();
            let size_2 = T::Real::from_usize(size).unwrap();
            let mean = acc_1 / size_1;
            let var = acc_2 / size_2 - (mean * mean.conj()).re();
            var.sqrt()
        };

        let (out, layout_out) =
            reduce_axes_difftype_cpu_rayon(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out, pool)?;

        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpL2NormAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ComplexFloat + FromPrimitive,
    T::Real: Clone + Send + Sync + ComplexFloat + FromPrimitive,
    D: DimAPI,
{
    type TOut = T::Real;

    fn l2_norm_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T::Real> {
        let pool = self.get_current_pool();

        let f_init = || T::Real::zero();
        let f = |acc: T::Real, x: T| acc + (x * x.conj()).re();
        let f_sum = |acc: T::Real, x: T::Real| acc + x;
        let f_out = |acc: T::Real| acc.sqrt();

        let result = reduce_all_cpu_rayon(a, la, f_init, f, f_sum, f_out, pool)?;
        Ok(result)
    }

    fn l2_norm_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T::Real>>, T::Real, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let f_init = || T::Real::zero();
        let f = |acc: T::Real, x: T| acc + (x * x.conj()).re();
        let f_sum = |acc: T::Real, x: T::Real| acc + x;
        let f_out = |acc: T::Real| acc.sqrt();

        let (out, layout_out) =
            reduce_axes_cpu_rayon(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out, pool)?;

        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpArgMinAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + PartialOrd + Send + Sync,
    D: DimAPI,
{
    type TOut = usize;

    fn argmin_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<usize>>, Self::TOut, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let f_comp = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y < x)
            } else {
                Some(true)
            }
        };
        let f_eq = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y == x)
            } else {
                Some(false)
            }
        };
        let (out, layout_out) =
            reduce_axes_arg_cpu_rayon(a, la, axes, f_comp, f_eq, RowMajor, pool)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }

    fn argmin_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<Self::TOut> {
        let pool = self.get_current_pool();

        let f_comp = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y < x)
            } else {
                Some(true)
            }
        };
        let f_eq = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y == x)
            } else {
                Some(false)
            }
        };
        let result = reduce_all_arg_cpu_rayon(a, la, f_comp, f_eq, RowMajor, pool)?;
        Ok(result)
    }
}

impl<T, D> OpArgMaxAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + PartialOrd + Send + Sync,
    D: DimAPI,
{
    type TOut = usize;

    fn argmax_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<usize>>, Self::TOut, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let f_comp = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y > x)
            } else {
                Some(true)
            }
        };
        let f_eq = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y == x)
            } else {
                Some(false)
            }
        };
        let (out, layout_out) =
            reduce_axes_arg_cpu_rayon(a, la, axes, f_comp, f_eq, RowMajor, pool)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }

    fn argmax_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<Self::TOut> {
        let pool = self.get_current_pool();

        let f_comp = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y > x)
            } else {
                Some(true)
            }
        };
        let f_eq = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y == x)
            } else {
                Some(false)
            }
        };
        let result = reduce_all_arg_cpu_rayon(a, la, f_comp, f_eq, RowMajor, pool)?;
        Ok(result)
    }
}

impl<T, D> OpUnraveledArgMinAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + PartialOrd + Send + Sync,
    D: DimAPI,
{
    fn unraveled_argmin_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<IxD>>, IxD, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let f_comp = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y < x)
            } else {
                Some(true)
            }
        };
        let f_eq = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y == x)
            } else {
                Some(false)
            }
        };
        let (out, layout_out) =
            reduce_axes_unraveled_arg_cpu_rayon(a, la, axes, f_comp, f_eq, pool)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }

    fn unraveled_argmin_all(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
    ) -> Result<D> {
        let pool = self.get_current_pool();

        let f_comp = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y < x)
            } else {
                Some(true)
            }
        };
        let f_eq = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y == x)
            } else {
                Some(false)
            }
        };
        let result = reduce_all_unraveled_arg_cpu_rayon(a, la, f_comp, f_eq, pool)?;
        Ok(result)
    }
}

impl<T, D> OpUnraveledArgMaxAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + PartialOrd + Send + Sync,
    D: DimAPI,
{
    fn unraveled_argmax_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<IxD>>, IxD, Self>, Layout<IxD>)> {
        let pool = self.get_current_pool();

        let f_comp = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y > x)
            } else {
                Some(true)
            }
        };
        let f_eq = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y == x)
            } else {
                Some(false)
            }
        };
        let (out, layout_out) =
            reduce_axes_unraveled_arg_cpu_rayon(a, la, axes, f_comp, f_eq, pool)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }

    fn unraveled_argmax_all(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
    ) -> Result<D> {
        let pool = self.get_current_pool();

        let f_comp = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y > x)
            } else {
                Some(true)
            }
        };
        let f_eq = |x: Option<T>, y: T| -> Option<bool> {
            if let Some(x) = x {
                Some(y == x)
            } else {
                Some(false)
            }
        };
        let result = reduce_all_unraveled_arg_cpu_rayon(a, la, f_comp, f_eq, pool)?;
        Ok(result)
    }
}
