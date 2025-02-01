#[macro_export]
macro_rules! macro_impl_rayon_reduction {
    ($Device: ident) => {
        use core::ops::{Add, Mul};
        use num::complex::ComplexFloat;
        use num::{Bounded, FromPrimitive, One, Zero};
        use rstsr_dtype_traits::MinMaxAPI;
        use $crate::feature_rayon::*;
        use $crate::prelude_dev::*;

        impl<T, D> OpSumAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + Zero + Add<Output = T>,
            D: DimAPI,
        {
            fn sum_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
                let pool = self.get_pool();
                reduce_all_cpu_rayon(a, la, T::zero, |acc, x| acc + x, |acc| acc, pool)
            }

            fn sum(
                &self,
                a: &Vec<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
                let pool = self.get_pool();
                let (out, layout_out) = reduce_axes_cpu_rayon(
                    a,
                    &la.to_dim()?,
                    axes,
                    T::zero,
                    |acc, x| acc + x,
                    |acc| acc,
                    pool,
                )?;
                Ok((Storage::new(out.into(), self.clone()), layout_out))
            }
        }

        impl<T, D> OpMinAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + MinMaxAPI + Bounded,
            D: DimAPI,
        {
            fn min_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
                let pool = self.get_pool();
                if la.size() == 0 {
                    rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
                }
                reduce_all_cpu_rayon(a, la, || T::max_value(), |acc, x| acc.min(x), |acc| acc, pool)
            }

            fn min(
                &self,
                a: &Vec<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
                let pool = self.get_pool();
                if la.size() == 0 {
                    rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
                }
                let (out, layout_out) = reduce_axes_cpu_rayon(
                    a,
                    &la.to_dim()?,
                    axes,
                    || T::max_value(),
                    |acc, x| acc.min(x),
                    |acc| acc,
                    pool,
                )?;
                Ok((Storage::new(out.into(), self.clone()), layout_out))
            }
        }

        impl<T, D> OpMaxAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + MinMaxAPI + Bounded,
            D: DimAPI,
        {
            fn max_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
                let pool = self.get_pool();
                if la.size() == 0 {
                    rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
                }
                reduce_all_cpu_rayon(a, la, || T::min_value(), |acc, x| acc.max(x), |acc| acc, pool)
            }

            fn max(
                &self,
                a: &Vec<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
                let pool = self.get_pool();
                if la.size() == 0 {
                    rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
                }
                let (out, layout_out) = reduce_axes_cpu_rayon(
                    a,
                    &la.to_dim()?,
                    axes,
                    || T::min_value(),
                    |acc, x| acc.max(x),
                    |acc| acc,
                    pool,
                )?;
                Ok((Storage::new(out.into(), self.clone()), layout_out))
            }
        }

        impl<T, D> OpProdAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + One + Mul<Output = T>,
            D: DimAPI,
        {
            fn prod_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
                let pool = self.get_pool();
                reduce_all_cpu_rayon(a, la, T::one, |acc, x| acc * x, |acc| acc, pool)
            }

            fn prod(
                &self,
                a: &Vec<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
                let pool = self.get_pool();
                let (out, layout_out) = reduce_axes_cpu_rayon(
                    a,
                    &la.to_dim()?,
                    axes,
                    T::one,
                    |acc, x| acc * x,
                    |acc| acc,
                    pool,
                )?;
                Ok((Storage::new(out.into(), self.clone()), layout_out))
            }
        }

        impl<T, D> OpMeanAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + ComplexFloat + FromPrimitive,
            D: DimAPI,
        {
            fn mean_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
                let pool = self.get_pool();
                let size = la.size();
                let sum = reduce_all_cpu_rayon(
                    a,
                    la,
                    T::zero,
                    |acc, x| acc + x,
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                Ok(sum)
            }

            fn mean(
                &self,
                a: &Vec<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
                let pool = self.get_pool();
                let (layout_axes, _) = la.dim_split_axes(axes)?;
                let size = layout_axes.size();
                let (out, layout_out) = reduce_axes_cpu_rayon(
                    a,
                    &la.to_dim()?,
                    axes,
                    T::zero,
                    |acc, x| acc + x,
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                Ok((Storage::new(out.into(), self.clone()), layout_out))
            }
        }

        impl<T, D> OpVarAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + ComplexFloat + FromPrimitive,
            D: DimAPI,
        {
            fn var_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
                let pool = self.get_pool();
                let size = la.size();
                let e_x1 = reduce_all_cpu_rayon(
                    a,
                    la,
                    T::zero,
                    |acc, x| acc + x,
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                let e_x2 = reduce_all_cpu_rayon(
                    a,
                    la,
                    T::zero,
                    |acc, x| acc + x.clone() * x.clone(),
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                Ok(e_x2 - e_x1.clone() * e_x1.clone())
            }

            fn var(
                &self,
                a: &Vec<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
                let pool = self.get_pool();
                let (layout_axes, _) = la.dim_split_axes(axes)?;
                let size = layout_axes.size();
                let (mut e_x1, layout_out) = reduce_axes_cpu_rayon(
                    a,
                    &la.to_dim()?,
                    axes,
                    T::zero,
                    |acc, x| acc + x,
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                let (e_x2, _) = reduce_axes_cpu_rayon(
                    a,
                    &la.to_dim()?,
                    axes,
                    T::zero,
                    |acc, x| acc + x.clone() * x.clone(),
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                e_x1.iter_mut()
                    .zip(e_x2.iter())
                    .for_each(|(x1, x2)| *x1 = x2.clone() - x1.clone() * x1.clone());
                Ok((Storage::new(e_x1.into(), self.clone()), layout_out))
            }
        }

        impl<T, D> OpStdAPI<T, D> for $Device
        where
            T: Clone + Send + Sync + ComplexFloat + FromPrimitive,
            D: DimAPI,
        {
            fn std_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
                let pool = self.get_pool();
                let size = la.size();
                let e_x1 = reduce_all_cpu_rayon(
                    a,
                    la,
                    T::zero,
                    |acc, x| acc + x,
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                let e_x2 = reduce_all_cpu_rayon(
                    a,
                    la,
                    T::zero,
                    |acc, x| acc + x.clone() * x.clone(),
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                Ok((e_x2 - e_x1.clone() * e_x1.clone()).sqrt())
            }

            fn std(
                &self,
                a: &Vec<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
                let pool = self.get_pool();
                let (layout_axes, _) = la.dim_split_axes(axes)?;
                let size = layout_axes.size();
                let (mut e_x1, layout_out) = reduce_axes_cpu_rayon(
                    a,
                    &la.to_dim()?,
                    axes,
                    T::zero,
                    |acc, x| acc + x,
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                let (e_x2, _) = reduce_axes_cpu_rayon(
                    a,
                    &la.to_dim()?,
                    axes,
                    T::zero,
                    |acc, x| acc + x.clone() * x.clone(),
                    |acc| acc / T::from_usize(size).unwrap(),
                    pool,
                )?;
                e_x1.iter_mut()
                    .zip(e_x2.iter())
                    .for_each(|(x1, x2)| *x1 = (x2.clone() - x1.clone() * x1.clone()).sqrt());
                Ok((Storage::new(e_x1.into(), self.clone()), layout_out))
            }
        }
    };
}
