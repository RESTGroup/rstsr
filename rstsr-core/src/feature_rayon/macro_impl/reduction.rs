#[macro_export]
macro_rules! macro_impl_rayon_reduction {
    ($Device: ident) => {
        use core::ops::Add;
        use num::Zero;
        use $crate::feature_rayon::*;
        use $crate::prelude_dev::*;

        impl<T, D> OpSumAPI<T, D> for $Device
        where
            T: Zero + Add<Output = T> + Clone + Send + Sync,
            D: DimAPI,
        {
            fn sum_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
                let pool = self.get_pool();
                reduce_all_cpu_rayon(a, la, T::zero, |acc, x| acc + x, pool)
            }

            fn sum(
                &self,
                a: &Vec<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
                let pool = self.get_pool();
                let (out, layout_out) =
                    reduce_axes_cpu_rayon(a, &la.to_dim()?, axes, T::zero, |acc, x| acc + x, pool)?;
                let storage = self.outof_cpu_vec(out)?;
                Ok((storage, layout_out))
            }
        }
    };
}
