use crate::feature_rayon::reduction::{reduce_all_cpu_rayon, reduce_axes_cpu_rayon};
use crate::prelude_dev::*;
use core::ops::Add;
use num::Zero;

impl<T, D> OpSumAPI<T, D> for DeviceFaer
where
    T: Zero + Add<Output = T> + Clone + Send + Sync,
    D: DimAPI,
{
    fn sum_all(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T> {
        let nthreads = self.get_num_threads();
        reduce_all_cpu_rayon(a.rawvec(), la, T::zero, |acc, x| acc + x, nthreads)
    }

    fn sum(
        &self,
        a: &Storage<T, Self>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<T, Self>, Layout<IxD>)> {
        let nthreads = self.get_num_threads();
        let (out, layout_out) = reduce_axes_cpu_rayon(
            a.rawvec(),
            &la.to_dim()?,
            axes,
            T::zero,
            |acc, x| acc + x,
            nthreads,
        )?;
        Ok((Storage::new(out, self.clone()), layout_out))
    }
}
