use crate::feature_rayon::reduce_all_cpu_rayon;
use crate::prelude_dev::*;
use core::ops::Add;
use num::Zero;

impl<T, D> OpSumAPI<T, D> for DeviceFaer
where
    T: Zero + Add<Output = T> + Clone + Send + Sync,
    D: DimAPI,
{
    fn sum_all(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T> {
        let a = a.rawvec();
        let nthreads = self.get_num_threads();
        reduce_all_cpu_rayon(a, la, T::zero, |acc, x| acc + x, nthreads)
    }

    fn sum(
        &self,
        _a: &Storage<T, Self>,
        _la: &Layout<D>,
        _axes: &[isize],
    ) -> Result<(Storage<T, Self>, Layout<IxD>)> {
        unimplemented!()
    }
}
