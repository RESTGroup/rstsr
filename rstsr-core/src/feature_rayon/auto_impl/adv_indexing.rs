use crate::prelude_dev::*;

impl<T, D> DeviceIndexSelectAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    fn index_select(
        &self,
        c: &mut <Self as DeviceRawAPI<T>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: usize,
        indices: &[usize],
    ) -> Result<()> {
        let pool = self.get_current_pool();
        index_select_cpu_rayon(c, lc, a, la, axis, indices, pool)
    }
}
