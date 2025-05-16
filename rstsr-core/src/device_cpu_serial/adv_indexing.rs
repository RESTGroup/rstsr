use crate::prelude_dev::*;

impl<T, D> DeviceIndexSelectAPI<T, D> for DeviceCpuSerial
where
    T: Clone,
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
        index_select_cpu_serial(c, lc, a, la, axis, indices)
    }
}
